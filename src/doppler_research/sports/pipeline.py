"""Causal event panel, expanding event-time walk-forward and audit traces.

Leakage rules enforced here:
* features use only quote messages *received* at or before the origin;
* targets use only the replayed path strictly after the origin;
* a row may train a later forecast only if its whole target window ended
  before that forecast's origin (fully observed);
* comparable-news means use the same fully-observed rule.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

import numpy as np
import pandas as pd

from .audit import MODEL_REGISTRY, PKG, Trace, ns_iso
from .config import SportsConfig
from .events import (band, choose_revision, fixture_news_times, forecast_origin, is_inactive,
                     link_fixture, selection_role)
from .models import ALL_MODELS, BASELINE_MODELS, RIDGE_MODELS, comparable_history, fit_ridge
from .probability import logit
from .replay import NS, MarketReplay, build_replays, logit_path
from .schema import validate_all
from .spectral import FEATURES as DOPPLER, FeatureUnavailable, WindowSpec, doppler_features
from .targets import TargetUnavailable, realized_targets
from .volatility import ewma_variance, horizon_garch, trailing_rv

EXCLUSION_COLUMNS = ["panel", "news_id", "event_key", "horizon_minutes", "reason", "detail"]


@dataclass
class Prepared:
    tables: dict[str, pd.DataFrame]
    replays: dict[str, MarketReplay]
    news_times: dict[str, list[tuple[int, str]]]


@dataclass
class PanelResult:
    panel: pd.DataFrame
    exclusions: pd.DataFrame
    traces: dict[str, Trace]
    grid_lineage: list[dict] = field(default_factory=list)
    target_lineage: list[dict] = field(default_factory=list)


def prepare(frames: dict[str, pd.DataFrame], config: SportsConfig) -> Prepared:
    tables = validate_all(frames["markets"], frames["quotes"], frames["news_events"],
                          frames["fixtures_results"], config.max_clock_skew_seconds)
    replays = build_replays(tables["markets"], tables["quotes"], config.max_quote_age_seconds)
    return Prepared(tables, replays, fixture_news_times(tables["news_events"], tables["fixtures_results"]))


def window_spec(config: SportsConfig) -> WindowSpec:
    spm = config.steps_per_minute
    return WindowSpec(window_steps=int(round(config.recent_window_minutes * spm)),
                      baseline_windows=config.baseline_windows,
                      dynamics_steps=max(1, int(round(config.dynamics_step_minutes * spm))),
                      cadence_minutes=config.cadence_seconds / 60.0,
                      min_coverage=config.min_window_coverage,
                      scale_grid=config.scale_factor_grid)


def horizon_steps(config: SportsConfig, minutes: int) -> int:
    return int(round(minutes * config.steps_per_minute))


def origin_features(replay: MarketReplay, selection_id: str, origin_ns: int, config: SportsConfig,
                    trace: Trace, lineage: list[dict] | None = None) -> dict:
    """All pre-origin features. Raises FeatureUnavailable with an exclusion reason."""
    spec = window_spec(config)
    garch_steps = int(round(config.garch_fit_minutes * config.steps_per_minute))
    trailing_steps = int(round(config.trailing_rv_minutes * config.steps_per_minute))
    n = max(spec.required_increments, garch_steps, trailing_steps)
    state = replay.grid(origin_ns, n, config.cadence_seconds, selection_id)
    x = logit_path(state, config.probability_basis, config.logit_epsilon)
    inc_valid = state.valid[1:] & state.valid[:-1]
    dx = np.where(inc_valid, np.diff(x), np.nan)
    trace.step(f"{PKG}.replay.MarketReplay.grid",
               {"market_key": replay.market_key, "selection_id": selection_id, "end_utc": ns_iso(origin_ns),
                "cadence_seconds": config.cadence_seconds, "grid_points": n + 1,
                "rule": "last OPEN quote received <= grid time; invalid across breaks/staleness/kickoff"},
               {"first_grid_utc": ns_iso(int(state.times_ns[0])), "valid_points": int(state.valid.sum()),
                "invalid_points": int((~state.valid).sum()),
                "invalid_reasons": pd.Series(state.reason[~state.valid]).value_counts().to_dict(),
                "distinct_quote_rows_used": int(len(set(state.source_row[state.valid].tolist()))),
                "origin_quote_source_row": int(state.source_row[-1]),
                "basis": config.probability_basis})
    if lineage is not None:
        w, k = spec.window_steps, spec.baseline_windows
        for i in range(len(state.times_ns)):
            back = len(state.times_ns) - 1 - i
            seg = ("origin" if back == 0 else "recent" if back <= w else
                   f"baseline_{(back - 1) // w}" if back <= w * (k + 1) else "history_only")
            lineage.append({"event_key": trace.event_key, "panel": trace.panel,
                            "grid_time_utc": ns_iso(int(state.times_ns[i])), "segment": seg,
                            "valid": bool(state.valid[i]), "reason": state.reason[i],
                            "probability": state.probability(config.probability_basis)[i],
                            "logit": x[i], "increment": dx[i - 1] if i else np.nan,
                            "quote_source_row": int(state.source_row[i])})
    if not state.valid[-1]:
        raise FeatureUnavailable(f"origin_quote_invalid:{state.reason[-1]}")
    spread, depth = float(state.spread[-1]), float(state.depth[-1])
    if spread > config.max_spread_probability or depth < config.min_depth:
        raise FeatureUnavailable("thin_or_wide_book_at_origin")

    spectral = doppler_features(dx[-spec.required_increments:], inc_valid[-spec.required_increments:], spec)
    detail = spectral.pop("_detail")
    trace.step(f"{PKG}.spectral.doppler_features",
               {"increments": spec.required_increments, "window_minutes": config.recent_window_minutes,
                "baseline_windows": spec.baseline_windows, "dynamics_step_minutes": config.dynamics_step_minutes,
                "scale_factor_grid": list(config.scale_factor_grid)},
               {**{k: spectral[k] for k in DOPPLER}, "scale_factor": spectral["scale_factor"],
                "scale_fit_gain_js": spectral["scale_fit_gain_js"],
                "scale_fit_at_grid_edge": spectral["scale_fit_at_grid_edge"], **detail})
    trail = trailing_rv(dx, trailing_steps, 1)
    ewma = ewma_variance(dx[-garch_steps:], config.ewma_lambda, 1)
    steps = [horizon_steps(config, h) for h in config.horizons_minutes]
    garch = horizon_garch(dx[-garch_steps:], steps, config.min_garch_observations)
    trace.step(f"{PKG}.volatility.trailing_rv", {"window_minutes": config.trailing_rv_minutes}, trail)
    trace.step(f"{PKG}.volatility.ewma_variance", {"lambda": config.ewma_lambda,
                                                   "window_minutes": config.garch_fit_minutes}, ewma)
    trace.step(f"{PKG}.volatility.horizon_garch",
               {"fit_window_minutes": config.garch_fit_minutes, "horizon_steps": steps,
                "min_observations": config.min_garch_observations}, garch)
    minutes_to_ko = (replay.scheduled_start_ns - origin_ns) / NS / 60
    p0 = float(state.probability(config.probability_basis)[-1])
    out = {key: spectral[key] for key in DOPPLER}
    out.update({"scale_factor": spectral["scale_factor"], "scale_fit_gain_js": spectral["scale_fit_gain_js"],
                "feature_effective_increments": detail["effective_increments"],
                "feature_min_window_coverage": min(detail["window_coverage_oldest_to_recent"]),
                "p_origin": p0, "spread_origin": spread, "depth_origin": depth,
                "minutes_to_kickoff": minutes_to_ko,
                "log_trailing_rv": float(np.log(max(trail.get("realized_variation_trailing", np.nan), 1e-12)))
                if trail["ok"] else np.nan,
                "log_spread": float(np.log(max(spread, 1e-6))), "log_depth": float(np.log1p(depth)),
                "log_minutes_to_kickoff": float(np.log(max(minutes_to_ko, 1e-3))),
                "abs_logit_origin": float(abs(logit(p0, config.logit_epsilon))),
                "trailing_one_step_variance": trail.get("mean_squared_increment", np.nan),
                "ewma_one_step_variance": ewma.get("one_step_variance", np.nan),
                "garch_ok": garch["ok"], "garch_failure_reason": garch["failure_reason"],
                "garch_omega_pct2": garch.get("omega_pct2", np.nan), "garch_alpha": garch.get("alpha", np.nan),
                "garch_beta": garch.get("beta", np.nan), "garch_mu": garch.get("mu", np.nan),
                "garch_fit_observations": garch["fit_observations"],
                "_garch_forecasts": garch["forecasts"]})
    out["liquidity_band"] = band(depth, config.liquidity_band_edges, "depth")
    out["kickoff_band"] = band(minutes_to_ko, config.kickoff_band_edges_minutes, "ko_min")
    trace.step(f"{PKG}.pipeline.origin_features",
               {"rule": "state controls frozen at origin"},
               {k: out[k] for k in ("p_origin", "spread_origin", "depth_origin", "minutes_to_kickoff",
                                    "log_trailing_rv", "log_spread", "log_depth", "log_minutes_to_kickoff",
                                    "abs_logit_origin", "liquidity_band", "kickoff_band")})
    return out


def _shuffled_fixture(prep: Prepared, config: SportsConfig, news_id: str, actual_fixture: str,
                      origin_ns: int) -> str | None:
    """Seeded placebo: a different fixture whose market was live with enough history at the origin."""
    history_ns = int(max(config.garch_fit_minutes,
                         config.recent_window_minutes * (config.baseline_windows + 1) + 2 * config.dynamics_step_minutes)
                     * 60 * NS)
    latest_end = origin_ns + min(config.horizons_minutes) * 60 * NS
    live = sorted({r.fixture_id for r in prep.replays.values()
                   if r.fixture_id != actual_fixture and r.market_type in config.market_types
                   and r.first_message_ns() is not None and r.first_message_ns() <= origin_ns - history_ns
                   and r.scheduled_start_ns > latest_end})
    if not live:
        return None
    digest = int(hashlib.sha256(news_id.encode("utf-8")).hexdigest()[:8], 16)
    return live[int(np.random.default_rng([config.seed, digest]).integers(len(live)))]


def build_panel(prep: Prepared, config: SportsConfig, panel_name: str = "main",
                origin_shift_minutes: float = 0.0, shuffled_links: bool = False,
                keep_lineage: bool = True) -> PanelResult:
    t = prep.tables
    news, fixtures, markets = t["news_events"], t["fixtures_results"], t["markets"]
    rows, excl, traces = [], [], {}
    grid_lineage: list[dict] = []
    target_lineage: list[dict] = []

    def exclude(news_id: str, reason: str, event_key: str = "", horizon=None, detail: str = "") -> None:
        excl.append({"panel": panel_name, "news_id": news_id, "event_key": event_key,
                     "horizon_minutes": horizon, "reason": reason, "detail": detail})

    candidates = []
    for news_id, revisions in news.groupby("news_id", sort=True):
        chosen, reason, earlier = choose_revision(revisions, config)
        if chosen is None:
            exclude(news_id, reason); continue
        if config.eligible_event_types is not None and chosen["event_type"] not in config.eligible_event_types:
            exclude(news_id, "event_type_not_eligible", detail=str(chosen["event_type"])); continue
        origin, reason, rule = forecast_origin(chosen, config)
        if origin is None:
            exclude(news_id, reason); continue
        if origin_shift_minutes:
            origin = origin - pd.Timedelta(minutes=origin_shift_minutes)
            rule = {**rule, "placebo_shift_minutes": -origin_shift_minutes}
        fixture, link_type, reason = link_fixture(chosen, origin, fixtures)
        if fixture is None:
            exclude(news_id, reason); continue
        role, reason = selection_role(chosen, fixture, config)
        if role is None:
            exclude(news_id, reason); continue
        if shuffled_links:
            other = _shuffled_fixture(prep, config, news_id, fixture["fixture_id"], int(origin.value))
            if other is None:
                exclude(news_id, "no_live_fixture_for_shuffled_link"); continue
            fixture = fixtures[fixtures["fixture_id"] == other].iloc[0]
            link_type = "shuffled_placebo_link"
        if is_inactive(fixture):
            exclude(news_id, "fixture_postponed_or_void", detail=str(fixture["fixture_status"])); continue
        mk = markets[(markets["fixture_id"] == fixture["fixture_id"]) &
                     (markets["market_type"].isin(config.market_types))]
        if mk.empty:
            exclude(news_id, "no_market_for_fixture"); continue
        candidates.append((news_id, chosen, earlier, origin, rule, fixture, link_type, role,
                           sorted(mk["market_key"].unique())))

    # Two eligible news items on one fixture with near-identical origins cannot be separated.
    by_fixture: dict[str, list[tuple[pd.Timestamp, str]]] = {}
    for c in candidates:
        by_fixture.setdefault(c[5]["fixture_id"], []).append((c[3], c[0]))
    window = pd.Timedelta(seconds=config.simultaneous_news_seconds)
    simultaneous = set()
    for items in by_fixture.values():
        for o1, n1 in items:
            for o2, n2 in items:
                if n1 != n2 and abs(o1 - o2) <= window:
                    simultaneous.add(n1)

    for news_id, chosen, earlier, origin, rule, fixture, link_type, role, market_keys in candidates:
        if news_id in simultaneous:
            exclude(news_id, "simultaneous_news"); continue
        origin_ns = int(origin.value)
        for market_key in market_keys:
            replay = prep.replays[market_key]
            mrow = markets[markets["market_key"] == market_key].iloc[0]
            selection = replay.selection_for_role(role)
            event_key = f"{news_id}|{market_key}|{selection}"
            if selection is None or not replay.complete_roles:
                exclude(news_id, "missing_runner", event_key); continue
            if mrow["market_created_utc"] > origin:
                exclude(news_id, "market_not_listed_at_origin", event_key); continue
            if origin_ns >= replay.scheduled_start_ns:
                exclude(news_id, "origin_not_pre_match", event_key); continue
            trace = Trace(event_key, panel_name)
            trace.step(f"{PKG}.events.choose_revision",
                       {"news_id": news_id, "revisions_supplied": int((news["news_id"] == news_id).sum()),
                        "eligible_confidence": list(config.eligible_confidence)},
                       {"revision_id": chosen["revision_id"], "event_type": chosen["event_type"],
                        "confidence": chosen["confidence"], "news_source_row": int(chosen["source_row"]),
                        "raw_hash": chosen["raw_hash"], "earlier_revisions": earlier})
            trace.step(f"{PKG}.events.forecast_origin", {"mode": config.mode, **rule},
                       {"origin_utc": origin})
            trace.step(f"{PKG}.events.link_fixture",
                       {"subject_team_id": chosen["subject_team_id"], "news_fixture_id": chosen["fixture_id"]},
                       {"fixture_id": fixture["fixture_id"], "link_type": link_type, "market_key": market_key,
                        "selection_role": role, "selection_id": selection,
                        "scope": "marginal_single_selection (not a coherent 3-outcome surface)"})
            lineage = grid_lineage if keep_lineage else None
            try:
                feats = origin_features(replay, selection, origin_ns, config, trace, lineage)
            except FeatureUnavailable as exc:
                exclude(news_id, str(exc), event_key); continue
            garch_by_steps = feats.pop("_garch_forecasts")
            others = [ns for ns, nid in prep.news_times.get(fixture["fixture_id"], []) if nid != news_id]
            base = {"panel": panel_name, "event_key": event_key, "news_id": news_id,
                    "revision_id": chosen["revision_id"], "event_type": chosen["event_type"],
                    "confidence": chosen["confidence"], "subject_team_id": chosen["subject_team_id"],
                    "fixture_id": fixture["fixture_id"], "competition_id": mrow["competition_id"],
                    "market_key": market_key, "venue": mrow["venue"], "market_type": mrow["market_type"],
                    "selection_id": selection, "selection_role": role, "link_type": link_type,
                    "origin_utc": origin, "origin_ns": origin_ns,
                    "scheduled_start_utc": mrow["scheduled_start_utc"], **feats}
            any_horizon = False
            for h in config.horizons_minutes:
                end_ns = origin_ns + h * 60 * NS
                if any(origin_ns < ns <= end_ns for ns in others):
                    exclude(news_id, "later_news_inside_target_window", event_key, h); continue
                try:
                    tgt = realized_targets(replay, selection, origin_ns, h, config.probability_basis,
                                           config.logit_epsilon, config.target_sampling,
                                           config.target_suspension_policy, config.cadence_seconds)
                except TargetUnavailable as exc:
                    exclude(news_id, str(exc), event_key, h); continue
                alt_sampling = "grid" if config.target_sampling == "tick" else "tick"
                try:
                    alt = realized_targets(replay, selection, origin_ns, h, config.probability_basis,
                                           config.logit_epsilon, alt_sampling,
                                           config.target_suspension_policy, config.cadence_seconds)
                    alt_rv = alt["realized_variation"]
                except TargetUnavailable:
                    alt_rv = np.nan
                lin = tgt.pop("_lineage")
                if keep_lineage:
                    for tn, lx, sr in zip(lin["times_ns"], lin["logit"], lin["source_row"]):
                        target_lineage.append({"event_key": event_key, "panel": panel_name, "horizon_minutes": h,
                                               "time_utc": ns_iso(tn), "logit": lx, "quote_source_row": sr})
                trace.horizon_step(h, f"{PKG}.targets.realized_targets",
                                   {"window": f"({ns_iso(origin_ns)}, {ns_iso(end_ns)}]",
                                    "sampling": config.target_sampling,
                                    "suspension_policy": config.target_suspension_policy,
                                    "quote_source_rows": lin["source_row"]},
                                   {k: v for k, v in tgt.items()})
                trace.horizon(h)["target_name"] = config.target
                steps = horizon_steps(config, h)
                rows.append({**base, "horizon_minutes": h, "horizon_steps": steps, "target_end_ns": tgt["target_end_ns"],
                             "target_end_utc": pd.Timestamp(tgt["target_end_ns"], tz="UTC"),
                             **{k: v for k, v in tgt.items() if k != "target_end_ns"},
                             f"realized_variation_{alt_sampling}_sampled": alt_rv,
                             "garch_horizon": garch_by_steps[steps],
                             "trailing_rv_horizon": feats["trailing_one_step_variance"] * steps,
                             "ewma_horizon": feats["ewma_one_step_variance"] * steps})
                any_horizon = True
            if any_horizon:
                traces[event_key] = trace
    panel = pd.DataFrame(rows)
    if not panel.empty:
        panel = panel.sort_values(["horizon_minutes", "origin_ns", "event_key"]).reset_index(drop=True)
    return PanelResult(panel, pd.DataFrame(excl, columns=EXCLUSION_COLUMNS), traces, grid_lineage, target_lineage)


def _comparables(panel_h: pd.DataFrame, config: SportsConfig) -> list[dict]:
    """For each row, the comparable earlier fully observed events (by event time)."""
    keys = sorted({k for level in config.comparable_fallback for k in level} | {"event_key"})
    arrays = {k: panel_h[k].astype(str).to_numpy() for k in keys}
    target = panel_h[config.target].to_numpy(dtype=float)
    garch = panel_h["garch_horizon"].to_numpy(dtype=float)
    ends = panel_h["target_end_ns"].to_numpy()
    out = []
    for i, row in enumerate(panel_h.itertuples(index=False)):
        prior_mask = ends < row.origin_ns
        prior = {k: v[prior_mask] for k, v in arrays.items()}
        comp = comparable_history(prior, {k: str(getattr(row, k)) for k in keys},
                                  config.comparable_fallback, config.min_comparable_events)
        y = target[prior_mask][comp["mask"]]
        g = garch[prior_mask][comp["mask"]]
        ok_ratio = np.isfinite(g) & (g > 0)
        out.append({"level": comp["level"], "level_keys": comp["level_keys"],
                    "event_keys": prior["event_key"][comp["mask"]].tolist(),
                    "mean": float(y.mean()) if len(y) else np.nan,
                    "median": float(np.median(y)) if len(y) else np.nan,
                    "garch_ratio": float(np.mean(y[ok_ratio] / g[ok_ratio])) if ok_ratio.sum() >= config.min_comparable_events
                    else np.nan})
    return out


def _forecast_row(row: dict, train: pd.DataFrame, comp: dict, config: SportsConfig) -> dict[str, dict]:
    """Every model's forecast for one row, with an audit decomposition."""
    floor = config.forecast_floor
    h = row["horizon_minutes"]
    out: dict[str, dict] = {}
    out["trailing_rv"] = {"forecast": row["trailing_rv_horizon"], "detail": {
        "one_step_variance": row["trailing_one_step_variance"], "horizon_steps": row["horizon_steps"],
        "rule": "mean squared grid increment over trailing window x horizon steps"}}
    out["ewma"] = {"forecast": row["ewma_horizon"], "detail": {
        "one_step_variance": row["ewma_one_step_variance"], "lambda": config.ewma_lambda,
        "horizon_steps": row["horizon_steps"]}}
    out["garch_horizon"] = {"forecast": row["garch_horizon"], "detail": {
        "omega_pct2": row["garch_omega_pct2"], "alpha": row["garch_alpha"], "beta": row["garch_beta"],
        "mu": row["garch_mu"], "fit_observations": row["garch_fit_observations"],
        "horizon_steps": row["horizon_steps"], "failure_reason": row["garch_failure_reason"],
        "rule": "sum_k (h_k + mu^2)/1e4 over horizon steps"}}
    comp_detail = {"fallback_level": comp["level"], "level_keys": comp["level_keys"],
                   "n_comparable_events": len(comp["event_keys"]), "comparable_event_keys": comp["event_keys"]}
    out["comparable_event_mean"] = {"forecast": comp["mean"], "detail": comp_detail}
    out["comparable_event_median"] = {"forecast": comp["median"], "detail": comp_detail}
    out["event_adjusted_garch"] = {"forecast": row["garch_horizon"] * comp["garch_ratio"], "detail": {
        "garch_horizon": row["garch_horizon"], "mean_target_to_garch_ratio": comp["garch_ratio"], **comp_detail}}
    for name in BASELINE_MODELS:
        out[name]["function"] = MODEL_REGISTRY[name]
    for name, cols in RIDGE_MODELS.items():
        x = np.array([row[c] for c in cols], dtype=float)
        tr = train[cols].to_numpy(dtype=float)
        ok = np.isfinite(tr).all(axis=1)
        if not np.isfinite(x).all() or ok.sum() < config.min_train_events:
            out[name] = {"forecast": np.nan, "function": f"{PKG}.models.fit_ridge",
                         "detail": {"unavailable": "nonfinite feature or insufficient finite training rows"}}
            continue
        fit = fit_ridge(tr[ok], train[config.target].to_numpy(dtype=float)[ok], cols, config.ridge_alpha, floor)
        ex = fit.explain(x)
        out[name] = {"forecast": ex["forecast"], "function": f"{PKG}.models.fit_ridge -> RidgeFit.explain",
                     "intercept": ex["intercept"], "terms": ex["terms"], "log_forecast": ex["log_forecast"],
                     "clipped": ex["clipped"], "n_train": fit.n_train, "ridge_alpha": fit.alpha,
                     "train_event_keys": train["event_key"][ok].tolist()}
    return out


def walk_forward(result: PanelResult, config: SportsConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (per-event forecasts, contribution rows, forecast-level skips)."""
    panel, traces = result.panel, result.traces
    records, contributions, skips = [], [], []
    if panel.empty:
        return pd.DataFrame(records), pd.DataFrame(contributions), pd.DataFrame(skips)
    panel = panel.copy()
    for h, part in panel.groupby("horizon_minutes", sort=True):
        part = part.sort_values(["origin_ns", "event_key"]).reset_index(drop=True)
        comps = _comparables(part, config)
        part["log_comparable_event_mean"] = [np.log(max(c["mean"], config.forecast_floor))
                                             if np.isfinite(c["mean"]) else np.nan for c in comps]
        ends = part["target_end_ns"].to_numpy()
        for i, row in enumerate(part.to_dict("records")):
            train = part[ends < row["origin_ns"]]
            if len(train) < config.min_train_events:
                skips.append({"panel": row["panel"], "event_key": row["event_key"], "horizon_minutes": h,
                              "reason": "insufficient_fully_observed_training_events",
                              "detail": f"{len(train)} < {config.min_train_events}"})
                continue
            models = _forecast_row(row, train, comps[i], config)
            trace = traces[row["event_key"]]
            trace.horizon_step(h, f"{PKG}.pipeline.walk_forward",
                               {"rule": "train only on rows whose target window ended before this origin"},
                               {"training_events": len(train),
                                "latest_training_target_end_utc": ns_iso(int(train["target_end_ns"].max()))})
            observed = float(row[config.target])
            base = {k: row[k] for k in ("panel", "event_key", "news_id", "event_type", "fixture_id",
                                         "competition_id", "market_key", "venue", "market_type", "selection_id",
                                         "selection_role", "origin_utc", "target_end_utc", "kickoff_band",
                                         "liquidity_band", "minutes_to_kickoff", "p_origin", "signed_reprice")}
            for name in ALL_MODELS:
                m = models[name]
                f = m["forecast"]
                if not np.isfinite(f) or f <= 0:
                    skips.append({"panel": row["panel"], "event_key": row["event_key"], "horizon_minutes": h,
                                  "reason": f"model_unavailable:{name}",
                                  "detail": str(m.get("detail", {}).get("failure_reason") or
                                                m.get("detail", {}).get("unavailable") or "nonfinite forecast")})
                    trace.horizon(h)["models"][name] = {**m, "forecast": None, "realized": observed}
                    continue
                fv = max(float(f), config.forecast_floor)
                qlike = float(np.log(fv) + observed / fv)
                m = {**m, "forecast": fv, "realized": observed, "qlike": qlike, "error": fv - observed}
                trace.horizon(h)["models"][name] = m
                records.append({**base, "horizon_minutes": h, "model": name, "target": config.target,
                                "forecast": fv, "realized": observed, "train_events": len(train),
                                "comparable_level": comps[i]["level"],
                                "comparable_events": len(comps[i]["event_keys"])})
                if "terms" in m:
                    contributions.append({"panel": row["panel"], "event_key": row["event_key"], "horizon_minutes": h,
                                          "model": name, "term": "intercept", "raw_value": np.nan,
                                          "train_mean": np.nan, "train_scale": np.nan, "standardized": np.nan,
                                          "coefficient": np.nan, "contribution_log": m["intercept"],
                                          "source_function": f"{PKG}.models.fit_ridge"})
                    for term in m["terms"]:
                        contributions.append({"panel": row["panel"], "event_key": row["event_key"],
                                              "horizon_minutes": h, "model": name, **term,
                                              "source_function": _feature_function(term["term"])})
                else:
                    for k, v in m["detail"].items():
                        if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool):
                            contributions.append({"panel": row["panel"], "event_key": row["event_key"],
                                                  "horizon_minutes": h, "model": name, "term": k,
                                                  "raw_value": float(v), "source_function": m["function"]})
                contributions.append({"panel": row["panel"], "event_key": row["event_key"], "horizon_minutes": h,
                                      "model": name, "term": "FORECAST", "raw_value": fv,
                                      "source_function": m["function"]})
    return pd.DataFrame(records), pd.DataFrame(contributions), pd.DataFrame(
        skips, columns=["panel", "event_key", "horizon_minutes", "reason", "detail"])


def _feature_function(name: str) -> str:
    from .audit import FEATURE_REGISTRY
    return FEATURE_REGISTRY.get(name, {}).get("function", "")


def forecast_at(prep: Prepared, config: SportsConfig, market_key: str, selection_id: str | None,
                origin: pd.Timestamp, horizon_minutes: int, event_type: str,
                history: PanelResult | None = None) -> tuple[pd.DataFrame, Trace]:
    """Forecast for a chosen market/selection/origin using only data received by then.

    Training rows are earlier fully observed events from the supplied files.
    No realized target is computed even if later quotes exist in the input.
    """
    if market_key not in prep.replays:
        raise ValueError(f"unknown market {market_key}")
    if horizon_minutes not in config.horizons_minutes:
        config = SportsConfig.from_dict({**config.to_dict(),
                                         "horizons_minutes": tuple(sorted({*config.horizons_minutes,
                                                                           horizon_minutes}))})
    replay = prep.replays[market_key]
    if selection_id is None:
        raise ValueError("selection_id is required")
    origin = pd.Timestamp(origin)
    if origin.tzinfo is None:
        raise ValueError("origin must include a timezone, e.g. 2026-11-16T14:00:00Z")
    origin_ns = int(origin.tz_convert("UTC").value)
    history = history or build_panel(prep, config, keep_lineage=False)
    mrow = prep.tables["markets"][prep.tables["markets"]["market_key"] == market_key].iloc[0]
    trace = Trace(f"adhoc|{market_key}|{selection_id}|{origin.isoformat()}", "adhoc_forecast")
    trace.step(f"{PKG}.pipeline.forecast_at", {"market_key": market_key, "selection_id": selection_id,
                                                "origin_utc": origin, "event_type": event_type,
                                                "horizon_minutes": horizon_minutes},
               {"note": "only quotes received <= origin are used; no realized target is computed"})
    feats = origin_features(replay, selection_id, origin_ns, config, trace)
    garch = feats.pop("_garch_forecasts")
    steps = horizon_steps(config, horizon_minutes)
    row = {"panel": "adhoc_forecast", "event_key": trace.event_key, "event_type": event_type,
           "market_type": mrow["market_type"], "competition_id": mrow["competition_id"],
           "horizon_minutes": horizon_minutes, "horizon_steps": steps, "origin_ns": origin_ns,
           "garch_horizon": garch[steps], "trailing_rv_horizon": feats["trailing_one_step_variance"] * steps,
           "ewma_horizon": feats["ewma_one_step_variance"] * steps, "target_end_ns": np.iinfo(np.int64).max,
           config.target: np.nan, **feats}
    part = history.panel[history.panel["horizon_minutes"] == horizon_minutes] if not history.panel.empty \
        else history.panel
    part = part[part["target_end_ns"] < origin_ns] if not part.empty else part
    if len(part) < config.min_train_events:
        raise ValueError(f"only {len(part)} fully observed training events before origin; "
                         f"need {config.min_train_events}")
    joined = pd.concat([part, pd.DataFrame([row])], ignore_index=True).sort_values(["origin_ns", "event_key"])
    joined = joined.reset_index(drop=True)
    comps = _comparables(joined, config)
    joined["log_comparable_event_mean"] = [np.log(max(c["mean"], config.forecast_floor))
                                           if np.isfinite(c["mean"]) else np.nan for c in comps]
    i = int(np.flatnonzero(joined["event_key"].to_numpy() == trace.event_key)[0])
    train = joined[joined["target_end_ns"] < origin_ns]
    models = _forecast_row(joined.iloc[i].to_dict(), train, comps[i], config)
    records = []
    for name in ALL_MODELS:
        m = models[name]
        trace.horizon(horizon_minutes)["models"][name] = m
        trace.horizon(horizon_minutes)["target_name"] = config.target
        records.append({"market_key": market_key, "selection_id": selection_id, "origin_utc": origin,
                        "horizon_minutes": horizon_minutes, "event_type": event_type, "model": name,
                        "target": config.target, "forecast": m["forecast"], "train_events": len(train),
                        "forecast_scope": "marginal_single_selection",
                        "interpretation": "expected realized variation of logit(implied probability); "
                                          "not a direction, odds point or match-result probability"})
    return pd.DataFrame(records), trace
