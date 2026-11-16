"""Common-sample scores, clustered paired bootstrap and acceptance-gate checks."""
from __future__ import annotations

import numpy as np
import pandas as pd

PAIRS = [
    ("doppler_only", "garch_horizon"),
    ("doppler_only", "trailing_rv"),
    ("state_doppler", "state_only"),
    ("event_state_doppler", "event_state"),
    ("event_state_doppler", "event_adjusted_garch"),
    ("event_state_doppler", "comparable_event_mean"),
    ("event_state", "event_adjusted_garch"),
    ("event_state_doppler", "event_state_doppler_minus_js"),
    ("event_state_doppler", "event_state_doppler_minus_centroid"),
    ("event_state_doppler", "event_state_doppler_minus_scale"),
    ("event_state_doppler", "event_state_doppler_minus_energy"),
]
GATE_MODEL = "event_state_doppler"
GATE_COMPARATORS = ("event_state", "event_adjusted_garch")
BREAKDOWNS = ["event_type", "competition_id", "kickoff_band", "liquidity_band", "venue"]


def common_sample(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Keep event/horizon rows forecast by every model that appears for that horizon."""
    parts = []
    for (_, h), part in forecasts.groupby(["panel", "horizon_minutes"], sort=True):
        models = set(part["model"])
        counts = part.groupby("event_key")["model"].nunique()
        keep = counts[counts == len(models)].index
        parts.append(part[part["event_key"].isin(keep)])
    return pd.concat(parts, ignore_index=True) if parts else forecasts.iloc[0:0]


def add_losses(f: pd.DataFrame) -> pd.DataFrame:
    f = f.copy()
    p, y = f["forecast"].to_numpy(dtype=float), f["realized"].to_numpy(dtype=float)
    f["qlike_loss"] = np.log(p) + y / p
    f["absolute_error"] = np.abs(p - y)
    f["squared_error"] = (p - y) ** 2
    f["error"] = p - y
    return f


def _summary(f: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    rows = []
    for key, part in f.groupby(group, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(group, key)), "target": part["target"].iloc[0],
                     "target_units": "squared logit change summed over the horizon",
                     "n_events": part["event_key"].nunique(), "n_fixtures": part["fixture_id"].nunique(),
                     "qlike": part["qlike_loss"].mean(), "mae": part["absolute_error"].mean(),
                     "rmse": float(np.sqrt(part["squared_error"].mean())), "bias": part["error"].mean(),
                     "mean_forecast": part["forecast"].mean(), "mean_realized": part["realized"].mean()})
    return pd.DataFrame(rows)


def _cluster(f: pd.DataFrame, how: str) -> pd.Series:
    if how == "fixture":
        return f["fixture_id"].astype(str)
    return pd.to_datetime(f["origin_utc"], utc=True).dt.strftime("%G-W%V")


def paired_bootstrap(f: pd.DataFrame, replicates: int, seed: int, cluster: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for (panel, h), part in f.groupby(["panel", "horizon_minutes"], sort=True):
        wide = part.pivot(index="event_key", columns="model", values="qlike_loss")
        meta = part.drop_duplicates("event_key").set_index("event_key")
        clusters = _cluster(meta.reset_index(), cluster).set_axis(meta.index)
        for a, b in PAIRS:
            if a not in wide or b not in wide:
                continue
            delta = (wide[a] - wide[b]).dropna()
            groups = [delta[clusters.loc[delta.index] == c].to_numpy() for c in sorted(clusters.loc[delta.index].unique())]
            if len(groups) >= 2:
                draws = np.empty(replicates)
                for i in range(replicates):
                    pick = rng.integers(0, len(groups), size=len(groups))
                    draws[i] = np.concatenate([groups[k] for k in pick]).mean()
                lo, hi = np.quantile(draws, [0.025, 0.975])
            else:
                lo = hi = np.nan
            rows.append({"panel": panel, "horizon_minutes": h, "model_a": a, "model_b": b,
                         "n_events": len(delta), "n_clusters": len(groups), "cluster_unit": cluster,
                         "mean_qlike_delta_a_minus_b": float(delta.mean()), "bootstrap_ci_low": lo,
                         "bootstrap_ci_high": hi, "fraction_a_better": float((delta < 0).mean()),
                         "replicates": replicates, "seed": seed})
    return pd.DataFrame(rows)


def gate_checks(comparisons: pd.DataFrame) -> pd.DataFrame:
    """Spec gate: a Doppler benefit must beat event+state AND event-adjusted GARCH."""
    rows = []
    if comparisons.empty:
        return pd.DataFrame(rows)
    for (panel, h), part in comparisons.groupby(["panel", "horizon_minutes"], sort=True):
        checks = {}
        for b in GATE_COMPARATORS:
            r = part[(part["model_a"] == GATE_MODEL) & (part["model_b"] == b)]
            checks[b] = bool(len(r) and r["bootstrap_ci_high"].iloc[0] < 0)
        rows.append({"panel": panel, "horizon_minutes": h, "model": GATE_MODEL,
                     **{f"beats_{b}_ci_below_zero": v for b, v in checks.items()},
                     "gate": "pass" if all(checks.values()) else "not_demonstrated",
                     "note": "exploratory unless the panel is an untouched held-out sample"})
    return pd.DataFrame(rows)


def signed_report(panel: pd.DataFrame) -> pd.DataFrame:
    """Signed repricing summary vs a zero-drift (no-change quote) baseline."""
    rows = []
    for (name, h), part in panel.groupby(["panel", "horizon_minutes"], sort=True):
        s = part["signed_reprice"].to_numpy(dtype=float)
        rows.append({"panel": name, "horizon_minutes": h, "n_events": len(s), "mean_signed_reprice": s.mean(),
                     "zero_drift_mae": np.abs(s).mean(), "zero_drift_rmse": float(np.sqrt(np.mean(s ** 2))),
                     "share_up": float((s > 0).mean()), "share_down": float((s < 0).mean()),
                     "share_unchanged": float((s == 0).mean()),
                     "note": "no directional model is fitted; volatility forecasts are sign-agnostic"})
    return pd.DataFrame(rows)


def event_counts(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ["event_type", "market_type", "competition_id", "kickoff_band", "liquidity_band", "venue"]:
        for (name, h, value), part in panel.groupby(["panel", "horizon_minutes", col], sort=True):
            rows.append({"panel": name, "horizon_minutes": h, "dimension": col, "value": value,
                         "n_events": part["event_key"].nunique(), "n_fixtures": part["fixture_id"].nunique(),
                         "median_feature_coverage": part["feature_min_window_coverage"].median()})
    return pd.DataFrame(rows)


def score(forecasts: pd.DataFrame, replicates: int, seed: int, cluster: str
          ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (scored rows, pooled scores, breakdown scores, bootstrap comparisons, gate checks)."""
    if forecasts.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty
    f = add_losses(common_sample(forecasts))
    pooled = _summary(f, ["panel", "horizon_minutes", "model"]).sort_values(
        ["panel", "horizon_minutes", "qlike"]).reset_index(drop=True)
    breakdown = []
    for col in BREAKDOWNS:
        b = _summary(f, ["panel", "horizon_minutes", col, "model"]).rename(columns={col: "group_value"})
        b.insert(2, "group_by", col)
        breakdown.append(b)
    comparisons = paired_bootstrap(f, replicates, seed, cluster)
    return f, pooled, pd.concat(breakdown, ignore_index=True), comparisons, gate_checks(comparisons)
