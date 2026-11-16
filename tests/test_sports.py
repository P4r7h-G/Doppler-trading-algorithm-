from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from doppler_research.sports.audit import explain, read_traces
from doppler_research.sports.config import SportsConfig
from doppler_research.sports.events import choose_revision, forecast_origin
from doppler_research.sports.models import fit_ridge
from doppler_research.sports.pipeline import build_panel, forecast_at, prepare, walk_forward
from doppler_research.sports.replay import NS, MarketReplay
from doppler_research.sports.schema import SchemaError, parse_utc, validate_markets, validate_news, validate_quotes
from doppler_research.sports.spectral import WindowSpec, doppler_features, snapshot
from doppler_research.sports.study import run_study
from doppler_research.sports.synthetic import generate
from doppler_research.sports.targets import TargetUnavailable, realized_targets
from doppler_research.sports.volatility import horizon_garch

KO = pd.Timestamp("2026-09-01T15:00:00Z")


def iso(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def mini_markets(roles=("home", "draw", "away")) -> pd.DataFrame:
    return pd.DataFrame([{"venue": "TST", "sport": "football", "competition_id": "L1", "fixture_id": "F1",
                          "market_id": "1.1", "market_type": "MATCH_ODDS", "selection_id": f"S{r}",
                          "selection_name": r, "selection_role": r, "home_team_id": "H", "away_team_id": "A",
                          "scheduled_start_utc": iso(KO), "market_created_utc": iso(KO - pd.Timedelta(days=1)),
                          "source": "unit"} for r in roles])


def quote(ts: pd.Timestamp, role: str, back: float, lay: float, status: str = "OPEN", seq: int | None = None,
          **extra) -> dict:
    row = {"venue": "TST", "market_id": "1.1", "selection_id": f"S{role}",
           "provider_published_at_utc": iso(ts), "received_at_utc": iso(ts + pd.Timedelta(milliseconds=100)),
           "status": status, "best_back_odds": back, "best_back_size": 100.0, "best_lay_odds": lay,
           "best_lay_size": 100.0, "source": "unit", **extra}
    if seq is not None:
        row["sequence"] = seq
    return row


def book(ts: pd.Timestamp, seq: int | None = None, shift: float = 0.0, **extra) -> list[dict]:
    return [quote(ts, "home", 2.0 + shift, 2.02 + shift, seq=seq, **extra),
            quote(ts, "draw", 3.5, 3.55, seq=seq, **extra), quote(ts, "away", 4.0, 4.1, seq=seq, **extra)]


def replay_from(rows: list[dict], markets: pd.DataFrame | None = None, max_age: float = 900) -> MarketReplay:
    m = validate_markets(markets if markets is not None else mini_markets())
    q, rejected = validate_quotes(pd.DataFrame(rows).astype(str), m)
    assert rejected.empty, rejected
    return MarketReplay(m, q, max_age)


def ns(ts: pd.Timestamp) -> np.ndarray:
    return np.array([ts.value], dtype=np.int64)


# ---------- spectral features ----------

def spec(**kw) -> WindowSpec:
    base = dict(window_steps=40, baseline_windows=3, dynamics_steps=2, cadence_minutes=1.0,
                min_coverage=0.9, scale_grid=(0.25, 4.0, 81))
    base.update(kw)
    return WindowSpec(**base)


def test_planted_frequency_shift_recovers_scale_factor():
    s = spec()
    t = np.arange(s.required_increments)
    rng = np.random.default_rng(3)
    x = np.sin(2 * np.pi * 0.1 * t) + 0.01 * rng.standard_normal(len(t))
    x[-s.window_steps - 2 * s.dynamics_steps:] = np.sin(2 * np.pi * 0.2 * t[-s.window_steps - 2 * s.dynamics_steps:])
    f = doppler_features(x, np.ones(len(x), bool), s)
    assert f["scale_factor"] == pytest.approx(2.0, rel=0.1)
    assert f["centroid_log_ratio"] > 0.3
    assert f["js"] > 0.1
    unchanged = doppler_features(np.sin(2 * np.pi * 0.1 * t), np.ones(len(t), bool), s)
    assert unchanged["log_scale_factor"] == pytest.approx(0.0, abs=0.05)
    assert unchanged["js"] < 0.05


def test_spectral_features_are_sign_invariant():
    s = spec()
    x = np.random.default_rng(9).standard_normal(s.required_increments)
    ok = np.ones(len(x), bool)
    a, b = doppler_features(x, ok, s), doppler_features(-x, ok, s)
    for key in a:
        if not key.startswith("_"):
            assert a[key] == pytest.approx(b[key], abs=1e-12)


def test_spectral_coverage_gate_and_bounds():
    s = spec()
    x = np.random.default_rng(1).standard_normal(s.required_increments)
    ok = np.ones(len(x), bool)
    f = snapshot(x, ok, s)
    assert 0 <= f["js"] <= np.log(2) + 1e-12
    ok[-10:] = False
    with pytest.raises(ValueError, match="coverage"):
        snapshot(x, ok, s)


# ---------- schema ----------

def test_timezones_are_converted_and_naive_rejected():
    out, raw = parse_utc(pd.Series(["2026-06-01T15:00:00+01:00", "2026-06-01T14:00:00Z"]), "t")
    assert out.iloc[0] == out.iloc[1] == pd.Timestamp("2026-06-01T14:00:00Z")
    assert raw.iloc[0].endswith("+01:00")
    with pytest.raises(SchemaError, match="no timezone"):
        parse_utc(pd.Series(["2026-06-01T14:00:00"]), "t")


def test_quote_rejections_are_reported_not_silent():
    m = validate_markets(mini_markets())
    t = KO - pd.Timedelta(hours=1)
    rows = book(t, seq=1) + [
        quote(t, "home", 2.1, 2.0, seq=2),                      # crossed
        quote(t, "draw", 1.0, 1.01, seq=3),                     # odds <= 1
        {**quote(t, "away", 4.0, 4.1, seq=4), "best_back_size": 0},
        quote(t, "home", 2.0, 2.02, seq=1),                     # duplicate stream message
        {**quote(t, "home", 2.0, 2.02, seq=5), "received_at_utc": iso(t - pd.Timedelta(seconds=5))},
        {**quote(t, "home", 2.0, 2.02, seq=6), "selection_id": "UNKNOWN"},
    ]
    q, rejected = validate_quotes(pd.DataFrame(rows).astype(str), m)
    assert len(q) == 3
    assert sorted(rejected["reject_reason"]) == sorted(["crossed_book", "odds_not_above_1", "nonpositive_size",
                                                        "duplicate_stream_message", "received_before_published",
                                                        "unknown_market_selection"])
    assert (rejected["source_row"] >= 2).all()


def test_markets_need_roles_and_unique_keys():
    bad = mini_markets()
    bad.loc[0, "selection_role"] = ""
    with pytest.raises(SchemaError, match="selection_role"):
        validate_markets(bad)
    with pytest.raises(SchemaError, match="duplicate"):
        validate_markets(pd.concat([mini_markets(), mini_markets().iloc[[0]]]))


# ---------- replay ----------

def test_replay_uses_received_time_and_marks_stale_quotes():
    t = KO - pd.Timedelta(hours=2)
    r = replay_from(book(t), max_age=600)
    st = r.state(np.array([t.value, (t + pd.Timedelta(milliseconds=150)).value,
                           (t + pd.Timedelta(seconds=700)).value]), "Shome")
    assert list(st.reason) == ["no_quote_yet", "valid", "stale_quote"]
    assert st.p_raw[1] == pytest.approx((1 / 2.0 + 1 / 2.02) / 2)
    total = (1 / 2.0 + 1 / 2.02) / 2 + (1 / 3.5 + 1 / 3.55) / 2 + (1 / 4.0 + 1 / 4.1) / 2
    assert st.p_norm[1] == pytest.approx((1 / 2.0 + 1 / 2.02) / 2 / total)


def test_no_fill_through_suspension_sequence_gap_or_reconnect():
    t = KO - pd.Timedelta(hours=2)
    later = t + pd.Timedelta(minutes=10)
    rows = book(t, seq=1) + [quote(t + pd.Timedelta(minutes=5), s, np.nan, np.nan, "SUSPENDED", seq=2)
                             for s in ("home", "draw", "away")]
    r = replay_from(rows)
    assert r.state(ns(later), "Shome").reason[0] == "suspended_or_break"
    rows_gap = book(t, seq=1) + [quote(t + pd.Timedelta(minutes=5), "draw", 3.5, 3.6, seq=5)]
    r = replay_from(rows_gap)
    st = r.state(ns(later), "Shome")
    assert not st.valid[0] and st.reason[0] == "suspended_or_break"
    rows_reconnect = book(t, seq=1) + [quote(t + pd.Timedelta(minutes=5), "draw", 3.5, 3.6, seq=2, reconnect=True)]
    r = replay_from(rows_reconnect)
    assert r.state(ns(later), "Shome").reason[0] == "suspended_or_break"
    rows_fresh = rows_reconnect + book(t + pd.Timedelta(minutes=6), seq=3)
    assert replay_from(rows_fresh).state(ns(later), "Shome").valid[0]


def test_missing_runner_and_in_play_and_kickoff():
    t = KO - pd.Timedelta(hours=2)
    two = mini_markets(("home", "away"))
    r = replay_from([quote(t, "home", 2, 2.02), quote(t, "away", 4, 4.1)], markets=two)
    assert r.state(ns(t + pd.Timedelta(minutes=1)), "Shome").reason[0] == "missing_runner"
    rows = book(t) + book(t + pd.Timedelta(minutes=5), in_play=True)
    r = replay_from(rows)
    assert r.state(ns(t + pd.Timedelta(minutes=6)), "Shome").reason[0] == "suspended_or_break"
    r = replay_from(book(KO - pd.Timedelta(minutes=1)))
    assert r.state(ns(KO), "Shome").reason[0] == "at_or_after_scheduled_start"


# ---------- targets ----------

def test_realized_targets_definitions_and_kickoff_rule():
    t0 = KO - pd.Timedelta(hours=2)
    rows = book(t0) + book(t0 + pd.Timedelta(minutes=2), shift=0.1) + book(t0 + pd.Timedelta(minutes=4), shift=-0.05)
    r = replay_from(rows)
    origin = (t0 + pd.Timedelta(minutes=1)).value
    out = realized_targets(r, "Shome", origin, 5, "raw_midpoint", 1e-4, "tick", "exclude", 60)
    mids = [((1 / (2 + s)) + 1 / (2.02 + s)) / 2 for s in (0.0, 0.1, -0.05)]
    x = np.log(np.array(mids) / (1 - np.array(mids)))
    assert out["signed_reprice"] == pytest.approx(x[2] - x[0])
    assert out["realized_variation"] == pytest.approx((x[1] - x[0]) ** 2 + (x[2] - x[1]) ** 2)
    assert out["net_move_proxy"] == pytest.approx((x[2] - x[0]) ** 2)
    with pytest.raises(TargetUnavailable, match="scheduled_start"):
        realized_targets(r, "Shome", (KO - pd.Timedelta(minutes=10)).value, 30, "raw_midpoint", 1e-4,
                         "tick", "exclude", 60)


def test_suspension_inside_target_window_policy():
    t0 = KO - pd.Timedelta(hours=2)
    rows = book(t0) + [quote(t0 + pd.Timedelta(minutes=2), s, np.nan, np.nan, "SUSPENDED")
                       for s in ("home", "draw", "away")] + book(t0 + pd.Timedelta(minutes=3), shift=0.2)
    r = replay_from(rows)
    origin = (t0 + pd.Timedelta(minutes=1)).value
    with pytest.raises(TargetUnavailable, match="interrupted"):
        realized_targets(r, "Shome", origin, 5, "raw_midpoint", 1e-4, "tick", "exclude", 60)
    bridged = realized_targets(r, "Shome", origin, 5, "raw_midpoint", 1e-4, "tick", "bridge", 60)
    assert bridged["target_invalid_points_bridged"] > 0
    assert bridged["realized_variation"] == pytest.approx(bridged["net_move_proxy"])


# ---------- news timing ----------

def news_frame(**overrides) -> pd.DataFrame:
    return validate_news(raw_news(**overrides))


def raw_news(**overrides) -> pd.DataFrame:
    row = {"news_id": "N1", "event_type": "injury", "subject_team_id": "H", "subject_player_id": "",
           "source_id": "wire", "source_url": "https://example.org", "revision_id": "r1",
           "source_published_at_utc": "2026-09-01T12:00:00Z", "first_seen_at_utc": "2026-09-01T12:00:20Z",
           "confidence": "confirmed", "raw_hash": "x", "fixture_id": "",
           "release_scheduled_known_at_utc": ""}
    row.update(overrides)
    return pd.DataFrame([row])


def test_post_news_origin_uses_later_availability_plus_latency():
    n = news_frame().iloc[0]
    origin, reason, rule = forecast_origin(n, SportsConfig(post_news_latency_seconds=30))
    assert origin == pd.Timestamp("2026-09-01T12:00:50Z") and not reason
    origin, reason, _ = forecast_origin(news_frame(first_seen_at_utc="").iloc[0], SportsConfig())
    assert origin is None and reason == "no_firm_event_timestamp"
    origin, reason, _ = forecast_origin(news_frame(first_seen_at_utc="2026-09-01T11:00:00Z").iloc[0],
                                        SportsConfig())
    assert reason == "first_seen_before_published"


def test_pre_release_origin_requires_schedule_known_in_advance():
    cfg = SportsConfig(mode="pre_release", pre_release_lead_minutes=5)
    n = news_frame(release_scheduled_known_at_utc="2026-08-30T12:00:00Z").iloc[0]
    origin, reason, _ = forecast_origin(n, cfg)
    assert origin == pd.Timestamp("2026-09-01T11:55:00Z") and not reason
    _, reason, _ = forecast_origin(news_frame().iloc[0], cfg)
    assert reason == "release_not_scheduled_at_origin"
    leaked = news_frame(release_scheduled_known_at_utc="2026-08-30T12:00:00Z",
                        first_seen_at_utc="2026-09-01T11:50:00Z").iloc[0]
    _, reason, _ = forecast_origin(leaked, cfg)
    assert reason == "news_public_before_pre_release_origin"


def test_revisions_are_not_overwritten():
    n = pd.concat([news_frame(revision_id="r0", confidence="rumour", first_seen_at_utc="2026-09-01T11:00:00Z",
                              source_published_at_utc="2026-09-01T11:00:00Z"), news_frame()])
    chosen, reason, earlier = choose_revision(n, SportsConfig())
    assert chosen["revision_id"] == "r1" and earlier == ["r0"]
    with pytest.raises(SchemaError, match="revision"):
        validate_news(pd.concat([raw_news(), raw_news()]))


# ---------- models ----------

def test_horizon_garch_sums_the_conditional_variance_path():
    rng = np.random.default_rng(5)
    x = rng.standard_normal(400) * 0.01
    out = horizon_garch(x, [1, 5, 30], 120)
    assert out["ok"]
    a, b, w, mu = out["alpha"], out["beta"], out["omega_pct2"], out["mu"] * 100
    h = [out["first_step_variance_pct2"]]
    for _ in range(29):
        h.append(w + (a + b) * h[-1])
    for steps in (1, 5, 30):
        assert out["forecasts"][steps] == pytest.approx(sum(v + mu ** 2 for v in h[:steps]) / 1e4)
    assert out["forecasts"][1] < out["forecasts"][5] < out["forecasts"][30]
    assert not horizon_garch(x[:50], [5], 120)["ok"]


def test_ridge_explanation_is_an_exact_decomposition():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((40, 3))
    y = np.exp(x @ np.array([0.5, -0.2, 0.0]) + rng.standard_normal(40) * 0.1)
    fit = fit_ridge(x, y, ["a", "b", "c"], 1.0, 1e-10)
    ex = fit.explain(np.array([0.3, -1.0, 2.0]))
    assert ex["intercept"] + sum(t["contribution_log"] for t in ex["terms"]) == pytest.approx(ex["log_forecast"])
    assert ex["forecast"] == pytest.approx(np.exp(ex["log_forecast"]))
    assert ex["terms"][0]["coefficient"] > 0 > ex["terms"][1]["coefficient"]


# ---------- end to end on synthetic data ----------

FAST = dict(bootstrap_replicates=100, min_train_events=8, placebo_shift_minutes=None, shuffled_link_test=False)


@pytest.fixture(scope="module")
def synthetic_frames():
    tables = generate(n_fixtures=24, seed=7)
    return {k: v.astype(str).replace({"nan": ""}) for k, v in tables.items()}


@pytest.fixture(scope="module")
def prepared(synthetic_frames):
    return prepare(synthetic_frames, SportsConfig(**FAST))


def test_exclusion_reasons_cover_planted_problems(prepared):
    res = build_panel(prepared, SportsConfig(**FAST), keep_lineage=False)
    reasons = set(res.exclusions["reason"])
    assert {"fixture_postponed_or_void", "simultaneous_news", "no_firm_event_timestamp",
            "target_window_reaches_scheduled_start"} <= reasons
    rumour = res.panel[res.panel["news_id"] == "SYN-N-013-0"]
    assert set(rumour["revision_id"]) <= {"r1"}
    assert not prepared.tables["rejected_quotes"].empty


def test_features_do_not_change_when_post_origin_quotes_change(synthetic_frames):
    cfg = SportsConfig(**FAST)
    base = build_panel(prepare(synthetic_frames, cfg), cfg, keep_lineage=False).panel
    row = base[base["horizon_minutes"] == 30].iloc[0]
    q = synthetic_frames["quotes"].copy()
    recv = pd.to_datetime(q["received_at_utc"], utc=True, format="ISO8601")
    after = (recv > row["origin_utc"]) & (q["market_id"] == row["market_key"].split(":", 1)[1]) & \
        (q["status"] == "OPEN")
    assert after.any()
    q.loc[after, "best_back_odds"] = (q.loc[after, "best_back_odds"].astype(float) * 1.3).round(2).astype(str)
    q.loc[after, "best_lay_odds"] = (q.loc[after, "best_lay_odds"].astype(float) * 1.3 + 0.02).round(2).astype(str)
    frames = {**synthetic_frames, "quotes": q}
    moved = build_panel(prepare(frames, cfg), cfg, keep_lineage=False).panel
    m = moved[(moved["event_key"] == row["event_key"]) & (moved["horizon_minutes"] == 30)].iloc[0]
    feature_cols = ["js", "centroid_log_ratio", "log_scale_factor", "log_energy_recent", "log_trailing_rv",
                    "log_spread", "garch_horizon", "p_origin"]
    for col in feature_cols:
        assert m[col] == pytest.approx(row[col], nan_ok=True), col
    assert m["realized_variation"] != pytest.approx(row["realized_variation"])


def test_quotes_published_before_but_received_after_origin_are_not_used(synthetic_frames):
    cfg = SportsConfig(**FAST)
    no_seq = {**synthetic_frames, "quotes": synthetic_frames["quotes"].drop(columns="sequence")}
    base = build_panel(prepare(no_seq, cfg), cfg, keep_lineage=False).panel
    row = base[base["horizon_minutes"] == 30].iloc[0]
    q = no_seq["quotes"].copy()
    market_id = row["market_key"].split(":", 1)[1]
    late = q.iloc[[0]].copy()
    late["market_id"] = market_id
    late["selection_id"] = row["selection_id"]
    late["provider_published_at_utc"] = iso(row["origin_utc"] - pd.Timedelta(seconds=1))
    late["received_at_utc"] = iso(row["origin_utc"] + pd.Timedelta(seconds=1))
    late["status"], late["best_back_odds"], late["best_lay_odds"] = "OPEN", "1.5", "1.52"
    frames = {**synthetic_frames, "quotes": pd.concat([q, late], ignore_index=True)}
    moved = build_panel(prepare(frames, cfg), cfg, keep_lineage=False).panel
    m = moved[(moved["event_key"] == row["event_key"]) & (moved["horizon_minutes"] == 30)].iloc[0]
    assert m["p_origin"] == pytest.approx(row["p_origin"])
    assert m["js"] == pytest.approx(row["js"])


def test_walk_forward_trains_only_on_fully_observed_earlier_events(prepared):
    cfg = SportsConfig(**FAST)
    res = build_panel(prepared, cfg, keep_lineage=False)
    forecasts, contributions, _ = walk_forward(res, cfg)
    assert not forecasts.empty
    ends = res.panel.set_index(["event_key", "horizon_minutes"])["target_end_ns"]
    origins = res.panel.set_index(["event_key", "horizon_minutes"])["origin_ns"]
    checked = 0
    for key, trace in res.traces.items():
        for h, block in trace.horizons.items():
            m = block["models"].get("event_state_doppler")
            if not m or "train_event_keys" not in m:
                continue
            origin = origins[(key, int(h))]
            assert all(ends[(k, int(h))] < origin for k in m["train_event_keys"])
            comp = block["models"]["comparable_event_mean"]["detail"]["comparable_event_keys"]
            assert all(ends[(k, int(h))] < origin for k in comp)
            checked += 1
    assert checked > 5
    ridge = contributions[(contributions["model"] == "event_state_doppler") & (contributions["term"] != "FORECAST")]
    total = ridge.groupby(["event_key", "horizon_minutes"])["contribution_log"].sum()
    fc = contributions[(contributions["model"] == "event_state_doppler") & (contributions["term"] == "FORECAST")]
    fc = fc.set_index(["event_key", "horizon_minutes"])["raw_value"]
    assert np.allclose(np.exp(total.loc[fc.index]), fc, rtol=1e-9)


def test_run_study_outputs_are_complete_auditable_and_deterministic(synthetic_frames, tmp_path):
    cfg = SportsConfig(bootstrap_replicates=100, min_train_events=8, placebo_shift_minutes=240)
    out = run_study(synthetic_frames, [{"table": "synthetic"}], cfg, tmp_path / "a")
    run_study(synthetic_frames, [{"table": "synthetic"}], cfg, tmp_path / "b")
    for name in ("common_sample_scores.csv", "per_event_forecasts.csv", "forecast_contributions.csv",
                 "bootstrap_comparisons.csv"):
        a, b = pd.read_csv(tmp_path / "a" / name), pd.read_csv(tmp_path / "b" / name)
        pd.testing.assert_frame_equal(a, b)
    scores = pd.read_csv(tmp_path / "a" / "common_sample_scores.csv")
    assert {"main", "placebo_timestamps"} <= set(scores["panel"])
    assert (scores.groupby(["panel", "horizon_minutes"])["n_events"].nunique() == 1).all()
    assert {"garch_horizon", "event_adjusted_garch", "event_state", "event_state_doppler",
            "event_state_doppler_minus_scale", "trailing_rv", "ewma"} <= set(scores["model"])
    manifest = json.loads((tmp_path / "a" / "run_manifest.json").read_text())
    assert manifest["code"]["source_sha256"] and manifest["config"]["seed"] == cfg.seed
    traces = read_traces(tmp_path / "a" / "audit" / "traces.jsonl")
    key = next(k for k, t in traces.items()
               if (t["horizons"].get("30", {}).get("models", {}).get("event_state_doppler") or {}).get("forecast"))
    text = explain(traces[key], 30, "event_state_doppler")
    for fn in ("events.forecast_origin", "replay.MarketReplay.grid", "spectral.doppler_features",
               "volatility.horizon_garch", "targets.realized_targets", "RidgeFit.explain", "REALIZED"):
        assert fn in text
    lineage = pd.read_csv(tmp_path / "a" / "feature_grid_lineage.csv")
    seg = lineage[lineage["event_key"] == lineage["event_key"].iloc[0]]
    counts = seg["increment_segment_at_origin_snapshot"].value_counts()
    assert counts["recent"] == counts["baseline_1"] == counts["baseline_3"] == 20
    assert seg["is_origin"].sum() == 1
    assert out["artifacts"]["gate_checks"]["gate"].isin(["pass", "not_demonstrated"]).all()


def test_adhoc_forecast_uses_no_future_target(prepared):
    cfg = SportsConfig(**FAST)
    market = sorted(prepared.replays)[-1]
    replay = prepared.replays[market]
    origin = pd.Timestamp(replay.scheduled_start_ns - 3 * 3600 * NS, tz="UTC")
    result, trace = forecast_at(prepared, cfg, market, replay.selection_for_role("home"), origin, 30, "injury")
    assert (result["forecast"].dropna() > 0).all()
    assert "realized" not in result.columns
    assert result["train_events"].iloc[0] >= cfg.min_train_events


def test_cli_end_to_end(tmp_path, capsys):
    from doppler_research.cli import main
    data = tmp_path / "data"
    main(["sports", "make-synthetic", "--output", str(data), "--fixtures", "20", "--seed", "7"])
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({**FAST, "data_declaration": {"provider": "synthetic"}}))
    main(["sports", "validate", "--input", str(data)])
    assert '"rejected_quotes_by_reason"' in capsys.readouterr().out
    main(["sports", "run", "--config", str(cfg), "--input", str(data), "--output", str(tmp_path / "run")])
    assert (tmp_path / "run" / "audit" / "traces.jsonl").exists()
    main(["sports", "explain", "--run", str(tmp_path / "run"), "--list"])
    keys = capsys.readouterr().out.split()
    main(["sports", "explain", "--run", str(tmp_path / "run"), "--event-key", keys[-1]])
    assert "Audit trail" in capsys.readouterr().out
