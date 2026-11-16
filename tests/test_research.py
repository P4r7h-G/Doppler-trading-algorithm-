from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from doppler_research.garch import conditional_variances, fit_garch
from doppler_research.io import package_three
from doppler_research.models import ridge_predict
from doppler_research.pipeline import Config, build_event_panel, walk_forward, forecast_latest
from doppler_research.schema import SchemaError, validate_events, validate_prices
from doppler_research.scenarios import calibrated_price_intervals
from doppler_research.scoring import score_forecasts
from doppler_research.spectral import features, snapshot


def synthetic_inputs(days: int = 500):
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=days)
    returns = rng.normal(0, 0.012, days)
    close = 100 * np.exp(np.cumsum(returns))
    open_ = close * np.exp(rng.normal(0, 0.003, days))
    high = np.maximum(open_, close) * 1.006
    low = np.minimum(open_, close) * 0.994
    prices = pd.DataFrame({"ticker": "TEST", "date": dates.strftime("%Y-%m-%d"),
                           "open": open_, "high": high, "low": low, "close": close,
                           "volume": rng.integers(100000, 200000, days),
                           "adjustment": "split_adjusted", "source": "synthetic_test"})
    event_dates = dates[130:days-2:18]
    events = pd.DataFrame({"event_id": [f"E{i}" for i in range(len(event_dates))],
                           "ticker": "TEST", "announcement_date": event_dates.strftime("%Y-%m-%d"),
                           "session": "after_close", "event_type": "earnings",
                           "source_url": "https://example.org/synthetic-test-only",
                           "verified_at": (event_dates - pd.Timedelta(days=1)).strftime("%Y-%m-%d"), "confidence": "verified"})
    return prices, events


def test_spectrum_is_causal_and_signed():
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.01, 120)
    f = features(r[:100])
    assert 0 <= f["js"] <= np.log(2) + 1e-10
    assert set(f) == {"js", "js_velocity", "js_acceleration", "centroid_log_ratio",
                      "centroid_velocity", "centroid_acceleration", "log_energy_recent",
                      "log_energy_ratio", "high_energy_share"}
    r[100:] = 100
    assert f == features(r[:100])
    # Constant spectra have no directional shift or JS deformation.
    constant = snapshot(np.zeros(80))
    assert constant["js"] == pytest.approx(0)
    assert constant["centroid_log_ratio"] == pytest.approx(0)


def test_garch_recursion_and_constraints():
    e = np.array([1.0, -2.0, 0.5])
    h = conditional_variances(e, 0.2, 0.1, 0.7)
    assert h[1] == pytest.approx(0.2 + 0.1 * e[0] ** 2 + 0.7 * h[0])
    rng = np.random.default_rng(4)
    fit = fit_garch(rng.normal(0, 0.01, 130), 120)
    assert fit.converged and fit.forecast_variance > 0
    assert fit.alpha >= 0 and fit.beta >= 0 and fit.alpha + fit.beta < 0.9991


def test_ticker_effects_are_fitted_from_training_rows():
    x = np.zeros((6, 1))
    y = np.array([0.01] * 3 + [0.04] * 3)
    tickers = np.array(["A"] * 3 + ["B"] * 3)
    a = ridge_predict(x, y, np.zeros(1), 0.1, tickers, "A")
    b = ridge_predict(x, y, np.zeros(1), 0.1, tickers, "B")
    assert b > a > 0


def test_schema_rejects_unverified_adjustment_and_unknown_timing():
    p, e = synthetic_inputs(150)
    p.loc[0, "adjustment"] = "unknown"
    with pytest.raises(SchemaError):
        validate_prices(p)
    e.loc[0, "session"] = "unverified_value"
    with pytest.raises(SchemaError):
        validate_events(e)
    e.loc[0, "session"] = "after_close"
    e.loc[0, "confidence"] = "reported"
    e.loc[0, "verified_at"] = ""
    assert pd.isna(validate_events(e).loc[0, "verified_at"])


def test_end_to_end_common_sample_and_no_reaction_day_feature_leak():
    p, e = synthetic_inputs()
    config = Config(min_train_events=4, min_prior_ticker_events=1, bootstrap_replicates=100)
    panel, skipped = build_event_panel(p, e, config)
    assert skipped.empty
    first = panel.iloc[0]
    assert first.asof_date == first.announcement_date
    assert first.reaction_date > first.asof_date
    p2 = p.copy()
    reaction = first.reaction_date.strftime("%Y-%m-%d")
    p2.loc[p2.date == reaction, ["open", "high", "low", "close"]] *= 1.5
    panel2, _ = build_event_panel(p2, e, config)
    for col in ("js", "centroid_log_ratio", "rv20", "garch", "volume_z"):
        assert panel2.loc[0, col] == pytest.approx(panel.loc[0, col])
    assert panel2.loc[0, "squared_return"] != panel.loc[0, "squared_return"]
    forecasts = walk_forward(panel, config)
    scored, scores, comparisons = score_forecasts(forecasts, 100, 4)
    assert scored.event_id.nunique() > 0
    assert scores[scores.panel == "pooled"].n.nunique() == 1
    assert {"garch", "historical_event_mean", "state_only", "doppler_only", "event_state_doppler"}.issubset(set(scored.model))
    assert ((scored.forecast_variance > 0) & np.isfinite(scored.qlike_loss)).all()
    assert not comparisons.empty


def test_before_open_alignment_and_strict_advance_notice():
    p, e = synthetic_inputs(170)
    e = e.iloc[[0]].copy()
    e.loc[:, "session"] = "before_open"
    panel, skipped = build_event_panel(p, e, Config(min_garch_returns=120))
    assert skipped.empty
    assert panel.loc[0, "reaction_date"] == panel.loc[0, "announcement_date"]
    assert panel.loc[0, "asof_date"] < panel.loc[0, "announcement_date"]
    e.loc[:, "verified_at"] = e["announcement_date"]
    with pytest.raises(SchemaError):
        validate_events(e)


def test_price_interval_uses_only_earlier_oos_returns():
    dates = pd.bdate_range("2022-01-01", periods=50)
    f = pd.DataFrame({"event_id": [f"E{i}" for i in range(50)], "ticker": "TEST",
                      "model": "garch", "asof_date": dates,
                      "reaction_date": dates + pd.Timedelta(days=1),
                      "realized_return": np.linspace(-0.02, 0.02, 50),
                      "forecast_variance": 0.0001, "asof_close": 100.0})
    first = calibrated_price_intervals(f, 0.0, min_prior=10).iloc[0]
    f.loc[49, "realized_return"] = 10.0
    after = calibrated_price_intervals(f, 0.0, min_prior=10).iloc[0]
    assert first.lower_95_price == pytest.approx(after.lower_95_price)
    assert first.upper_95_price == pytest.approx(after.upper_95_price)


def test_latest_after_close_forecast_has_no_price_point_guess():
    p, e = synthetic_inputs()
    last_date = p.date.iloc[-1]
    candidate = pd.DataFrame({"event_id": ["NEXT"], "ticker": ["TEST"],
                              "announcement_date": [last_date], "session": ["after_close"],
                              "event_type": ["earnings"], "source_url": ["https://example.org/synthetic-test-only"],
                              "verified_at": [(pd.Timestamp(last_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")],
                              "confidence": ["verified"]})
    result, intervals = forecast_latest(p, pd.concat([e, candidate], ignore_index=True), "NEXT",
                                        Config(min_train_events=4, min_prior_ticker_events=1, bootstrap_replicates=100))
    assert len(result) == 9
    assert (result.forecast_variance > 0).all()
    assert "predicted_price" not in result.columns
    assert intervals.empty


def test_three_data_exports_keep_apple_separate(tmp_path):
    from zipfile import ZipFile

    p, e = synthetic_inputs(170)
    other_p, other_e = p.copy(), e.copy()
    p["ticker"] = "AAPL"
    e["ticker"] = "AAPL"
    other_p["ticker"] = "TEST"
    other_e["ticker"] = "TEST"
    other_e["event_id"] = "OTHER-" + other_e["event_id"]
    prices_path, events_path = tmp_path / "prices.csv", tmp_path / "events.csv"
    pd.concat([p, other_p]).to_csv(prices_path, index=False)
    pd.concat([e, other_e]).to_csv(events_path, index=False)
    made = package_three(prices_path, events_path, tmp_path / "packages")
    assert len(made) == 3
    with ZipFile(made[0]) as z:
        assert set(pd.read_csv(z.open("events.csv")).ticker) == {"AAPL"}
    with ZipFile(made[1]) as z:
        assert set(pd.read_csv(z.open("prices.csv")).ticker) == {"AAPL"}
    with ZipFile(made[2]) as z:
        assert set(pd.read_csv(z.open("prices.csv")).ticker) == {"TEST"}
        assert set(pd.read_csv(z.open("events.csv")).ticker) == {"TEST"}
