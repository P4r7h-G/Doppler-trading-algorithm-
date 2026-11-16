"""Event alignment, causal feature assembly and expanding out-of-sample run."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .garch import fit_garch
from .models import MODEL_FEATURES, FLOOR, ridge_predict
from .schema import validate_events, validate_prices
from .spectral import features


@dataclass(frozen=True)
class Config:
    min_train_events: int = 12
    min_garch_returns: int = 120
    min_prior_ticker_events: int = 2
    ridge_alpha: float = 10.0
    bootstrap_replicates: int = 1000
    seed: int = 17
    target: str = "squared_return"
    bootstrap_cluster: str = "calendar_quarter"

    def __post_init__(self) -> None:
        if self.target not in {"squared_return", "gap_parkinson"}:
            raise ValueError("target must be squared_return or gap_parkinson")
        if self.min_train_events < 2 or self.min_garch_returns < 30 or self.min_prior_ticker_events < 1:
            raise ValueError("minimum history settings too small")
        if self.ridge_alpha <= 0 or self.bootstrap_replicates < 100:
            raise ValueError("invalid ridge/bootstrap settings")
        if self.bootstrap_cluster != "calendar_quarter":
            raise ValueError("only calendar_quarter bootstrap is implemented")


def build_event_panel(prices: pd.DataFrame, events: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    p, e = validate_prices(prices), validate_events(events)
    groups = {ticker: part.reset_index(drop=True) for ticker, part in p.groupby("ticker", sort=False)}
    rows, skipped = [], []
    for ev in e.to_dict("records"):
        eid, ticker, date, session = ev["event_id"], ev["ticker"], ev["announcement_date"], ev["session"]
        def skip(reason: str) -> None:
            skipped.append({"event_id": eid, "ticker": ticker, "announcement_date": date, "reason": reason})
        if ticker not in groups:
            skip("ticker_missing_prices"); continue
        if ev["confidence"] != "verified":
            skip("event_not_verified"); continue
        if session not in {"before_open", "after_close"}:
            skip("timing_unusable"); continue
        g = groups[ticker]
        matches = g.index[g["date"] == date].tolist()
        if not matches:
            skip("announcement_not_trading_date"); continue
        event_i = matches[0]
        asof_i = event_i if session == "after_close" else event_i - 1
        reaction_i = event_i + 1 if session == "after_close" else event_i
        if asof_i < 0:
            skip("no_pre_event_close"); continue
        if reaction_i >= len(g):
            skip("reaction_price_unavailable"); continue
        # r[k] is the close-to-close return ending at price row k+1.
        r = np.diff(np.log(g["close"].to_numpy(dtype=float)))
        history = r[:asof_i]
        if len(history) < max(82, config.min_garch_returns):
            skip("insufficient_causal_price_history"); continue
        try:
            spec = features(history)
            gf = fit_garch(history, config.min_garch_returns)
        except ValueError as exc:
            skip(str(exc)); continue
        recent = history[-20:]
        volume = g["volume"].to_numpy(dtype=float)
        vol_hist = np.log1p(volume[max(0, asof_i - 59):asof_i + 1])
        volume_z = float((vol_hist[-1] - vol_hist.mean()) / max(vol_hist.std(), 1e-12))
        asof_close = float(g.loc[asof_i, "close"])
        rr = g.loc[reaction_i]
        realized_return = float(np.log(float(rr["close"]) / asof_close))
        sqret = realized_return**2
        gap = float(np.log(float(rr["open"]) / asof_close))
        parkinson = float(np.log(float(rr["high"]) / float(rr["low"])) ** 2 / (4 * np.log(2)))
        row = {**ev, **spec, "asof_date": g.loc[asof_i, "date"],
               "reaction_date": rr["date"], "asof_close": asof_close,
               "reaction_close": float(rr["close"]), "realized_return": realized_return,
               "squared_return": sqret, "gap_parkinson": gap**2 + parkinson,
               "rv20": float(np.mean(recent**2)), "volume_z": volume_z,
               "garch": gf.forecast_variance, "garch_omega_pct2": gf.omega,
               "garch_alpha": gf.alpha, "garch_beta": gf.beta,
               "garch_mu": gf.mu, "garch_nll": gf.negative_log_likelihood,
               "garch_observations": gf.observations}
        rows.append(row)
    if not rows:
        raise ValueError("no usable verified events; inspect skipped reasons and input provenance")
    panel = pd.DataFrame(rows).sort_values(["asof_date", "ticker", "event_id"]).reset_index(drop=True)
    # For every event, only targets whose reaction closed before this as-of date count.
    prior_means, prior_medians, prior_counts, prior_ratios = [], [], [], []
    for row in panel.itertuples(index=False):
        prior = panel[(panel["ticker"] == row.ticker) & (panel["event_type"] == row.event_type) &
                      (panel["reaction_date"] < row.asof_date)]
        y = prior[config.target].to_numpy(dtype=float)
        prior_counts.append(len(prior))
        prior_means.append(float(y.mean()) if len(y) else np.nan)
        prior_medians.append(float(np.median(y)) if len(y) else np.nan)
        prior_ratios.append(float(np.mean(y / np.maximum(prior["garch"].to_numpy(dtype=float), FLOOR))) if len(y) else np.nan)
    panel["prior_event_count"] = prior_counts
    panel["historical_event_mean"] = prior_means
    panel["historical_event_median"] = prior_medians
    panel["event_garch_ratio"] = prior_ratios
    panel["log_prior_event_mean"] = np.log(np.maximum(panel["historical_event_mean"], FLOOR))
    return panel, pd.DataFrame(skipped, columns=["event_id", "ticker", "announcement_date", "reason"])


def walk_forward(panel: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Each row is predicted using strictly earlier, fully observed events."""
    records = []
    needed = list(MODEL_FEATURES)
    for row in panel.itertuples(index=False):
        if row.prior_event_count < config.min_prior_ticker_events:
            continue
        train = panel[(panel["reaction_date"] < row.asof_date) &
                      (panel["prior_event_count"] >= config.min_prior_ticker_events)]
        if len(train) < config.min_train_events:
            continue
        target = config.target
        observed = float(getattr(row, target))
        base = {"event_id": row.event_id, "ticker": row.ticker,
                "event_type": row.event_type, "announcement_date": row.announcement_date,
                "asof_date": row.asof_date, "reaction_date": row.reaction_date,
                "target": target, "observed_variance": observed,
                "realized_return": row.realized_return, "asof_close": row.asof_close}
        forecasts = {"garch": row.garch,
                     "historical_event_mean": row.historical_event_mean,
                     "historical_event_median": row.historical_event_median,
                     "event_adjusted_garch": row.garch * row.event_garch_ratio}
        for model in needed:
            columns = MODEL_FEATURES[model]
            forecasts[model] = ridge_predict(train[columns].to_numpy(dtype=float),
                                             train[target].to_numpy(dtype=float),
                                             np.asarray([getattr(row, c) for c in columns], dtype=float),
                                             config.ridge_alpha, train["ticker"].to_numpy(), row.ticker)
        for model, forecast in forecasts.items():
            records.append({**base, "model": model, "forecast_variance": max(float(forecast), FLOOR),
                            "train_events": len(train), "prior_ticker_events": row.prior_event_count})
    if not records:
        raise ValueError("no out-of-sample forecasts; reduce minimum history only with a documented reason or add earlier events")
    return pd.DataFrame(records)


def forecast_latest(prices: pd.DataFrame, events: pd.DataFrame, event_id: str,
                    config: Config, drift: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forecast a verified after-close event whose date equals the latest ticker close.

    It does not use the reaction price. Any optional price interval is calibrated
    from at least 40 earlier OOS events, not from the in-sample fitted residuals.
    """
    from .scenarios import calibrated_price_intervals
    from .scoring import score_forecasts

    p, e = validate_prices(prices), validate_events(events)
    candidate = e[e["event_id"] == event_id]
    if len(candidate) != 1:
        raise ValueError("event_id must identify exactly one event")
    ev = candidate.iloc[0]
    if ev["confidence"] != "verified" or ev["session"] != "after_close":
        raise ValueError("forecast-next requires a verified after-close event")
    g = p[p["ticker"] == ev["ticker"]].sort_values("date")
    if g.empty or ev["announcement_date"] != g["date"].max():
        raise ValueError("the selected event must equal the latest available close for its ticker")
    history = np.diff(np.log(g["close"].to_numpy(dtype=float)))
    if len(history) < max(82, config.min_garch_returns):
        raise ValueError("insufficient pre-event price history")
    spec = features(history)
    gf = fit_garch(history, config.min_garch_returns)
    vol = np.log1p(g["volume"].to_numpy(dtype=float)[-60:])
    state = {"rv20": float(np.mean(history[-20:]**2)),
             "volume_z": float((vol[-1] - vol.mean()) / max(vol.std(), 1e-12))}
    panel, _ = build_event_panel(p, e[e["event_id"] != event_id], config)
    origin = ev["announcement_date"]
    prior = panel[(panel["ticker"] == ev["ticker"]) & (panel["event_type"] == ev["event_type"]) &
                  (panel["reaction_date"] < origin)]
    if len(prior) < config.min_prior_ticker_events:
        raise ValueError("insufficient earlier same-ticker/event-type reactions")
    train = panel[(panel["reaction_date"] < origin) &
                  (panel["prior_event_count"] >= config.min_prior_ticker_events)]
    if len(train) < config.min_train_events:
        raise ValueError("insufficient fully observed earlier training events")
    y = prior[config.target].to_numpy(dtype=float)
    prior_mean = float(y.mean())
    feature_row = {**spec, **state, "log_prior_event_mean": float(np.log(max(prior_mean, FLOOR)))}
    forecasts = {"garch": gf.forecast_variance,
                 "historical_event_mean": prior_mean,
                 "historical_event_median": float(np.median(y)),
                 "event_adjusted_garch": gf.forecast_variance * float(np.mean(y / np.maximum(prior["garch"], FLOOR)))}
    for model, columns in MODEL_FEATURES.items():
        forecasts[model] = ridge_predict(train[columns].to_numpy(dtype=float),
                                         train[config.target].to_numpy(dtype=float),
                                         np.array([feature_row[col] for col in columns]), config.ridge_alpha,
                                         train["ticker"].to_numpy(), str(ev["ticker"]))
    result = pd.DataFrame([{"event_id": event_id, "ticker": ev["ticker"], "asof_date": origin,
                            "asof_close": float(g["close"].iloc[-1]), "target": config.target,
                            "model": model, "forecast_variance": max(float(value), FLOOR),
                            "train_events": len(train), "prior_ticker_events": len(prior),
                            "garch_omega_pct2": gf.omega, "garch_alpha": gf.alpha,
                            "garch_beta": gf.beta, "garch_mu": gf.mu} for model, value in forecasts.items()])
    intervals = pd.DataFrame(columns=[
        "event_id", "ticker", "asof_date", "model", "drift_log_return_assumption",
        "calibration_events", "lower_95_price", "upper_95_price", "method"])
    if drift is not None:
        try:
            past = walk_forward(panel[panel["reaction_date"] < origin], config)
        except ValueError as exc:
            if "no out-of-sample forecasts" not in str(exc):
                raise
        else:
            past, _, _ = score_forecasts(past, config.bootstrap_replicates, config.seed)
            candidate_rows = result[["event_id", "ticker", "asof_date", "model", "forecast_variance", "asof_close"]].copy()
            candidate_rows["reaction_date"] = pd.NaT
            candidate_rows["realized_return"] = np.nan
            intervals = calibrated_price_intervals(pd.concat([past, candidate_rows], ignore_index=True), drift)
            intervals = intervals[intervals["event_id"] == event_id].reset_index(drop=True)
    return result, intervals
