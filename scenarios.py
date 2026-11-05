"""Optional historical price scenarios with explicit drift and prior OOS calibration."""
from __future__ import annotations

import numpy as np
import pandas as pd


def calibrated_price_intervals(scored: pd.DataFrame, drift_per_session: float, min_prior: int = 40) -> pd.DataFrame:
    """Empirical 95% return interval from strictly earlier standardized OOS residuals.

    Drift is an externally supplied log-return assumption. No model predicts a
    directional price. Calibration pools prior events for each model and excludes
    events whose reaction close was unavailable before the current as-of date.
    """
    if not np.isfinite(drift_per_session):
        raise ValueError("finite explicit drift assumption required")
    if not 0 <= min_prior:
        raise ValueError("invalid calibration count")
    records = []
    for model, group in scored.groupby("model"):
        group = group.sort_values(["asof_date", "event_id"])
        for row in group.itertuples(index=False):
            prior = group[group["reaction_date"] < row.asof_date]
            if len(prior) < min_prior:
                continue
            z = (prior["realized_return"].to_numpy(dtype=float) - drift_per_session) / np.sqrt(prior["forecast_variance"].to_numpy(dtype=float))
            qlo, qhi = np.quantile(z, [0.025, 0.975])
            sd = np.sqrt(row.forecast_variance)
            records.append({"event_id": row.event_id, "ticker": row.ticker,
                            "asof_date": row.asof_date, "model": model,
                            "drift_log_return_assumption": drift_per_session,
                            "calibration_events": len(prior),
                            "lower_95_price": row.asof_close * np.exp(drift_per_session + qlo * sd),
                            "upper_95_price": row.asof_close * np.exp(drift_per_session + qhi * sd),
                            "method": "prior_OOS_standardized_return_empirical_quantiles"})
    return pd.DataFrame(records, columns=["event_id", "ticker", "asof_date", "model",
                                          "drift_log_return_assumption", "calibration_events",
                                          "lower_95_price", "upper_95_price", "method"])
