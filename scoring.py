"""Common-sample losses and paired calendar-quarter bootstrap diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models import FLOOR


def score_forecasts(forecasts: pd.DataFrame, replicates: int = 1000, seed: int = 17) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    f = forecasts.copy()
    required = set(f["model"].unique())
    counts = f.groupby("event_id")["model"].nunique()
    common_ids = counts[counts == len(required)].index
    f = f[f["event_id"].isin(common_ids)].copy()
    if f.empty:
        raise ValueError("no common-model forecast sample")
    if f.duplicated(["event_id", "model"]).any():
        raise ValueError("duplicate event/model forecast")
    p = np.maximum(f["forecast_variance"].to_numpy(dtype=float), FLOOR)
    y = f["observed_variance"].to_numpy(dtype=float)
    f["qlike_loss"] = np.log(p) + y / p
    f["absolute_error"] = np.abs(y - p)
    f["squared_error"] = (y - p)**2
    f["quarter"] = f["asof_date"].dt.to_period("Q").astype(str)
    summary = []
    for (panel, model), part in pd.concat([f.assign(panel="pooled"), f.assign(panel=f["ticker"])], ignore_index=True).groupby(["panel", "model"]):
        summary.append({"panel": panel, "target": part["target"].iloc[0], "model": model,
                        "n": len(part), "qlike": part["qlike_loss"].mean(),
                        "mae": part["absolute_error"].mean(),
                        "rmse": float(np.sqrt(part["squared_error"].mean())),
                        "bias": float((part["forecast_variance"] - part["observed_variance"]).mean())})
    scores = pd.DataFrame(summary).sort_values(["panel", "qlike"]).reset_index(drop=True)
    wide = f.pivot(index="event_id", columns="model", values="qlike_loss")
    meta = f.drop_duplicates("event_id").set_index("event_id")
    rng = np.random.default_rng(seed)
    pairs = [("doppler_only", "garch"), ("doppler_state", "state_only"),
             ("event_state_doppler", "event_state"),
             ("event_state_doppler", "historical_event_mean")]
    diagnostics = []
    for a, b in pairs:
        if a not in wide or b not in wide:
            continue
        delta = (wide[a] - wide[b]).rename("delta")
        paired = meta[["ticker", "quarter"]].join(delta)
        for panel_name, part in [("pooled", paired), *[(ticker, g) for ticker, g in paired.groupby("ticker")]]:
            quarters = part["quarter"].unique()
            if len(quarters) < 2:
                lo = hi = np.nan
            else:
                grouped = [part.loc[part["quarter"] == q, "delta"].to_numpy() for q in quarters]
                draws = np.empty(replicates)
                for i in range(replicates):
                    ix = rng.integers(0, len(grouped), size=len(grouped))
                    draws[i] = np.concatenate([grouped[k] for k in ix]).mean()
                lo, hi = np.quantile(draws, [0.025, 0.975])
            diagnostics.append({"panel": panel_name, "model_a": a, "model_b": b,
                                "n_events": len(part), "n_quarters": len(quarters),
                                "mean_qlike_delta_a_minus_b": float(part["delta"].mean()),
                                "bootstrap_ci_low": lo, "bootstrap_ci_high": hi,
                                "fraction_a_better": float((part["delta"] < 0).mean()),
                                "bootstrap_unit": "calendar_quarter", "replicates": replicates})
    return f, scores, pd.DataFrame(diagnostics)
