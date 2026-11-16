"""Inspectable forecast models for positive horizon variation targets.

Every ridge forecast decomposes exactly as
``log(forecast) = intercept + sum_j coef_j * (x_j - train_mean_j) / train_scale_j``
so each feature's contribution to the forecast can be exported and audited.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spectral import ABLATION_GROUPS, FEATURES as DOPPLER

STATE = ["log_trailing_rv", "log_spread", "log_depth", "log_minutes_to_kickoff", "abs_logit_origin"]
EVENT = ["log_comparable_event_mean"]

RIDGE_MODELS: dict[str, list[str]] = {
    "state_only": STATE,
    "doppler_only": DOPPLER,
    "state_doppler": STATE + DOPPLER,
    "event_state": EVENT + STATE,
    "event_state_doppler": EVENT + STATE + DOPPLER,
}
for _group, _cols in ABLATION_GROUPS.items():
    RIDGE_MODELS[f"event_state_doppler_minus_{_group}"] = [c for c in EVENT + STATE + DOPPLER if c not in _cols]

BASELINE_MODELS = ["trailing_rv", "ewma", "garch_horizon", "comparable_event_mean",
                   "comparable_event_median", "event_adjusted_garch"]
ALL_MODELS = BASELINE_MODELS + list(RIDGE_MODELS)
LOG_CLIP = (-40.0, 5.0)


@dataclass(frozen=True)
class RidgeFit:
    features: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    alpha: float
    n_train: int

    def explain(self, x: np.ndarray) -> dict:
        z = (np.asarray(x, dtype=float) - self.mean) / self.scale
        contributions = self.coef * z
        raw_log = float(self.intercept + contributions.sum())
        log_forecast = float(np.clip(raw_log, *LOG_CLIP))
        terms = [{"term": name, "raw_value": float(v), "train_mean": float(m), "train_scale": float(s),
                  "standardized": float(zz), "coefficient": float(c), "contribution_log": float(ct)}
                 for name, v, m, s, zz, c, ct in zip(self.features, x, self.mean, self.scale, z,
                                                     self.coef, contributions)]
        return {"intercept": self.intercept, "terms": terms, "log_forecast_unclipped": raw_log,
                "log_forecast": log_forecast, "clipped": raw_log != log_forecast,
                "forecast": float(np.exp(log_forecast))}


def fit_ridge(train_x: np.ndarray, train_y: np.ndarray, features: list[str], alpha: float,
              floor: float) -> RidgeFit:
    """Ridge on log target with centring/scaling from the training rows only."""
    x = np.asarray(train_x, dtype=float)
    y = np.asarray(train_y, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(features) or len(y) != len(x):
        raise ValueError("invalid ridge shapes")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("ridge inputs must be finite")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    xs = (x - mean) / scale
    target = np.log(np.maximum(y, floor))
    intercept = float(target.mean())
    coef = np.linalg.solve(xs.T @ xs + alpha * np.eye(x.shape[1]), xs.T @ (target - intercept))
    return RidgeFit(tuple(features), mean, scale, coef, intercept, alpha, len(y))


def comparable_history(prior: dict[str, np.ndarray], row: dict, levels: tuple[tuple[str, ...], ...],
                       min_events: int) -> dict:
    """Hierarchical fallback over predeclared grouping levels, most specific first.

    ``prior`` holds column arrays for strictly earlier, fully observed rows.
    """
    n = len(prior["event_key"])
    for depth, keys in enumerate(levels):
        mask = np.ones(n, bool)
        for k in keys:
            mask &= prior[k] == row[k]
        if mask.sum() >= min_events:
            return {"level": depth, "level_keys": list(keys), "mask": mask}
    if n >= min_events:
        return {"level": len(levels), "level_keys": ["pooled_all_prior"], "mask": np.ones(n, bool)}
    return {"level": -1, "level_keys": [], "mask": np.zeros(n, bool)}
