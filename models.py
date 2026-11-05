"""Simple train-only standardized ridge models for nonnegative variance."""
from __future__ import annotations

import numpy as np

from .spectral import FEATURES as DOPPLER

STATE = ["rv20", "volume_z"]
EVENT = ["log_prior_event_mean"]
MODEL_FEATURES = {
    "state_only": STATE,
    "doppler_only": DOPPLER,
    "doppler_state": STATE + DOPPLER,
    "event_state": EVENT + STATE,
    "event_state_doppler": EVENT + STATE + DOPPLER,
}
FLOOR = 1e-10


def ridge_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, alpha: float,
                  train_tickers: np.ndarray | None = None, test_ticker: str | None = None) -> float:
    """Ridge on log variance with train-only scaling and optional ticker effects."""
    x = np.asarray(train_x, dtype=float)
    y = np.asarray(train_y, dtype=float)
    z = np.asarray(test_x, dtype=float)
    if x.ndim != 2 or z.shape != (x.shape[1],) or len(y) != len(x):
        raise ValueError("invalid ridge shapes")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(z).all():
        raise ValueError("ridge inputs must be finite")
    if train_tickers is not None:
        t = np.asarray(train_tickers, dtype=str)
        if len(t) != len(x) or test_ticker is None:
            raise ValueError("ticker effect inputs must match training rows")
        categories = sorted(set(t))[1:]  # Reference ticker absorbed by intercept.
        if categories:
            x = np.column_stack([x, *[(t == category).astype(float) for category in categories]])
            z = np.r_[z, [float(test_ticker == category) for category in categories]]
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    xs = (x - mean) / scale
    zs = (z - mean) / scale
    target = np.log(np.maximum(y, FLOOR))
    intercept = float(target.mean())
    coef = np.linalg.solve(xs.T @ xs + alpha * np.eye(x.shape[1]), xs.T @ (target - intercept))
    return float(np.exp(np.clip(intercept + zs @ coef, -23, 2)))
