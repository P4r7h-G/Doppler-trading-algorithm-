"""Horizon-matched variance baselines on clock-grid logit increments.

All three return the forecast *sum* of squared increments over ``steps`` future
grid steps, so they are in the same units as a realized-variation target over
the same horizon.
"""
from __future__ import annotations

import numpy as np

from ..garch import conditional_variances, fit_garch


def horizon_garch(increments: np.ndarray, steps: list[int], min_observations: int) -> dict:
    """Fit GARCH(1,1) once at the origin; sum its conditional variances over each horizon.

    Fitted in percent-logit units by the shared quasi-MLE estimator. With
    ``h1 = omega + alpha*e_T^2 + beta*h_T`` and ``h_k = omega + (alpha+beta)*h_(k-1)``,
    the forecast is ``sum_k (h_k + mu^2) / 10^4``: a squared-increment target
    includes the mean, so ``mu^2`` per step is added. Gaps are dropped, so the
    fitted series concatenates valid increments; the count of dropped steps is
    reported.
    """
    x = np.asarray(increments, dtype=float)
    valid = x[np.isfinite(x)]
    detail = {"fit_observations": int(len(valid)), "dropped_invalid_steps": int(len(x) - len(valid))}
    try:
        fit = fit_garch(valid, min_observations)
    except ValueError as exc:
        return {**detail, "ok": False, "failure_reason": f"garch_failed: {exc}",
                "forecasts": {int(s): np.nan for s in steps}}
    mu_pct = fit.mu * 100.0
    eps = valid * 100.0 - mu_pct
    h = conditional_variances(eps, fit.omega, fit.alpha, fit.beta)
    path = np.empty(max(steps))
    path[0] = fit.omega + fit.alpha * eps[-1] ** 2 + fit.beta * h[-1]
    for k in range(1, len(path)):
        path[k] = fit.omega + (fit.alpha + fit.beta) * path[k - 1]
    forecasts = {int(s): float(np.sum(path[:s] + mu_pct ** 2) / 10000.0) for s in steps}
    return {**detail, "ok": True, "failure_reason": "", "forecasts": forecasts,
            "omega_pct2": fit.omega, "alpha": fit.alpha, "beta": fit.beta, "persistence": fit.alpha + fit.beta,
            "mu": fit.mu, "negative_log_likelihood": fit.negative_log_likelihood,
            "last_residual_pct": float(eps[-1]), "last_conditional_variance_pct2": float(h[-1]),
            "first_step_variance_pct2": float(path[0])}


def trailing_rv(increments: np.ndarray, window_steps: int, steps: int) -> dict:
    """Mean squared valid increment over the trailing window, times ``steps``."""
    x = np.asarray(increments, dtype=float)[-window_steps:]
    valid = x[np.isfinite(x)]
    if len(valid) == 0:
        return {"ok": False, "forecast": np.nan, "observations": 0, "mean_squared_increment": np.nan}
    msi = float(np.mean(valid ** 2))
    return {"ok": True, "forecast": msi * steps, "observations": int(len(valid)),
            "mean_squared_increment": msi, "realized_variation_trailing": float(np.sum(valid ** 2))}


def ewma_variance(increments: np.ndarray, lam: float, steps: int, warmup: int = 20) -> dict:
    """RiskMetrics recursion s_t = lam*s_(t-1) + (1-lam)*r_t^2 over valid increments."""
    x = np.asarray(increments, dtype=float)
    valid = x[np.isfinite(x)]
    if len(valid) < warmup + 1:
        return {"ok": False, "forecast": np.nan, "observations": int(len(valid))}
    s = float(np.mean(valid[:warmup] ** 2))
    for r in valid[warmup:]:
        s = lam * s + (1 - lam) * r * r
    return {"ok": True, "forecast": s * steps, "one_step_variance": s, "lambda": lam,
            "observations": int(len(valid))}
