"""Gaussian quasi-MLE GARCH(1,1), fitted anew at each forecast origin."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class GarchFit:
    omega: float
    alpha: float
    beta: float
    mu: float
    forecast_variance: float
    negative_log_likelihood: float
    converged: bool
    observations: int


def conditional_variances(residuals: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
    """h[t] = omega + alpha*e[t-1]^2 + beta*h[t-1]; h[0]=sample variance."""
    eps = np.asarray(residuals, dtype=float)
    h = np.empty(len(eps), dtype=float)
    h[0] = max(float(np.var(eps)), 1e-10)
    for t in range(1, len(eps)):
        h[t] = omega + alpha * eps[t - 1] ** 2 + beta * h[t - 1]
    return h


def fit_garch(returns: np.ndarray, min_observations: int = 120) -> GarchFit:
    """Input decimal log returns. Fit in percent units for numerical stability.

    Constraints: omega>0, alpha>=0, beta>=0, alpha+beta<1. The mean is
    the training-sample average; it is not a point price forecast.
    """
    r = np.asarray(returns, dtype=float)
    if len(r) < min_observations or not np.isfinite(r).all():
        raise ValueError("GARCH needs sufficient finite historical returns")
    x = r * 100.0
    mu = float(x.mean())
    eps = x - mu
    sample_var = max(float(np.var(eps)), 1e-8)

    def objective(theta: np.ndarray) -> float:
        omega, alpha, beta = theta
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
            return 1e15
        h = conditional_variances(eps, omega, alpha, beta)
        if not np.isfinite(h).all() or (h <= 0).any():
            return 1e15
        return float(0.5 * np.sum(np.log(2 * np.pi * h) + eps**2 / h))

    starts = [np.array([sample_var * 0.05, 0.08, 0.87]),
              np.array([sample_var * 0.2, 0.1, 0.7])]
    fits = [minimize(objective, x0=s, method="SLSQP",
                     bounds=[(1e-9, max(sample_var * 20, 1e-6)), (0, 0.998), (0, 0.998)],
                     constraints=[{"type": "ineq", "fun": lambda v: 0.999 - v[1] - v[2]}],
                     options={"maxiter": 300, "ftol": 1e-8}) for s in starts]
    good = [f for f in fits if f.success and np.isfinite(f.fun)]
    if not good:
        raise ValueError("GARCH optimizer did not converge")
    best = min(good, key=lambda f: f.fun)
    omega, alpha, beta = map(float, best.x)
    h = conditional_variances(eps, omega, alpha, beta)
    next_var = omega + alpha * eps[-1] ** 2 + beta * h[-1]
    return GarchFit(omega, alpha, beta, mu / 100, float(next_var / 10000),
                    float(best.fun), True, len(r))
