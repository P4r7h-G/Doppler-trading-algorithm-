"""Exchange odds to implied probabilities. Midpoints are quotes, not win chances."""
from __future__ import annotations

import numpy as np


def implied(back_odds: np.ndarray, lay_odds: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (p_back, p_lay, midpoint) with p = 1/decimal_odds.

    An uncrossed exchange book has back odds <= lay odds, so p_back >= p_lay.
    """
    p_back = 1.0 / np.asarray(back_odds, dtype=float)
    p_lay = 1.0 / np.asarray(lay_odds, dtype=float)
    return p_back, p_lay, (p_back + p_lay) / 2.0


def normalize(midpoints: np.ndarray) -> np.ndarray:
    """Divide each outcome's midpoint by the sum over all outcomes (last axis).

    A separately labelled transformation: raw midpoints are always retained.
    """
    m = np.asarray(midpoints, dtype=float)
    total = m.sum(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return m / total


def logit(p: np.ndarray, epsilon: float) -> np.ndarray:
    """log(p/(1-p)) after clipping to [epsilon, 1-epsilon] for numerics only."""
    q = np.clip(np.asarray(p, dtype=float), epsilon, 1.0 - epsilon)
    return np.log(q / (1.0 - q))
