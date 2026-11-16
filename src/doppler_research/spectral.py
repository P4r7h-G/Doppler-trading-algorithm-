"""Causal daily spectral features. All windows end at the forecast as-of close."""
from __future__ import annotations

import numpy as np

BASELINE = 60
RECENT = 20
GRID = np.linspace(1 / BASELINE, 0.5, 32)
FEATURES = ["js", "js_velocity", "js_acceleration", "centroid_log_ratio", "centroid_velocity", "centroid_acceleration", "log_energy_recent", "log_energy_ratio", "high_energy_share"]


def _spectrum(values: np.ndarray) -> tuple[np.ndarray, float]:
    x = np.asarray(values, dtype=float)
    x = (x - x.mean()) * np.hanning(len(x))
    raw = np.abs(np.fft.rfft(x)) ** 2
    freq = np.fft.rfftfreq(len(x), d=1.0)
    power = np.interp(GRID, freq[1:], raw[1:], left=0.0, right=0.0)
    energy = float(power.sum() / len(x))
    if energy <= 1e-18:
        return np.full(len(GRID), 1 / len(GRID)), 0.0
    return power / power.sum(), energy


def snapshot(returns: np.ndarray) -> dict[str, float]:
    """Use exactly the last 80 returns: disjoint 60-day baseline and 20-day recent."""
    r = np.asarray(returns, dtype=float)
    if len(r) < BASELINE + RECENT or not np.isfinite(r[-80:]).all():
        raise ValueError("spectral snapshot requires 80 finite daily log returns")
    base, eb = _spectrum(r[-80:-20])
    recent, er = _spectrum(r[-20:])
    mid = (base + recent) / 2
    # 0 log 0 is defined as zero. Natural-log JS is bounded by ln(2).
    js = 0.5 * (np.sum(np.where(base > 0, base * np.log(np.maximum(base, 1e-300) / mid), 0)) +
                np.sum(np.where(recent > 0, recent * np.log(np.maximum(recent, 1e-300) / mid), 0)))
    cb = float(GRID @ base)
    cr = float(GRID @ recent)
    return {"js": float(js), "centroid_log_ratio": float(np.log(cr / cb)),
            "log_energy_recent": float(np.log(er + 1e-18)),
            "log_energy_ratio": float(np.log((er + 1e-18) / (eb + 1e-18))),
            "high_energy_share": float(recent[GRID >= 0.25].sum())}


def features(returns: np.ndarray) -> dict[str, float]:
    """Three consecutive causal snapshots give first and second differences."""
    r = np.asarray(returns, dtype=float)
    if len(r) < 82:
        raise ValueError("spectral dynamics require 82 returns")
    now, prev, prev2 = snapshot(r), snapshot(r[:-1]), snapshot(r[:-2])
    now["js_velocity"] = now["js"] - prev["js"]
    now["js_acceleration"] = now["js"] - 2 * prev["js"] + prev2["js"]
    now["centroid_velocity"] = now["centroid_log_ratio"] - prev["centroid_log_ratio"]
    now["centroid_acceleration"] = now["centroid_log_ratio"] - 2 * prev["centroid_log_ratio"] + prev2["centroid_log_ratio"]
    return now
