"""Causal intraday spectral ("Doppler") features of logit-probability increments.

The recent window and each of the baseline windows have equal clock duration,
so their FFT frequencies coincide and no interpolation is used to compare
them. The fitted scale factor ``a`` is a frequency-scaling analogy, not a
physical Doppler equation. Power spectra are unchanged when every increment is
sign-reversed, so these features cannot identify the direction of repricing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FEATURES = ["js", "js_velocity", "js_acceleration", "centroid_log_ratio", "centroid_velocity",
            "centroid_acceleration", "log_energy_recent", "log_energy_ratio", "high_freq_share",
            "spectral_entropy_recent", "log_scale_factor", "scale_fit_residual_js"]
ABLATION_GROUPS = {
    "js": ["js", "js_velocity", "js_acceleration"],
    "centroid": ["centroid_log_ratio", "centroid_velocity", "centroid_acceleration"],
    "scale": ["log_scale_factor", "scale_fit_residual_js"],
    "energy": ["log_energy_recent", "log_energy_ratio", "high_freq_share", "spectral_entropy_recent"],
}
ENERGY_FLOOR = 1e-18


class FeatureUnavailable(ValueError):
    """Raised with an exclusion reason when a causal feature cannot be formed."""


@dataclass(frozen=True)
class WindowSpec:
    window_steps: int
    baseline_windows: int
    dynamics_steps: int
    cadence_minutes: float
    min_coverage: float
    scale_grid: tuple[float, float, int]

    @property
    def snapshot_steps(self) -> int:
        return self.window_steps * (self.baseline_windows + 1)

    @property
    def required_increments(self) -> int:
        return self.snapshot_steps + 2 * self.dynamics_steps


def periodogram(increments: np.ndarray, cadence_minutes: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Hann-tapered power at nonzero native frequencies (cycles/minute), and energy."""
    x = np.asarray(increments, dtype=float)
    x = (x - x.mean()) * np.hanning(len(x))
    power = np.abs(np.fft.rfft(x)) ** 2
    freq = np.fft.rfftfreq(len(x), d=cadence_minutes)
    return freq[1:], power[1:], float(power[1:].sum() / len(x))


def _normalized(power: np.ndarray) -> np.ndarray:
    total = power.sum()
    if total <= ENERGY_FLOOR:
        return np.full(len(power), 1.0 / len(power))
    return power / total


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Natural-log Jensen-Shannon divergence; 0 log 0 = 0; bounded by ln 2."""
    m = (p + q) / 2
    def kl(a: np.ndarray) -> float:
        return float(np.sum(np.where(a > 0, a * np.log(np.maximum(a, 1e-300) / np.maximum(m, 1e-300)), 0.0)))
    return 0.5 * kl(p) + 0.5 * kl(q)


def scale_grid(spec: tuple[float, float, int]) -> np.ndarray:
    lo, hi, n = spec
    grid = np.exp(np.linspace(np.log(lo), np.log(hi), int(n)))
    return np.unique(np.r_[grid, 1.0])


def fit_scale_factor(freq: np.ndarray, base_power: np.ndarray, recent_norm: np.ndarray,
                     grid: np.ndarray) -> dict:
    """Minimize JS(recent, normalize(P_base(f/a)/a)) over a predeclared grid of ``a``.

    ``a > 1`` means the recent spectrum looks like the baseline stretched toward
    higher frequency. Evaluating ``P_base(f/a)`` between native frequencies uses
    linear interpolation of the *baseline* only; it adds no frequency resolution.
    """
    distances = np.empty(len(grid))
    for i, a in enumerate(grid):
        scaled = np.interp(freq / a, freq, base_power, left=0.0, right=0.0) / a
        distances[i] = js_divergence(recent_norm, _normalized(scaled))
    best = int(np.argmin(distances))
    unscaled = float(distances[int(np.argmin(np.abs(grid - 1.0)))])
    return {"scale_factor": float(grid[best]), "log_scale_factor": float(np.log(grid[best])),
            "scale_fit_residual_js": float(distances[best]), "scale_fit_unscaled_js": unscaled,
            "scale_fit_gain_js": unscaled - float(distances[best]),
            "scale_fit_at_grid_edge": bool(best in (0, len(grid) - 1))}


def _coverage(valid: np.ndarray) -> float:
    return float(np.mean(valid)) if len(valid) else 0.0


def snapshot(increments: np.ndarray, valid: np.ndarray, spec: WindowSpec) -> dict:
    """Recent window = last ``window_steps`` increments; baselines = the K windows before it."""
    w, k = spec.window_steps, spec.baseline_windows
    x = np.asarray(increments, dtype=float)[-spec.snapshot_steps:]
    ok = np.asarray(valid, dtype=bool)[-spec.snapshot_steps:]
    if len(x) < spec.snapshot_steps:
        raise FeatureUnavailable("insufficient_pre_origin_quote_history")
    windows = [(x[i * w:(i + 1) * w], ok[i * w:(i + 1) * w]) for i in range(k + 1)]
    coverages = [_coverage(v) for _, v in windows]
    if min(coverages) < spec.min_coverage:
        raise FeatureUnavailable("feature_window_coverage_below_minimum")
    # Invalid steps (suspension/gap) contribute a zero increment, i.e. no price change.
    filled = [np.where(v, seg, 0.0) for seg, v in windows]
    base_powers, base_energy = [], []
    for seg in filled[:-1]:
        freq, power, energy = periodogram(seg, spec.cadence_minutes)
        base_powers.append(power)
        base_energy.append(energy)
    freq, recent_power, er = periodogram(filled[-1], spec.cadence_minutes)
    base_power = np.mean(base_powers, axis=0)
    eb = float(np.mean(base_energy))
    p_base, p_recent = _normalized(base_power), _normalized(recent_power)
    cb, cr = float(freq @ p_base), float(freq @ p_recent)
    nyquist = freq[-1]
    entropy = float(-np.sum(np.where(p_recent > 0, p_recent * np.log(np.maximum(p_recent, 1e-300)), 0))
                    / np.log(len(p_recent)))
    scale = fit_scale_factor(freq, base_power, p_recent, scale_grid(spec.scale_grid))
    return {"js": js_divergence(p_base, p_recent), "centroid_log_ratio": float(np.log(cr / cb)),
            "log_energy_recent": float(np.log(er + ENERGY_FLOOR)),
            "log_energy_ratio": float(np.log((er + ENERGY_FLOOR) / (eb + ENERGY_FLOOR))),
            "high_freq_share": float(p_recent[freq >= nyquist / 2].sum()),
            "spectral_entropy_recent": entropy, **scale,
            "_detail": {"frequencies_cycles_per_minute": freq.tolist(),
                        "recent_spectrum_normalized": p_recent.tolist(),
                        "baseline_spectrum_normalized": p_base.tolist(),
                        "window_coverage_oldest_to_recent": coverages,
                        "effective_increments": int(sum(v.sum() for _, v in windows))}}


def doppler_features(increments: np.ndarray, valid: np.ndarray, spec: WindowSpec) -> dict:
    """Snapshots at the origin and ``dynamics_steps`` / ``2*dynamics_steps`` earlier.

    Velocity and acceleration are clock-time first and second differences per minute.
    """
    x = np.asarray(increments, dtype=float)
    ok = np.asarray(valid, dtype=bool)
    if len(x) < spec.required_increments:
        raise FeatureUnavailable("insufficient_pre_origin_quote_history")
    s = spec.dynamics_steps
    now = snapshot(x, ok, spec)
    prev = snapshot(x[:-s], ok[:-s], spec)
    prev2 = snapshot(x[:-2 * s], ok[:-2 * s], spec)
    dt = s * spec.cadence_minutes
    out = {key: value for key, value in now.items() if not key.startswith("_")}
    for name in ("js", "centroid_log_ratio"):
        stem = "js" if name == "js" else "centroid"
        out[f"{stem}_velocity"] = (now[name] - prev[name]) / dt
        out[f"{stem}_acceleration"] = (now[name] - 2 * prev[name] + prev2[name]) / dt ** 2
    out["_detail"] = {**now["_detail"],
                      "snapshots": {"origin": {"js": now["js"], "centroid_log_ratio": now["centroid_log_ratio"]},
                                    f"origin_minus_{dt:g}m": {"js": prev["js"],
                                                              "centroid_log_ratio": prev["centroid_log_ratio"]},
                                    f"origin_minus_{2 * dt:g}m": {"js": prev2["js"],
                                                                  "centroid_log_ratio": prev2["centroid_log_ratio"]}}}
    return out
