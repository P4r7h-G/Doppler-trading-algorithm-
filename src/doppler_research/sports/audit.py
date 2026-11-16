"""Audit trail: which function produced which value, and how it moved the forecast.

Each forecast has an ordered trace of steps. A step names the fully qualified
function that ran, the inputs it saw (values, timestamps, quote source rows)
and the outputs it produced. Model steps decompose the forecast into terms.
"""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd

PKG = "doppler_research.sports"

FEATURE_REGISTRY: dict[str, dict[str, str]] = {
    "js": {"function": f"{PKG}.spectral.snapshot", "units": "nats",
           "formula": "Jensen-Shannon divergence between normalized recent and mean-baseline power spectra"},
    "js_velocity": {"function": f"{PKG}.spectral.doppler_features", "units": "nats/minute",
                    "formula": "(js(origin) - js(origin - s)) / s"},
    "js_acceleration": {"function": f"{PKG}.spectral.doppler_features", "units": "nats/minute^2",
                        "formula": "(js(origin) - 2 js(origin - s) + js(origin - 2s)) / s^2"},
    "centroid_log_ratio": {"function": f"{PKG}.spectral.snapshot", "units": "log ratio",
                           "formula": "log(power-weighted mean frequency recent / baseline); >0 = faster"},
    "centroid_velocity": {"function": f"{PKG}.spectral.doppler_features", "units": "1/minute",
                          "formula": "first clock-time difference of centroid_log_ratio"},
    "centroid_acceleration": {"function": f"{PKG}.spectral.doppler_features", "units": "1/minute^2",
                              "formula": "second clock-time difference of centroid_log_ratio"},
    "log_energy_recent": {"function": f"{PKG}.spectral.periodogram", "units": "log logit^2",
                          "formula": "log(sum of recent tapered power / window length)"},
    "log_energy_ratio": {"function": f"{PKG}.spectral.snapshot", "units": "log ratio",
                         "formula": "log(recent energy / mean baseline energy)"},
    "high_freq_share": {"function": f"{PKG}.spectral.snapshot", "units": "share",
                        "formula": "recent normalized power at frequencies >= Nyquist/2"},
    "spectral_entropy_recent": {"function": f"{PKG}.spectral.snapshot", "units": "0..1",
                                "formula": "Shannon entropy of recent normalized spectrum / log(bins)"},
    "log_scale_factor": {"function": f"{PKG}.spectral.fit_scale_factor", "units": "log",
                         "formula": "log a*, a* = argmin_a JS(recent, normalize(P_base(f/a)/a)) on a fixed grid"},
    "scale_fit_residual_js": {"function": f"{PKG}.spectral.fit_scale_factor", "units": "nats",
                              "formula": "JS distance remaining at the fitted a*"},
    "log_trailing_rv": {"function": f"{PKG}.volatility.trailing_rv", "units": "log logit^2",
                        "formula": "log(sum of squared valid grid increments over trailing window)"},
    "log_spread": {"function": f"{PKG}.replay.MarketReplay.state", "units": "log probability",
                   "formula": "log(1/best_back_odds - 1/best_lay_odds) at origin"},
    "log_depth": {"function": f"{PKG}.replay.MarketReplay.state", "units": "log stake",
                  "formula": "log(1 + best_back_size + best_lay_size) at origin"},
    "log_minutes_to_kickoff": {"function": f"{PKG}.pipeline.build_panel", "units": "log minutes",
                               "formula": "log(minutes from origin to scheduled start)"},
    "abs_logit_origin": {"function": f"{PKG}.probability.logit", "units": "logit",
                         "formula": "|logit(p) at origin|; logit variance depends on price level"},
    "log_comparable_event_mean": {"function": f"{PKG}.models.comparable_history", "units": "log target",
                                  "formula": "log mean target of earlier fully observed comparable news, "
                                             "hierarchical fallback"},
}

MODEL_REGISTRY: dict[str, str] = {
    "trailing_rv": f"{PKG}.volatility.trailing_rv",
    "ewma": f"{PKG}.volatility.ewma_variance",
    "garch_horizon": f"{PKG}.volatility.horizon_garch",
    "comparable_event_mean": f"{PKG}.models.comparable_history",
    "comparable_event_median": f"{PKG}.models.comparable_history",
    "event_adjusted_garch": f"{PKG}.volatility.horizon_garch x {PKG}.models.comparable_history",
}


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(v) for v in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not math.isfinite(float(value)) else float(value)
    if value is pd.NaT or value is pd.NA:
        return None
    return value


def ns_iso(ns: int) -> str:
    return pd.Timestamp(int(ns), tz="UTC").isoformat()


class Trace:
    def __init__(self, event_key: str, panel: str):
        self.event_key = event_key
        self.panel = panel
        self.steps: list[dict] = []
        self.horizons: dict[str, dict] = {}

    def step(self, function: str, inputs: dict, outputs: dict, note: str = "") -> None:
        entry = {"step": len(self.steps) + 1, "function": function, "inputs": inputs, "outputs": outputs}
        if note:
            entry["note"] = note
        self.steps.append(entry)

    def horizon(self, minutes: int) -> dict:
        return self.horizons.setdefault(str(minutes), {"steps": [], "models": {}})

    def horizon_step(self, minutes: int, function: str, inputs: dict, outputs: dict, note: str = "") -> None:
        h = self.horizon(minutes)
        entry = {"step": f"h{minutes}.{len(h['steps']) + 1}", "function": function,
                 "inputs": inputs, "outputs": outputs}
        if note:
            entry["note"] = note
        h["steps"].append(entry)

    def to_dict(self) -> dict:
        return to_jsonable({"event_key": self.event_key, "panel": self.panel,
                            "steps": self.steps, "horizons": self.horizons})


def write_traces(traces: dict[str, Trace], path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for key in sorted(traces):
            fh.write(json.dumps(traces[key].to_dict(), sort_keys=True) + "\n")


def read_traces(path) -> dict[str, dict]:
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                record = json.loads(line)
                out[record["event_key"]] = record
    return out


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, list) and len(v) > 8:
        return f"[{len(v)} values: {', '.join(_fmt(x) for x in v[:3])}, ...]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}={_fmt(x)}" for k, x in v.items()) + "}"
    return str(v)


def _render_step(step: dict, lines: list[str]) -> None:
    lines.append(f"[{step['step']}] {step['function']}")
    if step.get("note"):
        lines.append(f"      note: {step['note']}")
    for label in ("inputs", "outputs"):
        items = step.get(label) or {}
        if items:
            lines.append(f"      {label}:")
            for k, v in items.items():
                lines.append(f"        {k} = {_fmt(v)}")


def explain(trace: dict, horizon: int | None = None, model: str | None = None) -> str:
    """Human-readable causal chain from input rows to forecast and realized target."""
    lines = [f"Audit trail for {trace['event_key']} (panel: {trace['panel']})", ""]
    for step in trace["steps"]:
        _render_step(step, lines)
    horizons = [str(horizon)] if horizon is not None else sorted(trace["horizons"], key=int)
    for h in horizons:
        block = trace["horizons"].get(h)
        if block is None:
            lines.append(f"\nHorizon {h}m: not available for this event")
            continue
        lines.append(f"\n=== Horizon {h} minutes ===")
        for step in block["steps"]:
            _render_step(step, lines)
        models = [model] if model else sorted(block["models"])
        for name in models:
            m = block["models"].get(name)
            if m is None:
                lines.append(f"\nModel {name}: no forecast (see exclusions)")
                continue
            lines.append(f"\nModel {name}  <- {m['function']}")
            if "intercept" in m:
                lines.append(f"  intercept (mean log target of {m['n_train']} training events with finite features) = {m['intercept']:.6g}")
                for t in sorted(m["terms"], key=lambda t: -abs(t["contribution_log"] or 0)):
                    lines.append(f"  {t['term']:<28} raw={_fmt(t['raw_value']):>12}  z={_fmt(t['standardized']):>10}"
                                 f"  coef={_fmt(t['coefficient']):>10}  contribution={t['contribution_log']:+.5f}")
                lines.append(f"  log forecast = {m['log_forecast']:.6g}{' (clipped)' if m.get('clipped') else ''}")
            else:
                for k, v in m.get("detail", {}).items():
                    lines.append(f"  {k} = {_fmt(v)}")
            lines.append(f"  FORECAST {trace['horizons'][h].get('target_name', 'target')} = {_fmt(m['forecast'])}")
            if m.get("realized") is not None:
                lines.append(f"  REALIZED = {_fmt(m['realized'])}   QLIKE = {_fmt(m.get('qlike'))}   "
                             f"error (forecast - realized) = {_fmt(m.get('error'))}")
    return "\n".join(lines)
