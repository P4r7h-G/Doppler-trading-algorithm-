"""Realized targets after the forecast origin, from later quotes only.

``x_t = logit(p_t)``; ``signed_reprice = x(t0+h) - x(t0)``;
``net_move_proxy = signed_reprice^2``; ``realized_variation = sum of squared
successive valid changes in x over (t0, t0+h]``. Realized variation measures the
market-implied probability path, not the variance of the match result.
"""
from __future__ import annotations

import numpy as np

from .replay import NS, MarketReplay, logit_path


class TargetUnavailable(ValueError):
    pass


def realized_targets(replay: MarketReplay, selection_id: str, origin_ns: int, horizon_minutes: int,
                     basis: str, epsilon: float, sampling: str, suspension_policy: str,
                     cadence_seconds: int) -> dict:
    """Return all target variants and the lineage of quote rows that produced them."""
    end_ns = origin_ns + horizon_minutes * 60 * NS
    if end_ns > replay.scheduled_start_ns:
        raise TargetUnavailable("target_window_reaches_scheduled_start")
    if sampling == "tick":
        times = np.r_[origin_ns, replay.message_times_between(origin_ns, end_ns), end_ns]
        times = np.unique(times)
    else:
        steps = int(horizon_minutes * 60 // cadence_seconds)
        times = origin_ns + np.arange(steps + 1, dtype=np.int64) * cadence_seconds * NS
    state = replay.state(times, selection_id)
    x = logit_path(state, basis, epsilon)
    if not state.valid[0]:
        raise TargetUnavailable(f"origin_quote_invalid:{state.reason[0]}")
    if not state.valid[-1]:
        raise TargetUnavailable(f"endpoint_quote_invalid:{state.reason[-1]}")
    interior_invalid = int((~state.valid).sum())
    if interior_invalid and suspension_policy == "exclude":
        bad = state.reason[~state.valid][0]
        raise TargetUnavailable(f"target_window_interrupted:{bad}")
    keep = state.valid
    xv = x[keep]
    dx = np.diff(xv)
    signed = float(xv[-1] - xv[0])
    lineage = {"times_ns": times[keep].tolist(), "logit": xv.tolist(),
               "source_row": state.source_row[keep].tolist()}
    return {"target_end_ns": int(end_ns), "realized_variation": float(np.sum(dx ** 2)),
            "net_move_proxy": signed ** 2, "signed_reprice": signed, "absolute_reprice": abs(signed),
            "p_origin": float(state.probability(basis)[0]), "p_end": float(state.probability(basis)[-1]),
            "target_points": int(keep.sum()), "target_nonzero_changes": int(np.count_nonzero(dx)),
            "target_invalid_points_bridged": interior_invalid, "_lineage": lineage}
