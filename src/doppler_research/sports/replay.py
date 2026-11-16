"""Deterministic, causal replay of an exchange market from its received messages.

The state of a selection at instant ``t`` is the last OPEN quote *received* at
or before ``t``. It is invalid (never filled) when there is no quote yet, the
quote is older than ``max_quote_age_seconds``, a market break happened after
that quote (suspension/closure, reconnect, stream sequence gap, in-play flag),
or ``t`` is at or after the scheduled start.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .probability import implied, logit, normalize

NS = 1_000_000_000

VALID = "valid"
REASONS = ("no_quote_yet", "stale_quote", "suspended_or_break", "at_or_after_scheduled_start",
           "missing_runner")


@dataclass
class SelectionTape:
    selection_id: str
    role: str | None
    received_ns: np.ndarray
    usable: np.ndarray
    mid: np.ndarray
    spread: np.ndarray
    depth: np.ndarray
    traded_volume: np.ndarray
    source_row: np.ndarray


@dataclass
class BookState:
    """Arrays aligned with the requested instants."""
    times_ns: np.ndarray
    valid: np.ndarray
    reason: np.ndarray
    p_raw: np.ndarray
    p_norm: np.ndarray
    spread: np.ndarray
    depth: np.ndarray
    traded_volume: np.ndarray
    source_row: np.ndarray

    def probability(self, basis: str) -> np.ndarray:
        return self.p_raw if basis == "raw_midpoint" else self.p_norm


class MarketReplay:
    def __init__(self, market_rows: pd.DataFrame, quotes: pd.DataFrame, max_quote_age_seconds: float):
        first = market_rows.iloc[0]
        self.market_key = str(first["market_key"])
        self.fixture_id = str(first["fixture_id"])
        self.market_type = str(first["market_type"])
        self.scheduled_start_ns = int(pd.Timestamp(first["scheduled_start_utc"]).value)
        self.max_age_ns = int(max_quote_age_seconds * NS)
        self.selections = list(market_rows["selection_id"])
        self.roles = dict(zip(market_rows["selection_id"], market_rows["selection_role"].astype(object)))
        self.complete_roles = set(r for r in self.roles.values() if isinstance(r, str)) == {"home", "draw", "away"} \
            if self.market_type == "MATCH_ODDS" else True
        q = quotes.sort_values(["received_at_utc", "source_row"])
        self.message_ns = q["received_at_utc"].astype("int64").to_numpy() if len(q) else np.array([], dtype=np.int64)
        self.message_rows = q["source_row"].to_numpy() if len(q) else np.array([], dtype=int)
        self.break_ns = self._breaks(q)
        self.tapes = {s: self._tape(s, q[q["selection_id"] == s]) for s in self.selections}

    @staticmethod
    def _breaks(q: pd.DataFrame) -> np.ndarray:
        if q.empty:
            return np.array([], dtype=np.int64)
        recv = q["received_at_utc"].astype("int64")
        mask = q["status"].ne("OPEN") | q["reconnect"] | q["in_play"]
        times = list(recv[mask])
        if "sequence" in q.columns and q["sequence"].notna().any():
            seq = q[["sequence"]].assign(recv=recv).dropna().sort_values("sequence")
            seq = seq.drop_duplicates("sequence")
            jumps = seq["sequence"].diff() > 1
            times += list(seq.loc[jumps, "recv"])
        return np.array(sorted(set(times)), dtype=np.int64)

    def _tape(self, selection_id: str, q: pd.DataFrame) -> SelectionTape:
        usable = (q["status"].eq("OPEN") & ~q["in_play"]).to_numpy()
        back = q["best_back_odds"].to_numpy(dtype=float)
        lay = q["best_lay_odds"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            p_back, p_lay, mid = implied(back, lay)
        depth = (q["best_back_size"].to_numpy(dtype=float) + q["best_lay_size"].to_numpy(dtype=float))
        volume = q["traded_volume"].to_numpy(dtype=float) if "traded_volume" in q.columns \
            else np.full(len(q), np.nan)
        return SelectionTape(selection_id, self.roles.get(selection_id), q["received_at_utc"].astype("int64").to_numpy(),
                             usable, mid, p_back - p_lay, depth, volume, q["source_row"].to_numpy())

    def selection_for_role(self, role: str) -> str | None:
        for sid, r in self.roles.items():
            if r == role:
                return sid
        return None

    def _selection_state(self, tape: SelectionTape, t: np.ndarray, last_break: np.ndarray):
        idx = np.searchsorted(tape.received_ns, t, side="right") - 1
        has = idx >= 0
        j = np.where(has, idx, 0)
        if len(tape.received_ns) == 0:
            n = len(t)
            nan = np.full(n, np.nan)
            return np.zeros(n, bool), np.full(n, "no_quote_yet", dtype=object), nan, nan, nan, nan, np.full(n, -1)
        recv = tape.received_ns[j]
        usable = has & tape.usable[j]
        fresh = (t - recv) <= self.max_age_ns
        after_break = recv >= last_break
        valid = usable & fresh & after_break
        reason = np.full(len(t), VALID, dtype=object)
        reason[~has] = "no_quote_yet"
        reason[has & ~(tape.usable[j] & after_break)] = "suspended_or_break"
        reason[has & tape.usable[j] & after_break & ~fresh] = "stale_quote"
        rows = np.where(has, tape.source_row[j], -1)
        return valid, reason, tape.mid[j], tape.spread[j], tape.depth[j], tape.traded_volume[j], rows

    def state(self, times_ns: np.ndarray, selection_id: str) -> BookState:
        """Book state of ``selection_id`` at each instant, using only received messages."""
        t = np.asarray(times_ns, dtype=np.int64)
        b = np.searchsorted(self.break_ns, t, side="right") - 1
        breaks = np.r_[self.break_ns, np.iinfo(np.int64).min]
        last_break = np.where(b >= 0, breaks[np.maximum(b, 0)], np.iinfo(np.int64).min)
        mids, valid_all = [], np.ones(len(t), bool)
        reason = np.full(len(t), VALID, dtype=object)
        target = None
        for sid in self.selections:
            v, r, mid, spread, depth, vol, rows = self._selection_state(self.tapes[sid], t, last_break)
            mids.append(mid)
            first_bad = valid_all & ~v
            reason[first_bad] = r[first_bad]
            valid_all &= v
            if sid == selection_id:
                target = (mid, spread, depth, vol, rows)
        if target is None:
            raise KeyError(f"selection {selection_id} not in market {self.market_key}")
        pre_match = t < self.scheduled_start_ns
        reason[valid_all & ~pre_match] = "at_or_after_scheduled_start"
        valid_all &= pre_match
        if not self.complete_roles:
            reason[:] = "missing_runner"
            valid_all[:] = False
        stacked = np.vstack(mids).T
        p_norm = normalize(stacked)[:, self.selections.index(selection_id)]
        mid, spread, depth, vol, rows = target
        return BookState(t, valid_all, reason, np.where(valid_all, mid, np.nan),
                         np.where(valid_all, p_norm, np.nan), spread, depth, vol, rows)

    def grid(self, end_ns: int, steps: int, cadence_seconds: int, selection_id: str) -> BookState:
        """``steps+1`` instants ending exactly at ``end_ns``, spaced by the cadence."""
        times = end_ns - np.arange(steps, -1, -1, dtype=np.int64) * int(cadence_seconds * NS)
        return self.state(times, selection_id)

    def message_times_between(self, start_ns: int, end_ns: int) -> np.ndarray:
        """Receipt instants of every market message with start < t <= end."""
        lo = np.searchsorted(self.message_ns, start_ns, side="right")
        hi = np.searchsorted(self.message_ns, end_ns, side="right")
        return np.unique(self.message_ns[lo:hi])

    def first_message_ns(self) -> int | None:
        return int(self.message_ns[0]) if len(self.message_ns) else None


def logit_path(state: BookState, basis: str, epsilon: float) -> np.ndarray:
    return np.where(state.valid, logit(state.probability(basis), epsilon), np.nan)


def build_replays(markets: pd.DataFrame, quotes: pd.DataFrame, max_quote_age_seconds: float
                  ) -> dict[str, MarketReplay]:
    grouped = {k: g for k, g in quotes.groupby("market_key", sort=False)}
    empty = quotes.iloc[0:0]
    return {key: MarketReplay(rows, grouped.get(key, empty), max_quote_age_seconds)
            for key, rows in markets.groupby("market_key", sort=True)}
