"""Clearly synthetic football match-odds data for installation and pipeline tests.

The news response and the pre-news oscillation are *planted* by this generator.
Scores on these files say nothing about real markets.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

LADDER = [(1.01, 2.0, 0.01), (2.0, 3.0, 0.02), (3.0, 4.0, 0.05), (4.0, 6.0, 0.1), (6.0, 10.0, 0.2),
          (10.0, 20.0, 0.5), (20.0, 30.0, 1.0), (30.0, 50.0, 2.0), (50.0, 100.0, 5.0), (100.0, 1000.0, 10.0)]
TICKS = np.unique(np.round(np.concatenate([np.arange(lo, hi, step) for lo, hi, step in LADDER] + [[1000.0]]), 2))
STEP_SECONDS = 5
TAG = "SYNTHETIC_TEST_ONLY"


def _tick_floor(odds: np.ndarray) -> np.ndarray:
    i = np.clip(np.searchsorted(TICKS, odds, side="right") - 1, 0, len(TICKS) - 2)
    return TICKS[i]


def _tick_ceil(odds: np.ndarray) -> np.ndarray:
    i = np.clip(np.searchsorted(TICKS, odds, side="left"), 1, len(TICKS) - 1)
    return TICKS[i]


def _iso(ts: pd.Timestamp, offset_hours: int = 0) -> str:
    if offset_hours:
        return ts.tz_convert(f"Etc/GMT-{offset_hours}").isoformat()
    return ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def generate(n_fixtures: int = 48, hours_before_kickoff: float = 9.0, seed: int = 20261116,
             start: str = "2026-08-08T00:00:00Z") -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    teams = [f"SYN-T{i:02d}" for i in range(1, 17)]
    base = pd.Timestamp(start)
    kickoff_slots = [pd.Timedelta(hours=12, minutes=30), pd.Timedelta(hours=15), pd.Timedelta(hours=15),
                     pd.Timedelta(hours=17, minutes=30)]
    markets, quotes, news, fixtures = [], [], [], []
    seq = 0
    special = {"postponed": 5, "simultaneous": 9, "rumour": 13, "reconnect": 17, "no_timestamp": 21}
    pairing = []
    for i in range(n_fixtures):
        slot = i % len(kickoff_slots)
        if slot == 0:
            pairing = list(rng.permutation(teams))
        ko = base + pd.Timedelta(days=i // len(kickoff_slots)) + kickoff_slots[slot]
        home, away = pairing[2 * slot], pairing[2 * slot + 1]
        fid = f"SYN-FX-{i:03d}"
        comp = "SYN-LEAGUE-A" if i % 2 == 0 else "SYN-LEAGUE-B"
        status = "postponed" if i == special["postponed"] else "played"
        fixtures.append({"fixture_id": fid, "competition_id": comp, "home_team_id": home, "away_team_id": away,
                         "scheduled_start_utc": _iso(ko), "fixture_status": status, "source": TAG})
        market_id = f"1.{200000 + i}"
        sel = {"home": f"{i:03d}01", "draw": f"{i:03d}02", "away": f"{i:03d}03"}
        for role, name in (("home", home), ("draw", "The Draw"), ("away", away)):
            markets.append({"venue": "SYNEX", "sport": "football", "competition_id": comp, "fixture_id": fid,
                            "market_id": market_id, "market_type": "MATCH_ODDS", "selection_id": sel[role],
                            "selection_name": name, "selection_role": role, "home_team_id": home,
                            "away_team_id": away, "scheduled_start_utc": _iso(ko),
                            "market_created_utc": _iso(ko - pd.Timedelta(days=2)), "source": TAG})

        t0 = ko - pd.Timedelta(hours=hours_before_kickoff)
        n = int(hours_before_kickoff * 3600 / STEP_SECONDS)
        times = (t0 + pd.to_timedelta(np.arange(n) * STEP_SECONDS, unit="s")).asi8
        sec = np.arange(n) * STEP_SECONDS
        liquidity = float(np.exp(rng.uniform(np.log(800), np.log(40000))))
        base_sigma = rng.uniform(0.002, 0.006)
        strength = rng.normal(0.25, 0.5)
        z = np.zeros((n, 2))
        z[0] = [strength, rng.normal(-0.2, 0.3)]
        vol = np.full(n, base_sigma)
        drift = np.zeros((n, 2))

        # Planted news: an injury to one side at a random time, and the confirmed lineup.
        injury_s = rng.uniform(3.0 * 3600, (hours_before_kickoff - 3.0) * 3600)
        lineup_s = (hours_before_kickoff - 1.25) * 3600
        events = []
        injured_home = rng.random() < 0.5
        severity = float(rng.gamma(2.0, 0.6))
        events.append(("injury", injury_s, severity, home if injured_home else away, 0 if injured_home else 1))
        events.append(("lineup_confirmed", lineup_s, float(rng.gamma(1.5, 0.4)), home, 0))
        if i == special["simultaneous"]:
            events.append(("injury", injury_s + 30, float(rng.gamma(2.0, 0.6)), away if injured_home else home,
                           1 if injured_home else 0))
        for kind, at, sev, _, coord in events:
            after = sec >= at
            # Pre-news informed-flow oscillation (period ~2-3 minutes), planted for the spectral features.
            if kind == "injury":
                leak = (sec >= at - 900) & ~after
                period = rng.uniform(100, 180)
                drift[leak, coord] += 0.03 * sev * np.cos(2 * np.pi * sec[leak] / period) * (2 * np.pi * STEP_SECONDS / period)
            jump = -0.35 * sev * (1 if kind == "injury" else rng.choice([-1, 1]))
            ramp = (sec >= at) & (sec < at + 45)
            drift[ramp, coord] += jump / max(ramp.sum(), 1)
            decay = np.exp(-np.clip(sec - at, 0, None) / rng.uniform(600, 1500))
            vol = vol * np.where(after, 1 + 2.5 * sev * decay, 1.0)
        shocks = rng.standard_normal((n, 2)) * vol[:, None]
        z = z[0] + np.cumsum(drift + shocks, axis=0)
        logits = np.column_stack([z[:, 0], np.zeros(n), z[:, 1]])
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

        suspend = (i % 3 == 0) and i != special["simultaneous"]
        susp_lo, susp_hi = injury_s - 5, injury_s + rng.uniform(60, 120)
        gap_lo, gap_hi = (injury_s - 7200, injury_s - 6600) if i == special["reconnect"] else (np.inf, np.inf)
        half = 0.004 + 0.02 / np.sqrt(liquidity / 1000)
        backs = _tick_floor(1 / np.minimum(probs * (1 + half), 0.995))
        lays = _tick_ceil(1 / np.maximum(probs * (1 - half), 0.001))
        lays = np.where(lays <= backs, _tick_ceil(backs + 1e-9), lays)
        last = {r: (None, None, -1e9) for r in sel}
        suspended_emitted = False
        reconnect_pending = False
        for k in range(n):
            s = sec[k]
            if gap_lo <= s < gap_hi:
                reconnect_pending = True
                continue
            if suspend and susp_lo <= s < susp_hi:
                if not suspended_emitted:
                    seq += 1
                    for role in sel:
                        quotes.append(_row(market_id, sel[role], times[k], _latency_ns(rng), "SUSPENDED",
                                           np.nan, np.nan, np.nan, np.nan, seq, False))
                    suspended_emitted = True
                    last = {r: (None, None, -1e9) for r in sel}
                continue
            p, back, lay = probs[k], backs[k], lays[k]
            changed = [r for j, r in enumerate(sel) if (back[j], lay[j]) != last[r][:2] or s - last[r][2] >= 300]
            if reconnect_pending:
                changed = list(sel)
            if not changed:
                continue
            seq += 1
            for j, role in enumerate(sel):
                if role not in changed:
                    continue
                size_scale = liquidity * p[j]
                back_size = float(np.round(size_scale * rng.lognormal(0, 0.5), 2))
                lay_size = float(np.round(size_scale * rng.lognormal(0, 0.5), 2))
                quotes.append(_row(market_id, sel[role], times[k], _latency_ns(rng), "OPEN", back[j], back_size,
                                   lay[j], lay_size, seq, reconnect_pending))
                last[role] = (back[j], lay[j], s)
            reconnect_pending = False

        for e_idx, (kind, at, sev, team, _) in enumerate(events):
            published = t0 + pd.Timedelta(seconds=at - rng.uniform(5, 20))
            seen = published + pd.Timedelta(seconds=rng.uniform(3, 40))
            news_id = f"SYN-N-{i:03d}-{e_idx}"
            row = {"news_id": news_id, "event_type": kind, "subject_team_id": team,
                   "subject_player_id": f"SYN-P-{rng.integers(1000, 9999)}" if kind == "injury" else "",
                   "source_id": "SYN-WIRE", "source_url": f"https://example.org/{TAG}/{news_id}",
                   "source_published_at_utc": _iso(published, 1 if i % 4 == 1 else 0),
                   "first_seen_at_utc": _iso(seen), "revision_id": "r1", "confidence": "confirmed",
                   "fixture_id": fid if kind == "lineup_confirmed" else "",
                   "release_scheduled_known_at_utc": _iso(ko - pd.Timedelta(days=2)) if kind == "lineup_confirmed" else "",
                   "severity": round(sev, 3)}
            if i == special["no_timestamp"] and kind == "injury":
                row["first_seen_at_utc"] = ""
            if i == special["rumour"] and kind == "injury":
                rumour = {**row, "revision_id": "r0", "confidence": "rumour",
                          "source_published_at_utc": _iso(published - pd.Timedelta(minutes=50)),
                          "first_seen_at_utc": _iso(seen - pd.Timedelta(minutes=50))}
                news.append(rumour)
            news.append(row)

    q = pd.DataFrame(quotes, columns=QUOTE_FIELDS)
    for col in ("provider_published_at_utc", "received_at_utc"):
        ms = q[col].to_numpy().astype("datetime64[ns]").astype("datetime64[ms]")
        q[col] = np.char.add(np.datetime_as_string(ms, unit="ms"), "Z")
    q.insert(0, "venue", "SYNEX")
    q.insert(q.columns.get_loc("reconnect"), "in_play", False)
    q["source"] = TAG
    # Inject quality failures that the validator must reject and report.
    dup = q[q["status"] == "OPEN"].sample(25, random_state=seed)
    crossed = q[q["status"] == "OPEN"].sample(5, random_state=seed + 1).copy()
    crossed["best_back_odds"], crossed["best_lay_odds"] = crossed["best_lay_odds"] + 0.5, crossed["best_back_odds"]
    crossed["sequence"] = crossed["sequence"] + 0.5
    q = pd.concat([q, dup, crossed]).sort_values(["received_at_utc", "sequence"], kind="stable")
    n_df = pd.DataFrame(news)
    n_df["raw_hash"] = n_df.apply(lambda r: hashlib.sha256(
        f"{r.news_id}|{r.revision_id}|{r.event_type}|{r.subject_team_id}".encode()).hexdigest(), axis=1)
    q["sequence"] = q["sequence"].map(lambda v: str(int(v)) if float(v).is_integer() else str(v))
    return {"markets": pd.DataFrame(markets), "quotes": q.reset_index(drop=True), "news_events": n_df,
            "fixtures_results": pd.DataFrame(fixtures)}


QUOTE_FIELDS = ["market_id", "selection_id", "provider_published_at_utc", "received_at_utc", "status",
                "best_back_odds", "best_back_size", "best_lay_odds", "best_lay_size", "sequence", "reconnect"]


def _latency_ns(rng: np.random.Generator) -> int:
    return round(float(rng.lognormal(np.log(120), 0.5)) * 1_000_000)


def _row(market_id, selection_id, ts_ns, latency_ns, status, back, back_size, lay, lay_size, seq, reconnect) -> tuple:
    return (market_id, selection_id, ts_ns, ts_ns + latency_ns, status, float(back), back_size, float(lay),
            lay_size, seq, bool(reconnect))


def write(output: Path, **kwargs) -> dict[str, int]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tables = generate(**kwargs)
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    return {name: len(frame) for name, frame in tables.items()}
