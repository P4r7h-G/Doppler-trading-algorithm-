"""Canonical sports upload contracts.

Structural problems (missing columns, naive timestamps, duplicate keys) raise
``SchemaError``. Row-level quote-quality failures are removed and returned in a
rejection table with a reason, so nothing disappears silently.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd

SCHEMA_VERSION = "sports-canonical-1"

MARKET_COLUMNS = ["venue", "sport", "competition_id", "fixture_id", "market_id", "market_type",
                  "selection_id", "selection_name", "home_team_id", "away_team_id",
                  "scheduled_start_utc", "market_created_utc", "source"]
QUOTE_COLUMNS = ["venue", "market_id", "selection_id", "provider_published_at_utc", "received_at_utc",
                 "status", "best_back_odds", "best_back_size", "best_lay_odds", "best_lay_size", "source"]
QUOTE_OPTIONAL = ["sequence", "in_play", "traded_volume", "last_traded_price", "reconnect"]
NEWS_COLUMNS = ["news_id", "event_type", "subject_team_id", "subject_player_id", "source_id", "source_url",
                "source_published_at_utc", "first_seen_at_utc", "revision_id", "confidence", "raw_hash"]
NEWS_OPTIONAL = ["fixture_id", "release_scheduled_known_at_utc", "severity", "confirmation_status",
                 "human_review"]
FIXTURE_COLUMNS = ["fixture_id", "competition_id", "home_team_id", "away_team_id",
                   "scheduled_start_utc", "fixture_status", "source"]

QUOTE_STATUSES = {"OPEN", "SUSPENDED", "CLOSED"}
NEWS_CONFIDENCE = {"confirmed", "reported", "rumour"}
FIXTURE_STATUSES = {"scheduled", "played", "postponed", "abandoned", "void", "cancelled"}
SELECTION_ROLES = {"home", "draw", "away"}

_TZ_SUFFIX = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$")


class SchemaError(ValueError):
    pass


def _required(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise SchemaError(f"{label}: missing columns {missing}")
    if frame.empty:
        raise SchemaError(f"{label}: no rows")


def _text(frame: pd.DataFrame, columns: list[str], label: str, nullable: tuple[str, ...] = ()) -> None:
    for col in columns:
        values = frame[col].astype("string").str.strip()
        values = values.mask(values.eq(""))
        if col not in nullable and values.isna().any():
            raise SchemaError(f"{label}.{col}: missing value")
        frame[col] = values


def parse_utc(series: pd.Series, label: str, nullable: bool = False) -> tuple[pd.Series, pd.Series]:
    """Return (UTC timestamps, original text). Naive timestamps are rejected.

    The original text is retained so an auditor can see the source timezone.
    """
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        raw = series.astype("string")
        return series.dt.tz_convert("UTC").astype("datetime64[ns, UTC]"), raw
    if pd.api.types.is_datetime64_dtype(series.dtype):
        raise SchemaError(f"{label}: timezone-naive datetimes are not accepted; supply UTC offsets")
    raw = series.astype("string").str.strip()
    raw = raw.mask(raw.eq(""))
    present = raw.notna()
    if not nullable and not present.all():
        raise SchemaError(f"{label}: missing timestamp")
    naive = present & ~raw.fillna("").str.contains(_TZ_SUFFIX)
    if naive.any():
        example = raw[naive].iloc[0]
        raise SchemaError(f"{label}: timestamp '{example}' has no timezone; use ISO 8601 with Z or an offset")
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns, UTC]")
    if present.any():
        try:
            out.loc[present] = pd.to_datetime(raw[present], format="ISO8601", utc=True)
        except (ValueError, TypeError) as exc:
            raise SchemaError(f"{label}: unparseable ISO 8601 timestamp") from exc
    return out, raw


def _row_hashes(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    parts = [frame[c].astype("string").fillna("") for c in columns]
    text = parts[0].str.cat(parts[1:], sep="\x1f")
    return pd.Series([hashlib.sha256(s.encode("utf-8")).hexdigest()[:16] for s in text], index=frame.index)


def _with_provenance(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    out["source_row"] = np.arange(2, len(out) + 2)
    out["row_hash"] = _row_hashes(out, [c for c in columns if c in out.columns])
    return out


def validate_markets(frame: pd.DataFrame) -> pd.DataFrame:
    _required(frame, MARKET_COLUMNS, "markets")
    m = _with_provenance(frame, MARKET_COLUMNS)
    _text(m, ["venue", "sport", "competition_id", "fixture_id", "market_id", "market_type",
              "selection_id", "selection_name", "home_team_id", "away_team_id", "source"], "markets")
    m["market_type"] = m["market_type"].str.upper()
    for col in ("scheduled_start_utc", "market_created_utc"):
        m[col], m[f"{col}_raw"] = parse_utc(m[col], f"markets.{col}")
    if "selection_role" in m.columns:
        m["selection_role"] = m["selection_role"].astype("string").str.strip().str.lower()
    else:
        m["selection_role"] = pd.Series(pd.NA, index=m.index, dtype="string")
    match_odds = m["market_type"].eq("MATCH_ODDS")
    if (match_odds & ~m["selection_role"].isin(SELECTION_ROLES)).any():
        raise SchemaError("markets.selection_role must be home, draw or away for MATCH_ODDS; "
                          "selection names are not keys")
    m["market_key"] = m["venue"] + ":" + m["market_id"]
    if m.duplicated(["market_key", "selection_id"]).any():
        raise SchemaError("markets: duplicate venue/market_id/selection_id")
    per_market = m.groupby("market_key")[["fixture_id", "scheduled_start_utc", "home_team_id",
                                          "away_team_id", "market_type"]].nunique()
    if (per_market > 1).any().any():
        raise SchemaError("markets: fixture, start time, teams and type must be constant within a market")
    if m[match_odds].duplicated(["market_key", "selection_role"]).any():
        raise SchemaError("markets: a MATCH_ODDS market has two selections with the same role")
    if (m["market_created_utc"] > m["scheduled_start_utc"]).any():
        raise SchemaError("markets: market_created_utc after scheduled_start_utc")
    return m.sort_values(["market_key", "selection_id"]).reset_index(drop=True)


def validate_quotes(frame: pd.DataFrame, markets: pd.DataFrame,
                    max_clock_skew_seconds: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    _required(frame, QUOTE_COLUMNS, "quotes")
    q = _with_provenance(frame, QUOTE_COLUMNS + [c for c in QUOTE_OPTIONAL if c in frame.columns])
    _text(q, ["venue", "market_id", "selection_id", "status", "source"], "quotes")
    q["status"] = q["status"].str.upper()
    bad_status = ~q["status"].isin(QUOTE_STATUSES)
    if bad_status.any():
        raise SchemaError(f"quotes.status must be one of {sorted(QUOTE_STATUSES)}")
    for col in ("provider_published_at_utc", "received_at_utc"):
        q[col], q[f"{col}_raw"] = parse_utc(q[col], f"quotes.{col}")
    for col in ("best_back_odds", "best_back_size", "best_lay_odds", "best_lay_size",
                "traded_volume", "last_traded_price"):
        if col in q.columns:
            q[col] = pd.to_numeric(q[col], errors="coerce")
    if "sequence" in q.columns:
        q["sequence"] = pd.to_numeric(q["sequence"], errors="coerce")
    for flag in ("in_play", "reconnect"):
        if flag in q.columns:
            q[flag] = q[flag].astype("string").str.strip().str.lower().isin(["true", "1", "yes", "t"])
        else:
            q[flag] = False
    q["market_key"] = q["venue"] + ":" + q["market_id"]

    reasons = pd.Series(pd.NA, index=q.index, dtype="string")

    def reject(mask: pd.Series, reason: str) -> None:
        reasons.loc[mask & reasons.isna()] = reason

    known = q.set_index(["market_key", "selection_id"]).index.isin(
        markets.set_index(["market_key", "selection_id"]).index)
    reject(pd.Series(~known, index=q.index), "unknown_market_selection")
    late = q["received_at_utc"] < q["provider_published_at_utc"] - pd.Timedelta(seconds=max_clock_skew_seconds)
    reject(late, "received_before_published")
    is_open = q["status"].eq("OPEN")
    odds = q[["best_back_odds", "best_lay_odds"]]
    sizes = q[["best_back_size", "best_lay_size"]]
    reject(is_open & (~np.isfinite(odds).all(axis=1) | (odds <= 1).any(axis=1)), "odds_not_above_1")
    reject(is_open & (~np.isfinite(sizes).all(axis=1) | (sizes <= 0).any(axis=1)), "nonpositive_size")
    reject(is_open & (q["best_back_odds"] > q["best_lay_odds"]), "crossed_book")
    if "sequence" in frame.columns:
        reject(q["sequence"].isna(), "missing_sequence")
        dup = q.duplicated(["market_key", "selection_id", "sequence"], keep="first")
    else:
        dup = q.duplicated(["market_key", "selection_id", "provider_published_at_utc", "status",
                            "best_back_odds", "best_back_size", "best_lay_odds", "best_lay_size"], keep="first")
    reject(dup, "duplicate_stream_message")

    rejected = q[reasons.notna()].assign(reject_reason=reasons[reasons.notna()])
    clean = q[reasons.isna()].copy()
    clean = clean.sort_values(["market_key", "received_at_utc", "source_row"]).reset_index(drop=True)
    keep = ["source_row", "row_hash", "venue", "market_id", "selection_id", "provider_published_at_utc",
            "received_at_utc", "status", "reject_reason"]
    return clean, rejected[keep].reset_index(drop=True)


def validate_news(frame: pd.DataFrame) -> pd.DataFrame:
    _required(frame, NEWS_COLUMNS, "news_events")
    n = _with_provenance(frame, NEWS_COLUMNS + [c for c in NEWS_OPTIONAL if c in frame.columns])
    _text(n, ["news_id", "event_type", "subject_team_id", "subject_player_id", "source_id", "source_url",
              "revision_id", "confidence", "raw_hash"], "news_events",
          nullable=("subject_player_id", "source_url"))
    n["confidence"] = n["confidence"].str.lower()
    if not n["confidence"].isin(NEWS_CONFIDENCE).all():
        raise SchemaError(f"news_events.confidence must be one of {sorted(NEWS_CONFIDENCE)}")
    for col in ("source_published_at_utc", "first_seen_at_utc"):
        n[col], n[f"{col}_raw"] = parse_utc(n[col], f"news_events.{col}", nullable=True)
    if "release_scheduled_known_at_utc" in n.columns:
        n["release_scheduled_known_at_utc"], _ = parse_utc(
            n["release_scheduled_known_at_utc"], "news_events.release_scheduled_known_at_utc", nullable=True)
    else:
        n["release_scheduled_known_at_utc"] = pd.Series(pd.NaT, index=n.index, dtype="datetime64[ns, UTC]")
    if "fixture_id" in n.columns:
        n["fixture_id"] = n["fixture_id"].astype("string").str.strip().replace("", pd.NA)
    else:
        n["fixture_id"] = pd.Series(pd.NA, index=n.index, dtype="string")
    if n.duplicated(["news_id", "revision_id"]).any():
        raise SchemaError("news_events: duplicate news_id/revision_id; revisions must be separate rows")
    return n.sort_values(["first_seen_at_utc", "news_id", "revision_id"], na_position="last").reset_index(drop=True)


def validate_fixtures(frame: pd.DataFrame) -> pd.DataFrame:
    _required(frame, FIXTURE_COLUMNS, "fixtures_results")
    f = _with_provenance(frame, FIXTURE_COLUMNS)
    _text(f, ["fixture_id", "competition_id", "home_team_id", "away_team_id", "fixture_status", "source"],
          "fixtures_results")
    f["fixture_status"] = f["fixture_status"].str.lower()
    if not f["fixture_status"].isin(FIXTURE_STATUSES).all():
        raise SchemaError(f"fixtures_results.fixture_status must be one of {sorted(FIXTURE_STATUSES)}")
    f["scheduled_start_utc"], f["scheduled_start_utc_raw"] = parse_utc(
        f["scheduled_start_utc"], "fixtures_results.scheduled_start_utc")
    if f["fixture_id"].duplicated().any():
        raise SchemaError("fixtures_results: duplicate fixture_id")
    return f.reset_index(drop=True)


def validate_all(markets: pd.DataFrame, quotes: pd.DataFrame, news: pd.DataFrame,
                 fixtures: pd.DataFrame, max_clock_skew_seconds: float = 1.0) -> dict[str, pd.DataFrame]:
    m = validate_markets(markets)
    f = validate_fixtures(fixtures)
    unknown = sorted(set(m["fixture_id"]) - set(f["fixture_id"]))
    if unknown:
        raise SchemaError(f"markets reference fixtures absent from fixtures_results: {unknown[:5]}")
    q, rejected = validate_quotes(quotes, m, max_clock_skew_seconds)
    n = validate_news(news)
    return {"markets": m, "quotes": q, "news_events": n, "fixtures_results": f, "rejected_quotes": rejected}
