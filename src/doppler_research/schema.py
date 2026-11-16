"""Strict, explicit input contracts; no guessed tickers or event timing."""
from __future__ import annotations

import pandas as pd

PRICE_COLUMNS = ["ticker", "date", "open", "high", "low", "close", "volume", "adjustment", "source"]
EVENT_COLUMNS = ["event_id", "ticker", "announcement_date", "session", "event_type", "source_url", "verified_at", "confidence"]
SESSIONS = {"before_open", "after_close", "during_session", "unknown"}


class SchemaError(ValueError):
    pass


def _required(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise SchemaError(f"{label}: missing columns {missing}")
    if frame.empty:
        raise SchemaError(f"{label}: no rows")


def _date(series: pd.Series, label: str) -> pd.Series:
    try:
        parsed = pd.to_datetime(series, format="%Y-%m-%d", errors="raise")
    except (ValueError, TypeError) as exc:
        raise SchemaError(f"{label}: dates must be ISO YYYY-MM-DD") from exc
    if parsed.isna().any():
        raise SchemaError(f"{label}: missing dates")
    return parsed


def validate_prices(frame: pd.DataFrame) -> pd.DataFrame:
    _required(frame, PRICE_COLUMNS, "prices")
    p = frame[PRICE_COLUMNS].copy()
    p["date"] = _date(p["date"], "prices.date")
    for col in ("ticker", "adjustment", "source"):
        p[col] = p[col].astype("string").str.strip()
        if p[col].isna().any() or p[col].eq("").any():
            raise SchemaError(f"prices.{col}: missing value")
    if not p["adjustment"].eq("split_adjusted").all():
        raise SchemaError("prices.adjustment must be split_adjusted; raw/unknown data need an explicit adjustment review")
    for col in ("open", "high", "low", "close", "volume"):
        p[col] = pd.to_numeric(p[col], errors="coerce")
        if not p[col].notna().all():
            raise SchemaError(f"prices.{col}: nonnumeric or missing value")
    if (p[["open", "high", "low", "close"]] <= 0).any().any() or (p["volume"] < 0).any():
        raise SchemaError("prices: nonpositive OHLC or negative volume")
    if ((p["high"] < p[["open", "low", "close"]].max(axis=1)) | (p["low"] > p[["open", "high", "close"]].min(axis=1))).any():
        raise SchemaError("prices: OHLC range inconsistency")
    if p.duplicated(["ticker", "date"]).any():
        raise SchemaError("prices: duplicate ticker/date")
    return p.sort_values(["ticker", "date"]).reset_index(drop=True)


def validate_events(frame: pd.DataFrame) -> pd.DataFrame:
    _required(frame, EVENT_COLUMNS, "events")
    e = frame[EVENT_COLUMNS].copy()
    e["announcement_date"] = _date(e["announcement_date"], "events.announcement_date")
    raw_verified = e["verified_at"].astype("string")
    present = raw_verified.notna() & raw_verified.ne("")
    e["verified_at"] = pd.NaT
    if present.any():
        e.loc[present, "verified_at"] = _date(raw_verified[present], "events.verified_at").to_numpy()
    for col in ("event_id", "ticker", "session", "event_type", "source_url", "confidence"):
        e[col] = e[col].astype("string").str.strip()
    if e[["event_id", "ticker", "session", "event_type", "confidence"]].isna().any().any():
        raise SchemaError("events: missing identifying metadata")
    if e["event_id"].duplicated().any():
        raise SchemaError("events: duplicate event_id")
    if not e["session"].isin(SESSIONS).all():
        raise SchemaError(f"events.session must be one of {sorted(SESSIONS)}")
    if not e["confidence"].isin(["verified", "reported", "unverified"]).all():
        raise SchemaError("events.confidence must be verified, reported, or unverified")
    verified = e["confidence"].eq("verified")
    if (verified & e["verified_at"].isna()).any():
        raise SchemaError("verified events need a pre-announcement verified_at date")
    if (verified & (e["verified_at"] >= e["announcement_date"])).any():
        raise SchemaError("verified events need verified_at before announcement_date; same-day evidence could arrive after the forecast")
    if (e["confidence"].eq("verified") & (~e["source_url"].str.startswith(("http://", "https://"), na=False))).any():
        raise SchemaError("verified events require a source URL")
    return e.sort_values(["announcement_date", "ticker", "event_id"]).reset_index(drop=True)
