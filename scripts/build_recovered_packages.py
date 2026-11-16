"""Create local, honestly labeled dataset ZIPs from recovered user files.

These ZIPs belong in ignored private_data until redistribution rights are reviewed.
They are archival inputs, not point-in-time verified modelling data.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "private_data/recovered_uploads"
OUT = ROOT / "private_data/dataset_packages"
MAPPING = ROOT / "data/price_file_matches.csv"
PRIOR_EVENTS = ROOT / "private_data/prior_aapl_package/aapl_earnings_events_2016_2026.csv"


def canonical_price(path: Path, ticker: str) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path, dtype=str)
    p = pd.DataFrame({"ticker": ticker,
                      "date": pd.to_datetime(raw["Date"], format="%m/%d/%Y").dt.strftime("%Y-%m-%d"),
                      "open": raw["Open"], "high": raw["High"], "low": raw["Low"],
                      "close": raw["Close/Last"], "volume": raw["Volume"],
                      "adjustment": "unknown",
                      "source": f"user_NASDAQ_format_upload:{path.name};original_download_metadata_missing"})
    for col in ("open", "high", "low", "close", "volume"):
        p[col] = pd.to_numeric(p[col].str.replace("$", "", regex=False).str.replace(",", "", regex=False), errors="raise")
    p = p.sort_values("date").reset_index(drop=True)
    if p["date"].duplicated().any() or p[["open", "high", "low", "close", "volume"]].isna().any().any():
        raise ValueError(f"duplicate dates or missing OHLCV: {path.name}")
    if (p[["open", "high", "low", "close"]] <= 0).any().any() or (p["volume"] < 0).any():
        raise ValueError(f"nonpositive price or negative volume: {path.name}")
    bad_range = (p["high"] < p[["open", "low", "close"]].max(axis=1)) | (p["low"] > p[["open", "high", "close"]].min(axis=1))
    log_returns = np.diff(np.log(p["close"].to_numpy(dtype=float)))
    largest = int(np.argmax(np.abs(log_returns))) + 1
    audit = {"largest_absolute_close_log_return": float(abs(log_returns[largest - 1])),
             "largest_return_date": p.loc[largest, "date"],
             "structural_checks": "numeric complete OHLCV; unique dates; positive OHLC; nonnegative volume",
             "inconsistent_ohlc_rows": int(bad_range.sum()),
             "inconsistent_ohlc_dates": p.loc[bad_range, "date"].tolist(),
             "adjustment_conclusion": "unknown; continuity scan is not vendor adjustment provenance"}
    return p, audit


def canonical_aapl_events() -> pd.DataFrame:
    original = pd.read_csv(PRIOR_EVENTS, dtype=str).fillna("")
    session = original["timing_class"].map({"after_market_close": "after_close", "before_market_open": "before_open"}).fillna("unknown")
    return pd.DataFrame({"event_id": original["event_id"], "ticker": "AAPL",
                         "announcement_date": original["event_date"], "session": session,
                         "event_type": "earnings", "source_url": original["apple_source_url"],
                         "verified_at": "", "confidence": "reported"})


def write_zip(path: Path, files: dict[str, bytes], manifest: dict) -> None:
    manifest["files"] = {name: {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
                         for name, content in files.items()}
    with ZipFile(path, "w", ZIP_DEFLATED) as z:
        for name, content in files.items():
            z.writestr(name, content)
        z.writestr("manifest.json", json.dumps(manifest, indent=2))


def main() -> None:
    mapping = pd.read_csv(MAPPING, dtype=str)
    if set(mapping.ticker) != {"AAPL", "AMAT", "AVGO", "COST", "CSCO", "LIN", "LRCX", "META", "MU", "NVDA", "TSLA", "WMT"}:
        raise ValueError("unexpected ticker mapping")
    OUT.mkdir(parents=True, exist_ok=True)
    aapl_events = canonical_aapl_events()
    write_zip(OUT / "01_aapl_events_and_metadata.zip",
              {"aapl_events_original.csv": PRIOR_EVENTS.read_bytes(),
               "events_canonical_reported.csv": aapl_events.to_csv(index=False).encode("utf-8")},
              {"status": "ARCHIVAL_EVENTS_NOT_POINT_IN_TIME_VERIFIED",
               "earnings_rows": len(aapl_events), "product_release_rows": 0,
               "timing_provenance": "Apple Newsroom after-event release pages; no advance schedule evidence recovered",
               "exact_release_timestamps": "missing where blank in original CSV",
               "model_eligibility": "excluded until pre-announcement schedule evidence is supplied"})
    all_prices, audits, raw_other = [], {}, {}
    for row in mapping.itertuples(index=False):
        path = RAW / row.filename
        if not path.exists():
            raise FileNotFoundError(path)
        p, audit = canonical_price(path, row.ticker)
        all_prices.append(p)
        audits[row.ticker] = {"raw_filename": row.filename, "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                              "rows": len(p), "identity_match_url": row.match_reference_url, **audit}
        if row.ticker != "AAPL":
            raw_other[f"raw/{row.ticker}_{row.filename}"] = path.read_bytes()
    prices = pd.concat(all_prices, ignore_index=True)
    aapl_path = RAW / mapping.loc[mapping.ticker == "AAPL", "filename"].iloc[0]
    write_zip(OUT / "02_aapl_historical_ohlcv.zip",
              {"aapl_nasdaq_format_raw.csv": aapl_path.read_bytes(),
               "prices_canonical_unreviewed.csv": prices.loc[prices.ticker == "AAPL"].to_csv(index=False).encode("utf-8")},
              {"status": "PRICE_ADJUSTMENT_UNVERIFIED", "ticker": "AAPL", "audit": audits["AAPL"],
               "model_eligibility": "requires split-adjustment review and explicit attestation"})
    other = prices.loc[prices.ticker != "AAPL"]
    raw_other["prices_canonical_unreviewed.csv"] = other.to_csv(index=False).encode("utf-8")
    raw_other["events.csv"] = b"event_id,ticker,announcement_date,session,event_type,source_url,verified_at,confidence\n"
    raw_other["price_file_matches.csv"] = MAPPING.read_bytes()
    write_zip(OUT / "03_other_companies_prices_and_events.zip", raw_other,
              {"status": "PRICES_PRESENT_EVENTS_MISSING_ADJUSTMENT_UNVERIFIED",
               "tickers": sorted(other.ticker.unique().tolist()), "price_rows": len(other), "event_rows": 0,
               "price_audit": {key: value for key, value in audits.items() if key != "AAPL"},
               "model_eligibility": "requires split-adjustment review and dated advance event sources"})
    for path in sorted(OUT.glob("*.zip")):
        print(path.name, path.stat().st_size)


if __name__ == "__main__":
    main()
