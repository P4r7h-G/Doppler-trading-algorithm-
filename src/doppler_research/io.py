"""User-provided NASDAQ conversion and separated, traceable data packages."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pandas as pd

from .schema import EVENT_COLUMNS, PRICE_COLUMNS, validate_events, validate_prices


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def convert_nasdaq(path: Path, ticker: str, source: str, adjustment: str) -> pd.DataFrame:
    """Ticker and adjustment are explicit user attestations, never inferred."""
    raw = pd.read_csv(path, dtype=str)
    required = {"Date", "Open", "High", "Low", "Close/Last", "Volume"}
    if not required.issubset(raw.columns):
        raise ValueError(f"NASDAQ CSV missing {sorted(required - set(raw.columns))}")
    p = pd.DataFrame({"ticker": ticker.upper(),
                      "date": pd.to_datetime(raw["Date"], format="%m/%d/%Y", errors="raise").dt.strftime("%Y-%m-%d"),
                      "open": raw["Open"], "high": raw["High"], "low": raw["Low"],
                      "close": raw["Close/Last"], "volume": raw["Volume"],
                      "adjustment": adjustment, "source": source})
    for col in ("open", "high", "low", "close"):
        p[col] = p[col].str.replace("$", "", regex=False).str.replace(",", "", regex=False)
    p["volume"] = p["volume"].str.replace(",", "", regex=False)
    return validate_prices(p)


def package_three(prices_path: Path, events_path: Path, out_dir: Path) -> list[Path]:
    """Export AAPL events, AAPL OHLCV, and other tickers in separate ZIPs."""
    prices = validate_prices(pd.read_csv(prices_path))
    events = validate_events(pd.read_csv(events_path))
    if not (prices.ticker == "AAPL").any() or not (events.ticker == "AAPL").any():
        raise ValueError("populated packages require both AAPL prices and AAPL events")
    if not (prices.ticker != "AAPL").any() or not (events.ticker != "AAPL").any():
        raise ValueError("populated packages require both other-company prices and events")
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("01_aapl_events.zip", {"events.csv": events.loc[events.ticker == "AAPL", EVENT_COLUMNS]}),
        ("02_aapl_prices.zip", {"prices.csv": prices.loc[prices.ticker == "AAPL", PRICE_COLUMNS]}),
        ("03_other_companies.zip", {"prices.csv": prices.loc[prices.ticker != "AAPL", PRICE_COLUMNS],
                                    "events.csv": events.loc[events.ticker != "AAPL", EVENT_COLUMNS]}),
    ]
    made = []
    for name, files in specs:
        dest = out_dir / name
        with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as z:
            manifest = {"status": "user_supplied_unverified_by_project", "files": {},
                        "input_price_sha256": sha256(prices_path), "input_event_sha256": sha256(events_path)}
            for member, data in files.items():
                contents = data.to_csv(index=False).encode("utf-8")
                z.writestr(member, contents)
                manifest["files"][member] = {"rows": len(data), "sha256": hashlib.sha256(contents).hexdigest()}
            z.writestr("manifest.json", json.dumps(manifest, indent=2))
        made.append(dest)
    return made
