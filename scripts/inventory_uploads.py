"""Inventory user-owned raw CSVs without guessing ticker identity."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--directory", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--mapping", type=Path, help="Optional independently checked filename-to-ticker matches")
    args = ap.parse_args()
    mapping = pd.read_csv(args.mapping).set_index("filename") if args.mapping else None
    records = []
    for path in sorted(args.directory.glob("*.csv")):
        data = pd.read_csv(path)
        dates = pd.to_datetime(data["Date"], format="%m/%d/%Y") if "Date" in data else pd.Series(dtype="datetime64[ns]")
        match = mapping.loc[path.name] if mapping is not None and path.name in mapping.index else None
        if match is not None:
            day = data.loc[dates == pd.Timestamp(match["match_date"])]
            if len(day) != 1:
                raise ValueError(f"match date absent or duplicated in {path.name}")
            for source_col, match_col in [("Open", "match_open"), ("High", "match_high"),
                                          ("Low", "match_low"), ("Close/Last", "match_close")]:
                observed = float(str(day.iloc[0][source_col]).replace("$", "").replace(",", ""))
                if abs(observed - float(match[match_col])) > 0.011:
                    raise ValueError(f"OHLC mismatch for {path.name}: {source_col}")
        records.append({"filename": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "bytes": path.stat().st_size, "rows": len(data),
                        "min_date": dates.min().strftime("%Y-%m-%d") if len(dates) else None,
                        "max_date": dates.max().strftime("%Y-%m-%d") if len(dates) else None,
                        "ticker": str(match["ticker"]) if match is not None else None,
                        "ticker_status": str(match["identity_status"]) if match is not None else "unresolved",
                        "match_reference_url": str(match["match_reference_url"]) if match is not None else None,
                        "price_adjustment": None,
                        "source": "user-attached NASDAQ-format CSV; original download metadata not retained"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"files": records}, indent=2), encoding="utf-8")
    print(f"Inventoried {len(records)} files; {sum(r['ticker'] is not None for r in records)} externally matched")


if __name__ == "__main__":
    main()
