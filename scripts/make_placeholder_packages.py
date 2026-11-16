"""Create the three requested package slots with explicit missing-data status."""
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def main() -> None:
    out = Path("data/packages")
    out.mkdir(parents=True, exist_ok=True)
    entries = [
        ("01_aapl_events.zip", {"events.csv": "event_id,ticker,announcement_date,session,event_type,source_url,verified_at,confidence\n"},
         "43 AAPL earnings records are available locally, but pre-event schedule evidence and redistribution review are incomplete"),
        ("02_aapl_prices.zip", {"prices.csv": "ticker,date,open,high,low,close,volume,adjustment,source\n"},
         "AAPL OHLCV is available locally; adjustment provenance and redistribution rights remain unreviewed"),
        ("03_other_companies.zip", {"prices.csv": "ticker,date,open,high,low,close,volume,adjustment,source\n",
                                    "events.csv": "event_id,ticker,announcement_date,session,event_type,source_url,verified_at,confidence\n"},
         "11 other-company price files are locally matched by OHLC; adjustment provenance and event records are unresolved"),
    ]
    for name, members, blocker in entries:
        with ZipFile(out / name, "w", ZIP_DEFLATED) as z:
            for filename, contents in members.items():
                z.writestr(filename, contents)
            z.writestr("manifest.json", json.dumps({"status": "PLACEHOLDER_NO_HISTORICAL_ROWS",
                                                     "blocker": blocker, "row_counts": {key: 0 for key in members}}, indent=2))
    print("Created three clearly marked empty package templates")


if __name__ == "__main__":
    main()
