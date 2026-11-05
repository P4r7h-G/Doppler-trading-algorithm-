"""Generate clearly synthetic, reproducible examples for installation smoke tests."""
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    out = Path("examples/synthetic")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261030)
    dates = pd.bdate_range("2020-01-01", periods=600)
    log_returns = rng.normal(0, 0.012, len(dates))
    close = 100 * np.exp(np.cumsum(log_returns))
    open_ = close * np.exp(rng.normal(0, 0.003, len(dates)))
    prices = pd.DataFrame({"ticker": "SYNTH", "date": dates.strftime("%Y-%m-%d"),
                           "open": open_, "high": np.maximum(open_, close) * 1.006,
                           "low": np.minimum(open_, close) * 0.994, "close": close,
                           "volume": rng.integers(100000, 200000, len(dates)),
                           "adjustment": "split_adjusted", "source": "SYNTHETIC_TEST_ONLY"})
    edates = dates[140:-2:16]
    events = pd.DataFrame({"event_id": [f"SYNTH-{i:03}" for i in range(len(edates))],
                           "ticker": "SYNTH", "announcement_date": edates.strftime("%Y-%m-%d"),
                           "session": "after_close", "event_type": "earnings",
                           "source_url": "https://example.org/SYNTHETIC_TEST_ONLY",
                           "verified_at": (edates - pd.Timedelta(days=1)).strftime("%Y-%m-%d"), "confidence": "verified"})
    prices.to_csv(out / "prices.csv", index=False)
    events.to_csv(out / "events.csv", index=False)
    next_event = pd.DataFrame({"event_id": ["SYNTH-NEXT"], "ticker": ["SYNTH"],
                               "announcement_date": [dates[-1].strftime("%Y-%m-%d")],
                               "session": ["after_close"], "event_type": ["earnings"],
                               "source_url": ["https://example.org/SYNTHETIC_TEST_ONLY"],
                               "verified_at": [(dates[-1] - pd.Timedelta(days=1)).strftime("%Y-%m-%d")],
                               "confidence": ["verified"]})
    pd.concat([events, next_event], ignore_index=True).to_csv(out / "events_with_next.csv", index=False)
    print("Synthetic examples written to", out)


if __name__ == "__main__":
    main()
