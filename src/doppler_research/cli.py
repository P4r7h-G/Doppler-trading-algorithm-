"""Command-line entry point. Outputs are reproducible CSV/JSON artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

import pandas as pd
import numpy as np
import scipy

from .io import convert_nasdaq, package_three, sha256
from .pipeline import Config, build_event_panel, walk_forward, forecast_latest
from .schema import validate_events, validate_prices
from .scenarios import calibrated_price_intervals
from .scoring import score_forecasts


def run_files(prices_path: Path, events_path: Path, output: Path, config: Config,
              drift: float | None = None) -> dict:
    prices = pd.read_csv(prices_path)
    events = pd.read_csv(events_path)
    panel, skipped = build_event_panel(prices, events, config)
    forecasts = walk_forward(panel, config)
    scored, summary, comparisons = score_forecasts(forecasts, config.bootstrap_replicates, config.seed)
    output.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output / "event_features.csv", index=False)
    skipped.to_csv(output / "skipped_events.csv", index=False)
    scored.to_csv(output / "per_event_forecasts.csv", index=False)
    summary.to_csv(output / "common_sample_scores.csv", index=False)
    comparisons.to_csv(output / "bootstrap_comparisons.csv", index=False)
    if drift is not None:
        calibrated_price_intervals(scored, drift).to_csv(output / "price_scenarios.csv", index=False)
    manifest = {"software_version": "0.1.0", "status": "new_run_not_prior_study_replication",
                "environment": {"python": platform.python_version(), "numpy": np.__version__,
                                "pandas": pd.__version__, "scipy": scipy.__version__},
                "price_file_sha256": sha256(prices_path), "event_file_sha256": sha256(events_path),
                "config": config.__dict__, "drift_log_return_assumption": drift,
                "usable_events": len(panel), "skipped_events": len(skipped),
                "common_oos_events": scored["event_id"].nunique(),
                "tickers": sorted(panel["ticker"].unique().tolist())}
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="doppler-research", description="Causal event volatility research")
    sub = p.add_subparsers(dest="command", required=True)
    v = sub.add_parser("validate", help="Validate canonical user CSVs")
    v.add_argument("--prices", type=Path, required=True)
    v.add_argument("--events", type=Path, required=True)
    c = sub.add_parser("convert-nasdaq", help="Convert a NASDAQ download with explicit provenance")
    c.add_argument("--input", type=Path, required=True)
    c.add_argument("--ticker", required=True)
    c.add_argument("--source", required=True)
    c.add_argument("--adjustment", required=True, choices=["split_adjusted"])
    c.add_argument("--output", type=Path, required=True)
    r = sub.add_parser("run", help="Walk forward and write per-event plus pooled scores")
    r.add_argument("--prices", type=Path, required=True)
    r.add_argument("--events", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--config", type=Path)
    r.add_argument("--drift-log-return", type=float,
                   help="Explicit per-session drift; enables empirically calibrated historical price intervals")
    d = sub.add_parser("package-data", help="Make three separate data ZIPs from validated canonical inputs")
    d.add_argument("--prices", type=Path, required=True)
    d.add_argument("--events", type=Path, required=True)
    d.add_argument("--output", type=Path, required=True)
    n = sub.add_parser("forecast-next", help="Forecast a verified after-close event at the latest available price close")
    n.add_argument("--prices", type=Path, required=True)
    n.add_argument("--events", type=Path, required=True)
    n.add_argument("--event-id", required=True)
    n.add_argument("--output", type=Path, required=True)
    n.add_argument("--config", type=Path)
    n.add_argument("--drift-log-return", type=float,
                   help="Explicit drift assumption; requires 40 earlier OOS events for a price interval")
    return p


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.command == "validate":
        a = validate_prices(pd.read_csv(args.prices))
        b = validate_events(pd.read_csv(args.events))
        print(json.dumps({"prices": len(a), "events": len(b), "tickers": sorted(a.ticker.unique().tolist())}))
    elif args.command == "convert-nasdaq":
        converted = convert_nasdaq(args.input, args.ticker, args.source, args.adjustment)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        converted.to_csv(args.output, index=False)
        print(json.dumps({"output": str(args.output), "rows": len(converted), "input_sha256": sha256(args.input)}))
    elif args.command == "run":
        config = Config(**json.loads(args.config.read_text(encoding="utf-8"))) if args.config else Config()
        print(json.dumps(run_files(args.prices, args.events, args.output, config, args.drift_log_return), indent=2))
    elif args.command == "package-data":
        print("\n".join(map(str, package_three(args.prices, args.events, args.output))))
    elif args.command == "forecast-next":
        config = Config(**json.loads(args.config.read_text(encoding="utf-8"))) if args.config else Config()
        prices, events = pd.read_csv(args.prices), pd.read_csv(args.events)
        result, intervals = forecast_latest(prices, events, args.event_id, config, args.drift_log_return)
        args.output.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output / "next_event_variance_forecasts.csv", index=False)
        intervals.to_csv(args.output / "next_event_price_intervals.csv", index=False)
        manifest = {"status": "conditional_variance_forecast", "event_id": args.event_id,
                    "price_file_sha256": sha256(args.prices), "event_file_sha256": sha256(args.events),
                    "config": config.__dict__, "drift_log_return_assumption": args.drift_log_return,
                    "interval_rows": len(intervals)}
        (args.output / "forecast_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
