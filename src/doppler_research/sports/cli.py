"""``doppler-research sports ...`` commands. All run from local files with no credentials."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

from .audit import explain, read_traces, to_jsonable
from .config import SportsConfig
from .io import load_blobs, load_input
from .replay import NS, logit_path
from .schema import validate_all


def _horizon(text: str) -> int:
    m = re.fullmatch(r"(\d+)\s*(m|min|h)?", text.strip().lower())
    if not m:
        raise argparse.ArgumentTypeError("horizon like 30m, 2h or 30")
    value = int(m.group(1))
    return value * 60 if m.group(2) == "h" else value


def add_parser(sub: argparse._SubParsersAction) -> None:
    s = sub.add_parser("sports", help="Sports exchange-market repricing/volatility research")
    ss = s.add_subparsers(dest="sports_command", required=True)

    def inputs(p: argparse.ArgumentParser, files: bool = False) -> None:
        p.add_argument("--input", type=Path, help="Directory or ZIP with markets, quotes, news_events, fixtures_results")
        if files:
            for name in ("markets", "quotes", "news", "fixtures"):
                p.add_argument(f"--{name}", type=Path)
        p.add_argument("--config", type=Path, help="JSON config (see sports.example.json)")

    v = ss.add_parser("validate", help="Validate canonical files and report rejected quote rows")
    inputs(v, files=True)
    v.add_argument("--output", type=Path, help="Optional directory for rejected_quotes.csv and input_manifest.json")
    r = ss.add_parser("replay", help="Deterministic replay of book state on the clock grid, for hand audit")
    inputs(r)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--market-id", help="VENUE:MARKET_ID; default all markets")
    u = ss.add_parser("run", help="Walk-forward study with controls, scores and full audit trail")
    inputs(u)
    u.add_argument("--output", type=Path, required=True)
    f = ss.add_parser("forecast", help="Forecast one market/selection at an origin from data received by then")
    inputs(f)
    f.add_argument("--market-id", required=True, help="VENUE:MARKET_ID")
    f.add_argument("--selection-id", required=True)
    f.add_argument("--origin", required=True, help="ISO 8601 with timezone, e.g. 2026-11-16T14:00:00Z")
    f.add_argument("--horizon", type=_horizon, required=True, help="e.g. 30m")
    f.add_argument("--event-type", required=True, help="news event type used for comparable-news history")
    f.add_argument("--output", type=Path)
    e = ss.add_parser("explain", help="Print the function-by-function audit chain for one event")
    e.add_argument("--run", type=Path, required=True, help="Output directory of `sports run`")
    e.add_argument("--event-key", help="Event key; omit with --list to see keys")
    e.add_argument("--horizon", type=_horizon)
    e.add_argument("--model")
    e.add_argument("--list", action="store_true")
    m = ss.add_parser("make-synthetic", help="Write clearly synthetic example inputs")
    m.add_argument("--output", type=Path, required=True)
    m.add_argument("--fixtures", type=int, default=48)
    m.add_argument("--seed", type=int, default=20261116)


def _load(args, config: SportsConfig):
    if getattr(args, "input", None):
        return load_input(args.input, config.data_declaration)
    names = {"markets": args.markets, "quotes": args.quotes, "news_events": args.news,
             "fixtures_results": args.fixtures}
    if any(p is None for p in names.values()):
        raise SystemExit("supply --input DIR|ZIP or all of --markets --quotes --news --fixtures")
    return load_blobs({k: (Path(p).name, Path(p).read_bytes()) for k, p in names.items()}, config.data_declaration)


def _print(obj) -> None:
    print(json.dumps(to_jsonable(obj), indent=2, default=str))


def main(args: argparse.Namespace) -> None:
    cmd = args.sports_command
    if cmd == "make-synthetic":
        from .synthetic import write
        counts = write(args.output, n_fixtures=args.fixtures, seed=args.seed)
        _print({"output": str(args.output), "rows": counts, "status": "SYNTHETIC_TEST_ONLY"})
        return
    if cmd == "explain":
        traces = read_traces(args.run / "audit" / "traces.jsonl")
        if args.list or not args.event_key:
            print("\n".join(sorted(traces)))
            return
        if args.event_key not in traces:
            raise SystemExit(f"unknown event key; run with --list")
        print(explain(traces[args.event_key], args.horizon, args.model))
        return
    config = SportsConfig.from_json(args.config)
    frames, manifest = _load(args, config)
    if cmd == "validate":
        t = validate_all(frames["markets"], frames["quotes"], frames["news_events"], frames["fixtures_results"],
                         config.max_clock_skew_seconds)
        q, n = t["quotes"], t["news_events"]
        summary = {"inputs": manifest,
                   "rows": {k: len(v) for k, v in t.items()},
                   "rejected_quotes_by_reason": t["rejected_quotes"]["reject_reason"].value_counts().to_dict(),
                   "markets": int(t["markets"]["market_key"].nunique()),
                   "fixtures_by_status": t["fixtures_results"]["fixture_status"].value_counts().to_dict(),
                   "quote_received_range_utc": [q["received_at_utc"].min(), q["received_at_utc"].max()],
                   "news_without_first_seen": int(n["first_seen_at_utc"].isna().sum()),
                   "news_without_published": int(n["source_published_at_utc"].isna().sum()),
                   "news_by_type_and_confidence": n.groupby(["event_type", "confidence"]).size()
                   .rename("n").reset_index().to_dict("records")}
        if args.output:
            args.output.mkdir(parents=True, exist_ok=True)
            t["rejected_quotes"].to_csv(args.output / "rejected_quotes.csv", index=False)
            (args.output / "input_manifest.json").write_text(json.dumps(to_jsonable(manifest), indent=2))
        _print(summary)
    elif cmd == "replay":
        from .pipeline import prepare
        prep = prepare(frames, config)
        keys = [args.market_id] if args.market_id else sorted(prep.replays)
        rows = []
        for key in keys:
            replay = prep.replays[key]
            first = replay.first_message_ns()
            if first is None:
                continue
            step = config.cadence_seconds * NS
            start = (first // step + 1) * step
            times = np.arange(start, replay.scheduled_start_ns + 1, step, dtype=np.int64)
            for sid in replay.selections:
                st = replay.state(times, sid)
                x = logit_path(st, config.probability_basis, config.logit_epsilon)
                rows.append(pd.DataFrame({
                    "market_key": key, "selection_id": sid, "selection_role": replay.roles.get(sid),
                    "time_utc": pd.to_datetime(times, utc=True), "valid": st.valid, "reason": st.reason,
                    "p_raw_midpoint": st.p_raw, "p_normalized_midpoint": st.p_norm,
                    f"logit_{config.probability_basis}": x, "spread_probability": st.spread,
                    "depth": st.depth, "quote_source_row": st.source_row}))
        args.output.mkdir(parents=True, exist_ok=True)
        grid = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        grid.to_csv(args.output / "replayed_book_grid.csv", index=False)
        prep.tables["rejected_quotes"].to_csv(args.output / "rejected_quotes.csv", index=False)
        (args.output / "input_manifest.json").write_text(json.dumps(to_jsonable(manifest), indent=2))
        _print({"markets": len(keys), "grid_rows": len(grid),
                "invalid_by_reason": grid.loc[~grid["valid"], "reason"].value_counts().to_dict() if len(grid) else {},
                "output": str(args.output / "replayed_book_grid.csv")})
    elif cmd == "run":
        from .study import run_study
        out = run_study(frames, manifest, config, args.output, progress=lambda m: print(m, file=sys.stderr))
        _print({"output": str(args.output), **out["manifest"]["counts"]})
    elif cmd == "forecast":
        from .pipeline import forecast_at, prepare
        prep = prepare(frames, config)
        result, trace = forecast_at(prep, config, args.market_id, args.selection_id, pd.Timestamp(args.origin),
                                    args.horizon, args.event_type)
        if args.output:
            args.output.mkdir(parents=True, exist_ok=True)
            result.to_csv(args.output / "forecast.csv", index=False)
            (args.output / "forecast_trace.json").write_text(json.dumps(trace.to_dict(), indent=2))
            (args.output / "forecast_explained.txt").write_text(explain(trace.to_dict(), args.horizon))
        print(result[["model", "forecast", "train_events"]].to_string(index=False))
