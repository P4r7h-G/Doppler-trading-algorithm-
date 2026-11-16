"""Run a full study and write reproducible, auditable artifacts."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy

from .audit import FEATURE_REGISTRY, MODEL_REGISTRY, to_jsonable, write_traces
from .config import SportsConfig
from .io import code_fingerprint
from .models import RIDGE_MODELS
from .pipeline import build_panel, prepare, walk_forward
from .scoring import event_counts, score, signed_report

OUTPUT_FILES = {
    "run_manifest.json": "inputs (SHA-256), code fingerprint, config, data declaration, counts",
    "input_manifest.json": "per-file provenance: filename, SHA-256, rows, schema version, licence declaration",
    "rejected_quotes.csv": "quote rows removed by validation, with reason and source row",
    "excluded_events.csv": "news events or event-horizons excluded, with reason",
    "forecast_skips.csv": "event-horizon-model forecasts not produced, with reason",
    "event_panel.csv": "one row per event x horizon: origin, every feature, GARCH fit, targets",
    "per_event_forecasts.csv": "common-sample forecast vs realized target, with losses",
    "common_sample_scores.csv": "pooled QLIKE/MAE/RMSE/bias per panel, horizon and model",
    "breakdown_scores.csv": "same scores by event type, competition, kickoff band, liquidity band, venue",
    "bootstrap_comparisons.csv": "paired QLIKE differences with fixture-clustered bootstrap intervals",
    "gate_checks.csv": "spec acceptance gate: Doppler must beat event+state and event-adjusted GARCH",
    "signed_reprice_report.csv": "signed repricing vs zero-drift baseline (no directional model)",
    "event_counts.csv": "event counts by event type, market type, league, kickoff band, venue",
    "forecast_contributions.csv": "per forecast: each term's raw value, scaling, coefficient, contribution",
    "feature_grid_lineage.csv": "every pre-origin grid point: time, probability, increment, quote source row",
    "target_lineage.csv": "every post-origin target point: time, logit, quote source row",
    "audit/traces.jsonl": "ordered function-by-function trace for every event",
    "feature_registry.json": "which function computes each feature, with formula and units",
}


def _write(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def run_panel(prep, config: SportsConfig, name: str, shift: float = 0.0, shuffled: bool = False,
              keep_lineage: bool = True):
    result = build_panel(prep, config, name, shift, shuffled, keep_lineage)
    forecasts, contributions, skips = walk_forward(result, config)
    return result, forecasts, contributions, skips


def run_study(frames: dict[str, pd.DataFrame], file_manifest: list[dict], config: SportsConfig,
              output: Path | None = None, progress=None) -> dict:
    """Run the main panel plus placebo-timestamp and shuffled-link controls."""
    say = progress or (lambda _msg: None)
    say("validating inputs and building replays")
    prep = prepare(frames, config)
    panels = [("main", 0.0, False)]
    if config.placebo_shift_minutes:
        panels.append(("placebo_timestamps", float(config.placebo_shift_minutes), False))
    if config.shuffled_link_test:
        panels.append(("shuffled_news_links", 0.0, True))
    results, all_forecasts, all_contrib, all_skips = {}, [], [], []
    for name, shift, shuffled in panels:
        say(f"building {name} panel")
        result, forecasts, contributions, skips = run_panel(prep, config, name, shift, shuffled,
                                                            keep_lineage=(name == "main"))
        results[name] = result
        all_forecasts.append(forecasts)
        all_contrib.append(contributions)
        all_skips.append(skips)
    say("scoring")
    forecasts = pd.concat(all_forecasts, ignore_index=True)
    scored, pooled, breakdown, comparisons, gates = score(forecasts, config.bootstrap_replicates, config.seed,
                                                          config.bootstrap_cluster)
    panel = pd.concat([r.panel for r in results.values()], ignore_index=True)
    exclusions = pd.concat([r.exclusions for r in results.values()], ignore_index=True)
    traces = {f"{r_name}::{k}" if r_name != "main" else k: v
              for r_name, r in results.items() for k, v in r.traces.items()}
    for key, trace in traces.items():
        trace.event_key = key
    artifacts = {
        "rejected_quotes": prep.tables["rejected_quotes"], "excluded_events": exclusions,
        "forecast_skips": pd.concat(all_skips, ignore_index=True), "event_panel": panel,
        "per_event_forecasts": scored, "common_sample_scores": pooled, "breakdown_scores": breakdown,
        "bootstrap_comparisons": comparisons, "gate_checks": gates,
        "signed_reprice_report": signed_report(panel) if not panel.empty else pd.DataFrame(),
        "event_counts": event_counts(panel) if not panel.empty else pd.DataFrame(),
        "forecast_contributions": pd.concat(all_contrib, ignore_index=True),
        "feature_grid_lineage": pd.DataFrame(results["main"].grid_lineage),
        "target_lineage": pd.DataFrame(results["main"].target_lineage),
    }
    main_scored = scored[scored["panel"] == "main"] if not scored.empty else scored
    manifest = {
        "status": "sports_market_volatility_study",
        "claim_scope": "forecasts of realized variation of exchange-implied probability paths; "
                       "not match-result probabilities, odds points, direction or trading profit",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code": code_fingerprint(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "scipy": scipy.__version__},
        "inputs": file_manifest, "config": config.to_dict(), "data_declaration": config.data_declaration,
        "bootstrap_seed": config.seed, "models": {**MODEL_REGISTRY,
                                                  **{k: f"ridge on {v}" for k, v in RIDGE_MODELS.items()}},
        "counts": {
            "news_ids": int(prep.tables["news_events"]["news_id"].nunique()),
            "quotes_accepted": int(len(prep.tables["quotes"])),
            "quotes_rejected": int(len(prep.tables["rejected_quotes"])),
            "panel_rows_by_panel": panel.groupby("panel").size().to_dict() if not panel.empty else {},
            "exclusions_by_reason": exclusions[exclusions["panel"] == "main"]["reason"].value_counts().to_dict(),
            "common_sample_events_main_by_horizon": main_scored.groupby("horizon_minutes")["event_key"].nunique()
            .to_dict() if not main_scored.empty else {},
        },
        "outputs": OUTPUT_FILES,
    }
    if output is not None:
        output = Path(output)
        (output / "audit").mkdir(parents=True, exist_ok=True)
        say(f"writing outputs to {output}")
        for name, frame in artifacts.items():
            _write(frame, output / f"{name}.csv")
        write_traces(traces, output / "audit" / "traces.jsonl")
        (output / "feature_registry.json").write_text(json.dumps(FEATURE_REGISTRY, indent=2), encoding="utf-8")
        (output / "input_manifest.json").write_text(json.dumps(to_jsonable(file_manifest), indent=2),
                                                    encoding="utf-8")
        (output / "run_manifest.json").write_text(json.dumps(to_jsonable(manifest), indent=2, default=str),
                                                  encoding="utf-8")
    return {"manifest": manifest, "artifacts": artifacts, "traces": traces, "prepared": prep,
            "results": results}
