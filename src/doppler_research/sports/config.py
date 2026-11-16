"""Predeclared experiment settings. Freeze these before inspecting any results."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

DEFAULT_FALLBACK = [
    ["market_type", "competition_id", "event_type", "liquidity_band", "kickoff_band"],
    ["market_type", "event_type", "kickoff_band"],
    ["market_type", "event_type"],
    ["market_type"],
]


@dataclass(frozen=True)
class SportsConfig:
    mode: str = "post_news"
    horizons_minutes: tuple[int, ...] = (5, 30, 120)
    market_types: tuple[str, ...] = ("MATCH_ODDS",)
    eligible_event_types: tuple[str, ...] | None = None
    eligible_confidence: tuple[str, ...] = ("confirmed",)
    target_selection: str = "subject_team"
    probability_basis: str = "raw_midpoint"
    logit_epsilon: float = 1e-4

    post_news_latency_seconds: float = 30.0
    pre_release_lead_minutes: float = 5.0
    simultaneous_news_seconds: float = 120.0

    cadence_seconds: int = 60
    recent_window_minutes: int = 20
    baseline_windows: int = 3
    dynamics_step_minutes: int = 5
    min_window_coverage: float = 0.9
    scale_factor_grid: tuple[float, float, int] = (0.25, 4.0, 81)

    max_quote_age_seconds: float = 900.0
    max_clock_skew_seconds: float = 1.0
    max_spread_probability: float = 0.05
    min_depth: float = 0.0

    target: str = "realized_variation"
    target_sampling: str = "tick"
    target_suspension_policy: str = "exclude"

    garch_fit_minutes: int = 360
    min_garch_observations: int = 120
    trailing_rv_minutes: int = 30
    ewma_lambda: float = 0.94

    liquidity_band_edges: tuple[float, ...] = (1000.0, 10000.0)
    kickoff_band_edges_minutes: tuple[float, ...] = (60.0, 180.0, 720.0)
    comparable_fallback: tuple[tuple[str, ...], ...] = tuple(tuple(level) for level in DEFAULT_FALLBACK)
    min_comparable_events: int = 3

    min_train_events: int = 15
    ridge_alpha: float = 10.0
    forecast_floor: float = 1e-10
    bootstrap_replicates: int = 1000
    bootstrap_cluster: str = "fixture"
    seed: int = 17

    placebo_shift_minutes: float | None = 240.0
    shuffled_link_test: bool = True

    data_declaration: dict = field(default_factory=lambda: {
        "provider": "UNDECLARED", "licence_owner": "UNDECLARED",
        "permitted_use": "UNDECLARED", "capture_method": "UNDECLARED"})

    def __post_init__(self) -> None:
        if self.mode not in {"post_news", "pre_release"}:
            raise ValueError("mode must be post_news or pre_release (choose one per run)")
        if not self.horizons_minutes or any(h <= 0 for h in self.horizons_minutes):
            raise ValueError("horizons_minutes must be positive")
        if self.target_selection not in {"subject_team", "home", "draw", "away"}:
            raise ValueError("target_selection must be subject_team, home, draw or away")
        if self.probability_basis not in {"raw_midpoint", "normalized_midpoint"}:
            raise ValueError("probability_basis must be raw_midpoint or normalized_midpoint")
        if self.target not in {"realized_variation", "net_move_proxy"}:
            raise ValueError("target must be realized_variation or net_move_proxy")
        if self.target_sampling not in {"tick", "grid"}:
            raise ValueError("target_sampling must be tick or grid")
        if self.target_suspension_policy not in {"exclude", "bridge"}:
            raise ValueError("target_suspension_policy must be exclude or bridge")
        if not 0 < self.logit_epsilon < 0.01:
            raise ValueError("logit_epsilon must be in (0, 0.01)")
        if self.cadence_seconds <= 0 or 60 % self.cadence_seconds and self.cadence_seconds % 60:
            raise ValueError("cadence_seconds must divide or be a multiple of 60")
        if self.recent_window_minutes * 60 // self.cadence_seconds < 8:
            raise ValueError("recent window needs at least 8 grid increments")
        if self.baseline_windows < 1 or not 0 < self.min_window_coverage <= 1:
            raise ValueError("invalid baseline/coverage settings")
        if self.min_train_events < 5 or self.ridge_alpha <= 0 or self.bootstrap_replicates < 100:
            raise ValueError("invalid training/bootstrap settings")
        if not 0 < self.ewma_lambda < 1:
            raise ValueError("ewma_lambda must be in (0, 1)")
        if self.bootstrap_cluster not in {"fixture", "calendar_week"}:
            raise ValueError("bootstrap_cluster must be fixture or calendar_week")
        lo, hi, n = self.scale_factor_grid
        if not 0 < lo < 1 < hi or int(n) < 11:
            raise ValueError("scale_factor_grid must bracket 1 with at least 11 points")

    @property
    def steps_per_minute(self) -> float:
        return 60.0 / self.cadence_seconds

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> "SportsConfig":
        tuple_fields = {"horizons_minutes", "market_types", "eligible_event_types",
                        "eligible_confidence", "scale_factor_grid", "liquidity_band_edges",
                        "kickoff_band_edges_minutes"}
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"unknown sports config keys {unknown}")
        values = {}
        for key, value in raw.items():
            if key in tuple_fields and value is not None:
                value = tuple(value)
            elif key == "comparable_fallback":
                value = tuple(tuple(level) for level in value)
            values[key] = value
        return cls(**values)

    @classmethod
    def from_json(cls, path: Path | None) -> "SportsConfig":
        if path is None:
            return cls()
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
