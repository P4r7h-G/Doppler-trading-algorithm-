"""Point-in-time news events: revision choice, forecast origin and market link."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SportsConfig

INACTIVE_FIXTURE = {"postponed", "abandoned", "void", "cancelled"}


def choose_revision(revisions: pd.DataFrame, config: SportsConfig) -> tuple[pd.Series | None, str, list[str]]:
    """Earliest-received revision whose confidence is eligible. Later ones are never merged in.

    Returns (row or None, exclusion reason, earlier revision ids).
    """
    ordered = revisions.sort_values(["first_seen_at_utc", "source_row"], na_position="last")
    eligible = ordered[ordered["confidence"].isin(config.eligible_confidence)]
    if eligible.empty:
        return None, "no_revision_with_eligible_confidence", []
    chosen = eligible.iloc[0]
    earlier = ordered[(ordered["first_seen_at_utc"] < chosen["first_seen_at_utc"])]["revision_id"].tolist()
    return chosen, "", earlier


def forecast_origin(news: pd.Series, config: SportsConfig) -> tuple[pd.Timestamp | None, str, dict]:
    """Return (origin, exclusion reason, explanation of the rule applied)."""
    published, seen = news["source_published_at_utc"], news["first_seen_at_utc"]
    skew = pd.Timedelta(seconds=config.max_clock_skew_seconds)
    if config.mode == "post_news":
        if pd.isna(published) or pd.isna(seen):
            return None, "no_firm_event_timestamp", {}
        if seen < published - skew:
            return None, "first_seen_before_published", {}
        available = max(published, seen)
        origin = available + pd.Timedelta(seconds=config.post_news_latency_seconds)
        return origin, "", {"rule": "max(source_published_at_utc, first_seen_at_utc) + post_news_latency_seconds",
                            "source_published_at_utc": published, "first_seen_at_utc": seen,
                            "latency_seconds": config.post_news_latency_seconds}
    known = news["release_scheduled_known_at_utc"]
    if pd.isna(published):
        return None, "no_firm_event_timestamp", {}
    origin = published - pd.Timedelta(minutes=config.pre_release_lead_minutes)
    if pd.isna(known) or known > origin:
        return None, "release_not_scheduled_at_origin", {}
    if not pd.isna(seen) and seen <= origin:
        return None, "news_public_before_pre_release_origin", {}
    return origin, "", {"rule": "source_published_at_utc - pre_release_lead_minutes",
                        "source_published_at_utc": published,
                        "release_scheduled_known_at_utc": known,
                        "lead_minutes": config.pre_release_lead_minutes}


def link_fixture(news: pd.Series, origin: pd.Timestamp, fixtures: pd.DataFrame) -> tuple[pd.Series | None, str, str]:
    """Explicit fixture_id, else the subject team's next fixture after the origin."""
    if not pd.isna(news["fixture_id"]):
        match = fixtures[fixtures["fixture_id"] == news["fixture_id"]]
        if match.empty:
            return None, "", "fixture_not_found"
        return match.iloc[0], "explicit_fixture", ""
    team = news["subject_team_id"]
    candidates = fixtures[((fixtures["home_team_id"] == team) | (fixtures["away_team_id"] == team)) &
                          (fixtures["scheduled_start_utc"] > origin)]
    if candidates.empty:
        return None, "", "no_next_fixture_for_team"
    return candidates.sort_values("scheduled_start_utc").iloc[0], "direct_team_next_fixture", ""


def selection_role(news: pd.Series, fixture: pd.Series, config: SportsConfig) -> tuple[str | None, str]:
    if config.target_selection != "subject_team":
        return config.target_selection, ""
    if news["subject_team_id"] == fixture["home_team_id"]:
        return "home", ""
    if news["subject_team_id"] == fixture["away_team_id"]:
        return "away", ""
    return None, "subject_team_not_in_fixture"


def band(value: float, edges: tuple[float, ...], label: str) -> str:
    if not np.isfinite(value):
        return f"{label}_unknown"
    i = int(np.searchsorted(np.asarray(edges, dtype=float), value, side="right"))
    lo = "-inf" if i == 0 else f"{edges[i - 1]:g}"
    hi = "inf" if i == len(edges) else f"{edges[i]:g}"
    return f"{label}[{lo},{hi})"


def fixture_news_times(news: pd.DataFrame, fixtures: pd.DataFrame) -> dict[str, list[tuple[int, str]]]:
    """(first-seen ns, news_id) of every news revision, any confidence, touching each fixture."""
    out: dict[str, list[tuple[int, str]]] = {}
    for row in news.itertuples(index=False):
        if pd.isna(row.first_seen_at_utc):
            continue
        if not pd.isna(row.fixture_id):
            ids = [row.fixture_id]
        else:
            later = fixtures[((fixtures["home_team_id"] == row.subject_team_id) |
                              (fixtures["away_team_id"] == row.subject_team_id)) &
                             (fixtures["scheduled_start_utc"] > row.first_seen_at_utc)]
            ids = later.sort_values("scheduled_start_utc")["fixture_id"].head(1).tolist()
        for fid in ids:
            out.setdefault(fid, []).append((int(pd.Timestamp(row.first_seen_at_utc).value), row.news_id))
    return {k: sorted(v) for k, v in out.items()}


def is_inactive(fixture: pd.Series) -> bool:
    return fixture["fixture_status"] in INACTIVE_FIXTURE
