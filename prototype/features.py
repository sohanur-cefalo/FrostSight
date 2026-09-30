"""Shared feature engineering for observations, used by both train.py and risk.py.

surface_temp_trend_1h used to be a hardcoded diff(6) — correct only because
generate_data.py's synthetic series is exactly 10-minute spaced (6 steps = 1h).
Phase 2 (docs/adr/0006-real-data-integration.md) adds real MET forecast data,
which is hourly spaced, so a fixed row-count lag no longer means "1 hour ago"
for every source. This computes the trend from actual elapsed time instead.
"""

from __future__ import annotations

import pandas as pd

TREND_WINDOW = pd.Timedelta(hours=1)
TREND_TOLERANCE = pd.Timedelta(hours=2)  # accept "closest to 1h ago" within this window


def add_surface_temp_trend_1h(obs: pd.DataFrame) -> pd.DataFrame:
    """Add surface_temp_trend_1h: surface_temp_c minus the closest prior reading
    (same road_segment_id) approximately one hour earlier, 0.0 if none exists
    within tolerance (e.g. a segment's first observation).
    """
    obs = obs.sort_values(["road_segment_id", "event_time"]).reset_index(drop=True)
    obs["_target_time"] = obs["event_time"] - TREND_WINDOW

    past = obs[["road_segment_id", "event_time", "surface_temp_c"]].rename(
        columns={"event_time": "_past_time", "surface_temp_c": "_past_temp"}
    )

    merged = pd.merge_asof(
        obs.sort_values("_target_time"),
        past.sort_values("_past_time"),
        left_on="_target_time",
        right_on="_past_time",
        by="road_segment_id",
        direction="backward",
        tolerance=TREND_TOLERANCE,
    )
    merged["surface_temp_trend_1h"] = (merged["surface_temp_c"] - merged["_past_temp"]).fillna(0.0)
    return (
        merged.sort_values(["road_segment_id", "event_time"])
        .drop(columns=["_target_time", "_past_time", "_past_temp"])
        .reset_index(drop=True)
    )
