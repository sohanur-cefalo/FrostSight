"""Tracks where artifacts/segments.csv and artifacts/observations.csv actually
came from, so app.py's UI can say so accurately instead of a hardcoded label.
Each build script (generate_data.py, build_dataset.py, build_live_weather.py)
calls write_meta() after writing its CSVs.
"""

from __future__ import annotations

import json
import os

ARTIFACTS = "prototype/artifacts"
META_PATH = f"{ARTIFACTS}/meta.json"

DEFAULT_META = {"segments_source": "synthetic", "observations_source": "synthetic"}

LABELS = {
    "synthetic": "synthetic",
    "real_nvdb": "real NVDB road-weather stations",
    "real_met_forecast": "real MET forecast",
}


def write_meta(segments_source: str, observations_source: str) -> None:
    with open(META_PATH, "w") as f:
        json.dump({"segments_source": segments_source, "observations_source": observations_source}, f)


def read_meta() -> dict[str, str]:
    if not os.path.exists(META_PATH):
        return dict(DEFAULT_META)
    with open(META_PATH) as f:
        return {**DEFAULT_META, **json.load(f)}


def describe_sources() -> str:
    meta = read_meta()
    seg_label = LABELS.get(meta["segments_source"], meta["segments_source"])
    obs_label = LABELS.get(meta["observations_source"], meta["observations_source"])
    if seg_label == obs_label == "synthetic":
        return "Synthetic data"
    return f"Segments: {seg_label} · Weather: {obs_label}"
