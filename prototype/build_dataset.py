"""Phase 1 of docs/adr/0006-real-data-integration.md: real Troms road-weather
station geometry standing in for generate_data.py's interpolated fake segments.

Segments are now real NVDB road-weather station locations (object type 153,
`collector/nvdb.py`) with real elevation (`collector/elevation.py`), instead of
a straight line interpolated between two hand-picked coordinates. Weather
observations are still synthetic (Phase 2/3 in the ADR replace those) — this
script reuses generate_data.make_observations() unchanged so the only thing
that changes is where the segments actually are.

Run: python -m prototype.build_dataset
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from collector.elevation import fetch_elevations
from collector.nvdb import fetch_stations
from prototype.generate_data import make_observations
from prototype.provenance import write_meta

log = logging.getLogger("prototype.build_dataset")

PILOT_COUNTY = 55  # Troms, ADR-0002

# Elevation range (m) used to normalize the wind/shade exposure placeholder below.
# Troms road-weather stations run roughly sea level to a few hundred meters;
# this is a documented approximation (ADR-0001), not a validated terrain model.
_EXPOSURE_ELEVATION_SPAN = 400.0


def make_real_segments(county: int = PILOT_COUNTY) -> pd.DataFrame:
    stations = fetch_stations(county=county)
    if not stations:
        raise RuntimeError(f"NVDB returned no road-weather stations for county {county}")

    segments = pd.DataFrame(stations)
    # North-to-south order so consecutive rows trace the real corridor, the same
    # convention generate_data.py used (sort by road_segment_id recovers the path).
    segments = segments.sort_values("lat", ascending=False).reset_index(drop=True)

    segments["elevation_m"] = np.round(
        fetch_elevations(segments["lat"].tolist(), segments["lon"].tolist()), 1
    )

    # Placeholder wind/shade exposure factor (ADR-0001): higher elevation stands in
    # for more wind exposure until Phase 3 lands real terrain/wind climatology.
    exposure = np.clip(segments["elevation_m"] / _EXPOSURE_ELEVATION_SPAN, 0.05, 1.0)
    segments["exposure_factor"] = exposure.round(2)

    segments["road_segment_id"] = [f"{county}-{i:03d}" for i in range(len(segments))]
    segments["route_code"] = segments["road_name"]  # e.g. "E6", for grouping same-road hops on the map
    segments["road_name"] = segments["name"]  # the place name only (e.g. "Arnøya"), not "F7940 · Arnøya" —
    # route_code already carries the route, so the hover/UI shouldn't repeat it inside road_name too.

    return segments[
        ["road_segment_id", "road_name", "route_code", "lat", "lon", "elevation_m", "exposure_factor"]
    ]


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    segments = make_real_segments()
    observations = make_observations(segments)
    segments.to_csv("prototype/artifacts/segments.csv", index=False)
    observations.to_csv("prototype/artifacts/observations.csv", index=False)
    write_meta(segments_source="real_nvdb", observations_source="synthetic")
    log.info(
        "Wrote %d real NVDB segments and %d synthetic observations to prototype/artifacts/",
        len(segments),
        len(observations),
    )


if __name__ == "__main__":
    main()
