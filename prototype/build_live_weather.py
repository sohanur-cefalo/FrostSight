"""Phase 2 of docs/adr/0006-real-data-integration.md: real live weather inputs
(air temperature, wind speed, precipitation) from MET Locationforecast,
replacing generate_data.py's synthetic storm-curve simulation.

Requires segments.csv to already hold real station coordinates
(prototype/build_dataset.py, Phase 1) — MET forecasts are per real coordinate,
so there's nothing real to fetch for the interpolated synthetic corridor.

surface_temp_c is still an approximation from air temperature (the offset
heuristic already used in generate_data.py), because MET Locationforecast has
no road-surface sensor — only DATEX (Phase 3, pending account approval) gives
a real measured value for that column. This is documented, not silently
swapped for ground truth (see prototype/README.md "Known limits").

Run: python -m prototype.build_live_weather
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from collector.met import fetch_forecasts_for_stations
from prototype.generate_data import heuristic_icing_risk
from prototype.provenance import read_meta, write_meta

log = logging.getLogger("prototype.build_live_weather")

ARTIFACTS = "prototype/artifacts"
FORECAST_HOURS = 6  # matches generate_data.py's HOURS_OF_HISTORY window length


def make_live_observations(segments: pd.DataFrame) -> pd.DataFrame:
    stations = segments[["road_segment_id", "lat", "lon"]].to_dict("records")
    forecasts = fetch_forecasts_for_stations(stations, hours=FORECAST_HOURS)
    exposure_by_segment = segments.set_index("road_segment_id")["exposure_factor"]

    rows = []
    for road_segment_id, points in forecasts.items():
        if not points:
            log.warning("no MET forecast points for %s, skipping", road_segment_id)
            continue
        exposure = exposure_by_segment[road_segment_id]
        air_temp = np.array([p["air_temp_c"] for p in points])
        wind_speed = np.array([p["wind_speed_ms"] for p in points])
        precip_mm = np.array([p["precip_mm"] for p in points])
        # Same documented offset heuristic as generate_data.py's synthetic path,
        # now driven by real air temperature instead of a simulated storm curve.
        surface_temp = air_temp - 1.0 - 1.5 * exposure
        risk = heuristic_icing_risk(surface_temp, air_temp, wind_speed, precip_mm, exposure)

        for k, point in enumerate(points):
            rows.append(
                {
                    "road_segment_id": road_segment_id,
                    "event_time": point["time"],
                    "air_temp_c": round(float(air_temp[k]), 2),
                    "surface_temp_c": round(float(surface_temp[k]), 2),
                    "wind_speed_ms": round(float(wind_speed[k]), 2),
                    "precip_mm": round(float(precip_mm[k]), 2),
                    "heuristic_risk": round(float(risk[k]), 4),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    segments_path = f"{ARTIFACTS}/segments.csv"
    if not os.path.exists(segments_path):
        raise SystemExit(
            f"{segments_path} not found — run `python -m prototype.build_dataset` first "
            "(Phase 1: real station coordinates) before fetching live weather for them."
        )

    segments = pd.read_csv(segments_path)
    observations = make_live_observations(segments)
    observations.to_csv(f"{ARTIFACTS}/observations.csv", index=False)
    write_meta(segments_source=read_meta()["segments_source"], observations_source="real_met_forecast")
    log.info(
        "Wrote %d real MET forecast observations across %d segments to %s/observations.csv",
        len(observations),
        observations["road_segment_id"].nunique(),
        ARTIFACTS,
    )


if __name__ == "__main__":
    main()
