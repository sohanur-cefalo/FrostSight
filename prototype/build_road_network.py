"""Phase 1b of docs/adr/0006-real-data-integration.md: real road-link geometry
(not just station points), colored on the map by risk of the nearest real
weather station.

Requires segments.csv to already hold real station coordinates
(prototype/build_dataset.py). Demo scope: only links within a station's
neighborhood radius are kept (collector/nvdb.py's fetch_road_links_near_stations),
not the whole county's road network. At national scale, post-MVP, that radius
filter is dropped and every county is loaded — see the ADR.

Run: python -m prototype.build_road_network
"""

from __future__ import annotations

import json
import logging
import os

import pandas as pd

from collector.nvdb import fetch_road_links_near_stations

log = logging.getLogger("prototype.build_road_network")

ARTIFACTS = "prototype/artifacts"
ROAD_LINKS_PATH = f"{ARTIFACTS}/road_links.json"


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    segments_path = f"{ARTIFACTS}/segments.csv"
    if not os.path.exists(segments_path):
        raise SystemExit(
            f"{segments_path} not found — run `python -m prototype.build_dataset` first "
            "(real station coordinates are what road links get matched against)."
        )

    segments = pd.read_csv(segments_path)
    # fetch_road_links_near_stations reads a station's route from "road_name" —
    # build_dataset.py puts the plain route code (e.g. "E6") in "route_code" and
    # a display label (e.g. "E6 · Nordnes") in "road_name", so use route_code here.
    stations = segments[["road_segment_id", "lat", "lon"]].copy()
    stations["road_name"] = segments["route_code"] if "route_code" in segments else segments["road_name"]
    station_records = stations.to_dict("records")

    links = fetch_road_links_near_stations(station_records)
    with open(ROAD_LINKS_PATH, "w") as f:
        json.dump(links, f)
    log.info("Wrote %d real road links to %s", len(links), ROAD_LINKS_PATH)


if __name__ == "__main__":
    main()
