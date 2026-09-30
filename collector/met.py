"""MET Locationforecast (compact) per coordinate: real live air temperature,
wind speed and next-hour precipitation.

Phase 2 of docs/adr/0006-real-data-integration.md. No registration, only a
descriptive User-Agent is required (MET's usage policy). No historical
endpoint — this is a forward-looking forecast series (hourly for the first
~48h), not observed history; see docs/source-verification.md.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

log = logging.getLogger("collector.met")

BASE = "https://api.met.no/weatherapi/locationforecast/2.0/compact"
USER_AGENT = "frostsight-prototype (https://github.com/sohanur-cefalo/FrostSight)"
REQUEST_DELAY_SECONDS = 0.2  # be polite across many sequential station calls


def met_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def fetch_forecast(lat: float, lon: float, session: requests.Session | None = None) -> list[dict[str, Any]]:
    """Hourly forecast points for one coordinate: time, air_temp_c, wind_speed_ms, precip_mm
    (precipitation expected in the hour following each timestamp).
    """
    session = session or met_session()
    r = session.get(BASE, params={"lat": round(lat, 4), "lon": round(lon, 4)}, timeout=30)
    r.raise_for_status()
    timeseries = r.json()["properties"]["timeseries"]

    points: list[dict[str, Any]] = []
    for entry in timeseries:
        instant = entry["data"]["instant"]["details"]
        next_hour = entry["data"].get("next_1_hours", {}).get("details", {})
        points.append(
            {
                "time": entry["time"],
                "air_temp_c": instant["air_temperature"],
                "wind_speed_ms": instant["wind_speed"],
                "precip_mm": next_hour.get("precipitation_amount", 0.0),
            }
        )
    return points


def fetch_forecasts_for_stations(
    stations: list[dict[str, Any]], hours: int = 6
) -> dict[str, list[dict[str, Any]]]:
    """{road_segment_id: first `hours` real forecast points}, one MET call per station."""
    session = met_session()
    forecasts: dict[str, list[dict[str, Any]]] = {}
    for station in stations:
        points = fetch_forecast(station["lat"], station["lon"], session=session)[:hours]
        forecasts[station["road_segment_id"]] = points
        time.sleep(REQUEST_DELAY_SECONDS)
    log.info("fetched MET forecasts for %d stations", len(forecasts))
    return forecasts


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    points = fetch_forecast(69.6496, 18.9560)
    for p in points[:5]:
        print(p)
    print(f"... {len(points)} forecast points total")


if __name__ == "__main__":
    main()
