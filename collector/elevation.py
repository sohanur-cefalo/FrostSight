"""Elevation lookup for lat/lon points, batched.

Open-Meteo elevation API only (Kartverket hoydedata was unreachable at both probe
locations on 28 Sep 2026, see docs/source-verification.md finding 3). No key, no
registration.
"""

from __future__ import annotations

import logging
import time

import requests

log = logging.getLogger("collector.elevation")

BASE = "https://api.open-meteo.com/v1/elevation"
BATCH_SIZE = 100  # be polite; Open-Meteo has no documented hard batch limit but this stays well under any URL-length concern
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0


def _get_with_retries(params: dict[str, str]) -> dict:
    """Open-Meteo occasionally answers a transient 503; a couple of retries clear it."""
    for attempt in range(1, MAX_RETRIES + 1):
        r = requests.get(BASE, params=params, timeout=30)
        if r.status_code < 500 or attempt == MAX_RETRIES:
            r.raise_for_status()
            return r.json()
        log.warning("elevation API %s, retrying (%d/%d)", r.status_code, attempt, MAX_RETRIES)
        time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise AssertionError("unreachable")  # loop always returns or raises above


def fetch_elevations(lats: list[float], lons: list[float]) -> list[float]:
    """Elevation in meters for each (lat, lon) pair, same order as input."""
    if len(lats) != len(lons):
        raise ValueError("lats and lons must be the same length")

    elevations: list[float] = []
    for i in range(0, len(lats), BATCH_SIZE):
        batch_lats = lats[i : i + BATCH_SIZE]
        batch_lons = lons[i : i + BATCH_SIZE]
        body = _get_with_retries(
            {
                "latitude": ",".join(str(v) for v in batch_lats),
                "longitude": ",".join(str(v) for v in batch_lons),
            }
        )
        elevations.extend(body["elevation"])
    log.info("fetched elevation for %d points", len(elevations))
    return elevations
