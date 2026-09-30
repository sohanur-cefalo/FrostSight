"""NVDB API v4 extract: real Troms road-weather station points (object type 153).

Phase 1 of docs/adr/0006-real-data-integration.md. Plain Python, no Spark/Databricks
dependency (ADR-0004), writes flat rows the prototype can turn straight into
segments.csv — unlike docs/plan/02_M1_history_and_reference_data.md's T1.3, which
targets a Databricks landing volume, this is the local-prototype equivalent.

Facts (confirmed live, see docs/source-verification.md): the header X-Client is
mandatory; a 400 without it says "X-Client må være satt"; road-weather stations
(type 153) return in one page for Troms (30 total, no pagination needed); each
object's "Geometri, punkt" property is 2D WKT "POINT(x y)" in EPSG:5973 (UTM 33N
horizontal part is EPSG:25833) with no Z, so elevation comes from a separate
source (elevation.py, Open-Meteo fallback per ADR-0006).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import requests
from pyproj import Transformer

log = logging.getLogger("collector.nvdb")

BASE = "https://nvdbapiles.atlas.vegvesen.no"
STATION_OBJECT_TYPE = 153
PAGE_SIZE = 1000

# UTM 33N (EPSG:25833, same X/Y as NVDB's default EPSG:5973) -> WGS84 lon/lat.
_TO_WGS84 = Transformer.from_crs("EPSG:25833", "EPSG:4326", always_xy=True)

_PROP_NAME = "Navn"
_PROP_STATION_NUMBER = "Målestasjonsnummer"
_PROP_POINT = "Geometri, punkt"


def nvdb_session(client_id: str = "frostsight-prototype") -> requests.Session:
    s = requests.Session()
    s.headers.update({"X-Client": client_id})
    return s


def _iter_pages(session: requests.Session, url: str, params: dict[str, Any] | None) -> Iterator[dict]:
    """Yield objects page by page, following metadata.neste.href until an empty page."""
    while url:
        r = session.get(url, params=params, timeout=180)
        r.raise_for_status()
        body = r.json()
        objects = body.get("objekter", [])
        if not objects:
            return
        yield from objects
        nxt = body.get("metadata", {}).get("neste")
        url, params = (nxt["href"], None) if nxt else (None, None)


def _parse_point_wkt(wkt: str) -> tuple[float, float]:
    """"POINT(x y)" (or "POINT Z (x y z)") -> (x, y) in the source CRS."""
    inner = wkt.split("(", 1)[1].rstrip(")").strip()
    x_str, y_str, *_ = inner.split()
    return float(x_str), float(y_str)


def _road_name(lokasjon: dict) -> str | None:
    """Best-effort human road label from vegsystemreferanser, e.g. "E6"."""
    refs = lokasjon.get("vegsystemreferanser") or []
    if not refs:
        return None
    vegsystem = refs[0].get("vegsystem", {})
    kategori = vegsystem.get("vegkategori")
    nummer = vegsystem.get("nummer")
    if kategori and nummer:
        return f"{kategori}{nummer}"
    return None


def fetch_stations(county: int = 55, session: requests.Session | None = None) -> list[dict[str, Any]]:
    """Real road-weather station points for one county: id, name, station number,
    road label, and WGS84 lat/lon reprojected from NVDB's UTM 33N geometry.
    """
    session = session or nvdb_session()
    params = {"fylke": county, "antall": PAGE_SIZE, "inkluder": "egenskaper,lokasjon"}
    url = f"{BASE}/vegobjekter/{STATION_OBJECT_TYPE}"

    stations: list[dict[str, Any]] = []
    for obj in _iter_pages(session, url, params):
        props = {e["navn"]: e["verdi"] for e in obj.get("egenskaper", [])}
        point_wkt = props.get(_PROP_POINT)
        if not point_wkt:
            continue
        x, y = _parse_point_wkt(point_wkt)
        lon, lat = _TO_WGS84.transform(x, y)
        stations.append(
            {
                "nvdb_id": obj["id"],
                "station_number": props.get(_PROP_STATION_NUMBER),
                "name": props.get(_PROP_NAME) or f"Station {obj['id']}",
                "road_name": _road_name(obj.get("lokasjon", {})) or "unnamed",
                "lat": lat,
                "lon": lon,
            }
        )
    log.info("fetched %d NVDB road-weather stations for county %s", len(stations), county)
    return stations


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    stations = fetch_stations()
    for s in stations[:5]:
        print(s)
    print(f"... {len(stations)} stations total")


if __name__ == "__main__":
    main()
