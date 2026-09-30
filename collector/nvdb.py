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
import math
import re
from collections.abc import Iterator
from typing import Any

import requests
from pyproj import Transformer

log = logging.getLogger("collector.nvdb")

BASE = "https://nvdbapiles.atlas.vegvesen.no"
STATION_OBJECT_TYPE = 153
ROAD_LINK_ENDPOINT = f"{BASE}/vegnett/veglenkesekvenser/segmentert"
PAGE_SIZE = 1000
# Pedestrian/bike infrastructure shares the road network dataset; we only want
# real driving road criticality, so these typeVeg values are dropped.
_NON_ROAD_TYPES = ("sykkel", "gang")

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


def _route_filter(route_code: str) -> str | None:
    """"E6" -> "EV6", "F866" -> "FV866": NVDB's vegsystemreferanse filter format is
    <vegkategori><fase><nummer>, and "V" (Vedtatt/normal) is what real numbered
    public routes use. Returns None for a code we can't parse (e.g. "unnamed")."""
    m = re.match(r"^([A-Za-z]+)(\d+)$", route_code)
    if not m:
        return None
    category, number = m.groups()
    return f"{category.upper()}V{number}"


def _parse_linestring_wkt(wkt: str) -> list[tuple[float, float]]:
    """"LINESTRING Z (x y z, x y z, ...)" or "LINESTRING (x y, ...)" -> [(x, y), ...]
    in the source CRS, dropping any Z."""
    inner = wkt.split("(", 1)[1].rstrip(")")
    points = []
    for part in inner.split(","):
        x_str, y_str, *_ = part.split()
        points.append((float(x_str), float(y_str)))
    return points


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def fetch_road_links_near_stations(
    stations: list[dict[str, Any]],
    county: int = 55,
    max_distance_km: float = 12.0,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """Real road-network geometry (LineStrings) for the routes our stations sit
    on, kept to links within max_distance_km of at least one station — demo
    scope (docs/adr/0006-real-data-integration.md Phase 1b), not a full county
    or national road network. At national scale (post-MVP), this per-station
    radius filter is dropped and every county's road network is loaded, likely
    served as vector tiles rather than embedded line geometry client-side.
    """
    session = session or nvdb_session()
    route_codes = {c for s in stations if (c := _route_filter(s["road_name"]))}
    log.info("fetching road links for routes: %s", sorted(route_codes))

    links: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for route in route_codes:
        params = {"fylke": county, "vegsystemreferanse": route, "antall": PAGE_SIZE}
        for obj in _iter_pages(session, ROAD_LINK_ENDPOINT, params):
            if obj["veglenkesekvensid"] in seen_ids:
                continue
            type_veg = (obj.get("typeVeg") or "").lower()
            if any(t in type_veg for t in _NON_ROAD_TYPES):
                continue
            geometri = obj.get("geometri")
            if not geometri or not geometri["wkt"].upper().startswith("LINESTRING"):
                continue

            xy_points = _parse_linestring_wkt(geometri["wkt"])
            coords = [_TO_WGS84.transform(x, y) for x, y in xy_points]  # (lon, lat) per point
            mid_lon, mid_lat = coords[len(coords) // 2]
            if not any(
                _haversine_km(mid_lat, mid_lon, s["lat"], s["lon"]) <= max_distance_km for s in stations
            ):
                continue

            seen_ids.add(obj["veglenkesekvensid"])
            links.append(
                {
                    "veglenkesekvensid": obj["veglenkesekvensid"],
                    "route_code": route,
                    "coords": [{"lon": lon, "lat": lat} for lon, lat in coords],
                }
            )
    log.info("kept %d road links within %gkm of a station", len(links), max_distance_km)
    return links


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    stations = fetch_stations()
    for s in stations[:5]:
        print(s)
    print(f"... {len(stations)} stations total")


if __name__ == "__main__":
    main()
