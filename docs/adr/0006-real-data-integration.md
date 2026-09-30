# ADR-0006: Real-data integration plan for the prototype

- Status: Proposed
- Date: 2026-09-29
- Owner: Sohanur Rahman

## Context

Per ADR-0005, `prototype/` is the active track: synthetic data (`prototype/generate_data.py`) feeds a
PyTorch model (`prototype/model.py`, `prototype/train.py`) and a Dash dashboard (`prototype/app.py`). The
segments are a straight interpolated line between two hand-picked coordinates with random noise, and the
weather series is a synthetic storm curve — not real road geometry and not real observations.

`docs/source-verification.md` (28 Sep 2026 live checks) establishes what's actually available for Troms
(`fylke=55`, ADR-0002) today, with no waiting:

| Source | Registration needed | Gives us |
|---|---|---|
| NVDB API v4 (road links, stations, speed limits, accidents, avalanche) | No — `X-Client` header only | Real road geometry, real station points, real elevation (Z), real accident/slide history |
| MET Locationforecast | No — `User-Agent` header only | Live air temp, wind, precipitation forecast per coordinate |
| trafikkdata.no GraphQL | No | Real traffic volumes per registration point |
| Open-Meteo elevation | No | Elevation fallback if NVDB Z is dropped |
| MET Frost (historical observations) | Yes — free, email only | Historical 10-min road-weather station observations (air/wind/precip; resolution to confirm) |
| DATEX II (road weather + incidents) | Yes — free, but form asks for a fixed IP/DNS | Live 10-min **road surface temperature** and incidents — the one feature nothing else free gives |

The one feature genuinely blocked behind an account is **surface_temp_c** (road surface temperature),
because MET Locationforecast only reports air temperature. Everything else needed by
`model.FEATURE_COLUMNS` (`air_temp_c`, `wind_speed_ms`, `precip_mm`, `exposure_factor`,
`surface_temp_trend_1h`) can be built from no-registration sources today, `surface_temp_trend_1h` being a
rolling diff computed locally once `surface_temp_c` exists at multiple timestamps.

`prototype/README.md` already documents the intended seam: swap `generate_data.py`'s output
(`segments.csv`, `observations.csv`) for a real extract with the same columns, retrain, no dashboard or
model code changes required. This ADR is that swap plan, staged so real data lands incrementally instead
of blocking on the slowest account approval.

## Options considered

1. **Wait for all accounts (Frost + DATEX), then do one big swap.** Simplest to reason about, but blocks
   real road geometry — which needs no account — behind DATEX's fixed-IP approval, which is out of our
   control and already flagged as an open risk in `docs/source-verification.md`.
2. **Phase by registration requirement**, landing what's available today (NVDB, MET forecast) first and
   layering in Frost/DATEX as each account clears, keeping the same `segments.csv`/`observations.csv`
   contract at every phase so the model/app never need to change shape.

## Decision

Option 2. Three phases, each independently useful and each keeping the existing CSV contract:

### Phase 1 — real station geometry and terrain (done; no registration)

`collector/nvdb.py` + `prototype/build_dataset.py`:

- Pull Troms road-weather station points (`vegobjekter/153`, `fylke=55`) — 30 stations, matching
  `docs/source-verification.md`'s count.
- Reproject NVDB's default WKT (EPSG:5973/25833, UTM 33N) to WGS84 with `pyproj.Transformer` — this is
  what actually fixes the map, replacing interpolated fake coordinates with real station locations across
  Troms (`road_name` carries the real route, e.g. "E6 · Nordnes").
- `elevation_m` from Open-Meteo (NVDB's station points carry no Z; the M1 plan's Z-from-geometry idea only
  applies to line/road objects, not point stations — this ADR's original wording was wrong on that point).
- `exposure_factor` stays a documented placeholder (ADR-0001), derived from real elevation instead of
  `RNG.normal`.
- Output: a real `segments.csv`, one row per real station (not a snapped road-link chunk — see Phase 1b for
  why that distinction matters). `observations.csv` still synthetic at this point.

Acceptance (met): NVDB extract returns 30 stations for Troms; `app.py`'s map shows real station locations
instead of an interpolated line.

### Phase 1b — real road-link geometry, colored by nearest station (done; no registration)

Phase 1 alone only gives 30 points, not a road *shape* — real stations don't sit on one connected
corridor the way the synthetic route did, so the map could only draw sparse dots (see the "known limits"
discussion this phase resolves). `collector/nvdb.py`'s `fetch_road_links_near_stations()` +
`prototype/build_road_network.py` close that gap:

- Pull real road-link LineString geometry (`vegnett/veglenkesekvenser/segmentert`) filtered to the
  specific numbered routes our 30 stations sit on (`vegsystemreferanse=EV6` etc.), not the whole county's
  road network — demo scope, ~750 links for our 30 stations vs. "tens of thousands" for all of Troms.
- Kept only if within 12 km of at least one station (`max_distance_km`), and pedestrian/bike links
  (`typeVeg` containing "sykkel"/"gang") are dropped — this is vehicle road criticality, not a full
  transport network.
- Each kept link is colored by its **nearest** real station's current risk level (haversine to all 30
  stations) — a spatial nearest-neighbor match, since we still only have weather readings at 30 points,
  not continuously along the road. Rendered as one Scattermap trace per risk level (not per link, for
  render performance), each trace's coordinate list broken into disconnected line segments with `None`.
- Output: `prototype/artifacts/road_links.json` — a separate artifact from `segments.csv`, loaded by
  `app.py` if present (falls back to Phase 1's sparse station-connector view if absent).

Acceptance (met): live-tested at 25s / 748 links for all 30 stations; map shows real road-link geometry
(verified visually — dashed lines correctly hug the real coastal road shape near Tromsø) colored by risk.

**Post-MVP scaling note:** this stays demo-scoped (Troms, station-radius-filtered) on purpose. National
scale means dropping the per-station radius filter and loading every county's road network — at that
point embedding full line geometry client-side in a Plotly figure won't scale; the road network would
need to be served as vector tiles (e.g. via a tile server) instead of shipped as inline JSON, and this
ADR's nearest-station coloring approach would need a proper spatial index (e.g. a k-d tree or PostGIS)
rather than an O(links × stations) haversine loop.

### Phase 2 — real live weather inputs (done; no registration)

`collector/met.py` + `prototype/build_live_weather.py`: calls MET Locationforecast per real station
coordinate (from Phase 1) to get live `air_temp_c`, `wind_speed_ms`, `precip_mm`. This replaces the
synthetic storm curve with a real (if forecast-based, not observed) weather signal. `surface_temp_c` is
approximated from air temp with the existing documented offset heuristic
(`surface_temp ≈ air_temp - 1.0 - 1.5 * exposure_factor`, already in `generate_data.py`) until Phase 3
lands the real sensor value — labeled as approximated in `prototype/README.md`'s known limits, not
silently swapped for ground truth. `prototype/provenance.py` tracks which of segments/observations are
real vs. synthetic so the dashboard's topbar states it accurately instead of a hardcoded label.

Two correctness issues surfaced by using genuinely forward-looking (forecast) data instead of historical
data, both fixed: `risk.py` used to pick a segment's *max* `event_time` as "current," which is right for
past-observation data but picks the *most future* forecast point otherwise — now picks whichever reading
is closest to now. And `train.py`/`risk.py`'s trend feature used a hardcoded `diff(6)` assuming fixed
10-minute spacing — replaced with `prototype/features.py`'s time-aware version (`pandas.merge_asof`
against an actual 1-hour window) that works regardless of the data source's cadence. Also: the model is
now always trained on a freshly generated broad synthetic distribution, never on whatever's staged for
serving — a live forecast snapshot is one moment in time across 30 segments (often uniformly low risk on
a mild day), and training on it directly would collapse the model into "always low risk" instead of
learning the general weather-to-risk relationship.

Acceptance (met): live-tested — `observations.csv` populated from live MET calls for all 30 real
Phase-1 stations; dashboard correctly showed all-LOW risk on an actual mild Troms day, confirming it's
scoring real conditions rather than replaying a canned pattern.

### Phase 3 — real historical + live surface temperature (blocked on registration)

- Register a Frost client ID (`docs/source-verification.md` next action, owner E1) and add
  `collector/frost.py` to pull last winter's real 10-min observations for the Troms `STATENS VEGVESEN`
  stations — replaces the synthetic training history so the model trains on real recorded winters, not a
  heuristic curve.
- Submit the DATEX access form and resolve the fixed-IP question (ADR-0004); once approved, add
  `collector/datex.py` for live 10-min real `surface_temp_c` and incidents (situations) — this retires the
  Phase 2 approximation and is the only step that changes what a feature *means*, so retrain after it
  lands.
- Cross-reference NVDB accidents (570) `Føreforhold`/`Værforhold` fields as an optional stretch label
  source for validating (not necessarily retraining) the heuristic risk score against real recorded road
  conditions.

Acceptance: `surface_temp_c` sourced from a real road-weather sensor, not the Phase-2 approximation;
model retrained and re-validated against Frost history.

## Consequences

- `segments.csv`/`observations.csv` column shape never changes across phases — `model.py`, `risk.py`,
  `app.py` need no code changes at any phase, only `generate_data.py`'s replacement (new script, e.g.
  `prototype/build_dataset.py`, calling `collector/*` instead of `RNG`).
- Phases 1 and 2 can ship immediately with no external approval; Phase 3 is explicitly gated on Frost
  registration and the DATEX fixed-IP answer, both already tracked as open actions in
  `docs/source-verification.md`.
- Until Phase 3, `surface_temp_c` is a documented approximation from air temperature, not a measured
  value — the dashboard's "synthetic data" disclosure (per ADR-0005) should be updated to "real geometry +
  approximated surface temp" rather than removed outright.
- `collector/` gets its first real modules (`nvdb.py`, `met.py`, later `frost.py`/`datex.py`), written as
  plain Python with no Spark/Databricks dependency (ADR-0004 already requires this), so they're reusable
  if the Databricks track is ever revived.
