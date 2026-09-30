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

### Phase 1 — real road geometry and terrain (no registration; can start now)

Add `collector/nvdb.py` (the module `docs/plan/02_M1_history_and_reference_data.md` T1.3 already designs
in detail — reuse its pagination/CRS logic, but write flat CSVs into `prototype/artifacts/` instead of a
Databricks landing volume):

- Pull Troms road links (`vegnett/veglenkesekvenser/segmentert`, `fylke=55`) and road-weather station
  points (`vegobjekter/153`).
- Reproject NVDB's default WKT (EPSG:25833 / UTM 33N) to WGS84 with `pyproj.Transformer` (snippet already
  in the M1 plan doc) — this is what actually fixes the map, replacing interpolated fake coordinates with
  the real E8/E6 corridor shape.
- Segment the road by snapping to the 30 real station locations (one segment per station's nearest road
  link chunk) instead of 30 arbitrary interpolated points — this makes `road_segment_id` map to a real,
  identifiable place.
- Take `elevation_m` from the real Z coordinate in NVDB geometry (fallback: Open-Meteo elevation API, per
  `docs/source-verification.md` finding 3).
- `exposure_factor` stays a documented placeholder (ADR-0001), but derive it from real terrain (elevation
  variance / coastal distance around each real segment) instead of `RNG.normal`.
- Output: a real `segments.csv` with the same columns as today. `observations.csv` still synthetic at
  this point, but now generated over real coordinates/elevation so the storm simulation is at least sited
  correctly.

Acceptance: NVDB extract matches `docs/source-verification.md` counts (30 stations, road links in the
tens of thousands for Troms); map in `app.py` shows the real road path instead of a straight interpolated
line.

### Phase 2 — real live weather inputs (no registration; can start now, independent of Phase 1)

Add `collector/met.py` calling MET Locationforecast per real station coordinate (from Phase 1) to get live
`air_temp_c`, `wind_speed_ms`, `precip_mm`. This replaces the synthetic storm curve with a real (if
forecast-based, not observed) weather signal. `surface_temp_c` is approximated from air temp with the
existing documented offset heuristic
(`surface_temp ≈ air_temp - 1.0 - 1.5 * exposure_factor`, already in `generate_data.py`) until Phase 3
lands the real sensor value — label this column clearly as approximated in the dashboard/README, not
silently swapped for ground truth.

Acceptance: `observations.csv` populated from live MET calls for all real Phase-1 segments; dashboard
"data freshness" reflects real MET response timestamps.

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
