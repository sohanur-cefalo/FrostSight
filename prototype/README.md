# FrostSight prototype

Local, no-credentials MVP: synthetic road-weather data -> a small PyTorch icing-risk model ->
a Plotly Dash dashboard. See `docs/adr/0005-mvp-pivot-local-prototype.md` for why this replaced
the Databricks milestone plan as the active track.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r prototype/requirements.txt

python -m prototype.train   # trains the model on a fresh synthetic distribution; also seeds
                             # artifacts/segments.csv + observations.csv if they don't exist yet
python -m prototype.app     # http://127.0.0.1:8050
```

To serve real data instead of the synthetic corridor (`docs/adr/0006-real-data-integration.md`):

```bash
python -m prototype.build_dataset       # Phase 1: real Troms NVDB station locations -> segments.csv
python -m prototype.build_road_network  # Phase 1b: real road-link geometry near those stations -> road_links.json
python -m prototype.build_live_weather  # Phase 2: real MET forecast for those stations -> observations.csv
python -m prototype.train               # trains on a synthetic distribution regardless (see below)
python -m prototype.app
```

`train.py` always trains on a freshly generated synthetic distribution, never on whatever's currently
staged in `artifacts/observations.csv` — a live MET forecast snapshot covers one moment in time across 30
segments (often all-LOW on a mild day), which would collapse the model into "always low risk" instead of
learning weather-to-risk in general. The trained model then scores whatever real or synthetic data is
staged for serving (`risk.py`). Run `build_dataset`/`build_live_weather` before or after `train` — order
doesn't matter, since `train` never reads or overwrites the serving files once they exist.

## What's in here

| File | Role |
|---|---|
| `generate_data.py` | Synthetic Troms-county road segments + 10-min weather series, with a documented heuristic label |
| `build_dataset.py` | Phase 1 real-data path: real NVDB road-weather station geometry via `collector/nvdb.py` + `collector/elevation.py`, still-synthetic weather sited on those real coordinates |
| `clean_road_names.py` | One-off fixup for a `segments.csv` written before `build_dataset.py` stopped combining route + place into one string (`"F7940 · Arnøya"` -> `"Arnøya"`); a fresh `build_dataset.py` run doesn't need this |
| `build_road_network.py` | Phase 1b real-data path: real road-link geometry near those stations via `collector/nvdb.py`, written to `road_links.json`; the map colors each real link by its nearest station's risk level instead of only drawing station points |
| `build_live_weather.py` | Phase 2 real-data path: real live air temp/wind/precip per real station via `collector/met.py`; surface temp is still an approximation from air temp until Phase 3 (DATEX) |
| `features.py` | Shared `surface_temp_trend_1h` calc, time-aware so it works whether observations are 10-min synthetic or hourly real forecast points |
| `provenance.py` | Tracks whether the currently-staged segments/observations are synthetic or real, so the dashboard's topbar label is accurate |
| `model.py` | `IcingRiskNet`: small PyTorch feed-forward regressor, risk-level thresholds |
| `train.py` | Trains the model on a synthetic distribution (see above), saves `artifacts/icing_model.pt` |
| `risk.py` | Scores the "current" (closest-to-now) observation per segment, produces risk level + plain-English drivers |
| `app.py` | Dash app: risk map, road detail, gritting priority list, platform health |

## Known limits (say this out loud in the demo)

- By default (`generate_data.py`) segments and weather are both synthetic. `build_dataset.py` (Phase 1)
  swaps in real NVDB station locations/elevation; `build_live_weather.py` (Phase 2) swaps in real live
  MET forecast weather. Both are no-registration sources. Real measured road **surface** temperature (not
  approximated from air temp) needs Phase 3 (Frost/DATEX), which is pending account approval — see the ADR
  for the staged plan.
- Without `build_road_network.py`, real stations don't form one connected corridor the way the synthetic
  route does — the map falls back to only drawing a connecting line between two real stations when they're
  on the same numbered route and close together, and most stations render as independent points.
  `build_road_network.py` fixes this with real road-link geometry, but stays demo-scoped: only links within
  12 km of one of our 30 stations, on the specific routes those stations sit on — not the whole county's
  road network. Scaling to all of Norway post-MVP means dropping that radius filter and rethinking how the
  geometry is served (see the ADR's "Post-MVP scaling note").
- MET Locationforecast is forward-looking (a forecast), not historical — `build_live_weather.py`'s
  "observations" are really the next few hours' forecast, sited at "now" for scoring purposes. Once
  running for a while, real conditions may drift from what was forecast; re-run it to refresh.
- No historical accident validation yet.
- Swapping in more real data: point `generate_data.py`/`build_dataset.py`/`build_live_weather.py`'s output
  shape (`segments.csv`, `observations.csv`) at a real extract from `collector/`, no dashboard or model
  code changes needed.
