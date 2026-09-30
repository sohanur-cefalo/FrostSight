# FrostSight prototype

Local, no-credentials MVP: synthetic road-weather data -> a small PyTorch icing-risk model ->
a Plotly Dash dashboard. See `docs/adr/0005-mvp-pivot-local-prototype.md` for why this replaced
the Databricks milestone plan as the active track.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r prototype/requirements.txt

python -m prototype.train   # generates synthetic data (if artifacts/ is empty) + trains the model
python -m prototype.app     # http://127.0.0.1:8050
```

To use real road segment data instead of the synthetic corridor, run this before `train`:

```bash
python -m prototype.build_dataset   # real Troms NVDB road-weather stations -> segments.csv
python -m prototype.train           # trains on artifacts/observations.csv, whichever script wrote it
```

## What's in here

| File | Role |
|---|---|
| `generate_data.py` | Synthetic Troms-county road segments + 10-min weather series, with a documented heuristic label |
| `build_dataset.py` | Phase 1 real-data path (`docs/adr/0006-real-data-integration.md`): real NVDB road-weather station geometry via `collector/nvdb.py` + `collector/elevation.py`, still-synthetic weather sited on those real coordinates |
| `model.py` | `IcingRiskNet`: small PyTorch feed-forward regressor, risk-level thresholds |
| `train.py` | Trains the model on whatever's in `artifacts/observations.csv`, saves `artifacts/icing_model.pt` |
| `risk.py` | Scores the latest observation per segment, produces risk level + plain-English drivers |
| `app.py` | Dash app: risk map, road detail, gritting priority list, platform health |

## Known limits (say this out loud in the demo)

- By default (`generate_data.py`) segments and weather are both synthetic. `build_dataset.py` swaps in
  real NVDB road-weather station locations/elevation (Phase 1 of ADR-0006) but weather observations are
  still synthetic until Phase 2/3 (MET/Frost/DATEX) land — see the ADR for the staged plan.
- Real stations don't form one connected corridor the way the synthetic route does — the map only draws a
  connecting line between two real stations when they're on the same numbered route and close together;
  most stations render as independent points, which is the honest picture of what NVDB gives us today
  (point locations, not the connecting road-link geometry).
- No historical accident validation yet.
- Swapping in more real data: point `generate_data.py`/`build_dataset.py`'s output shape (`segments.csv`,
  `observations.csv`) at a real extract from `collector/`, retrain, no dashboard changes needed.
