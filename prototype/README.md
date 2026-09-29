# FrostSight prototype

Local, no-credentials MVP: synthetic road-weather data -> a small PyTorch icing-risk model ->
a Plotly Dash dashboard. See `docs/adr/0005-mvp-pivot-local-prototype.md` for why this replaced
the Databricks milestone plan as the active track.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r prototype/requirements.txt

python -m prototype.train   # generates synthetic data + trains the model into prototype/artifacts/
python -m prototype.app     # http://127.0.0.1:8050
```

## What's in here

| File | Role |
|---|---|
| `generate_data.py` | Synthetic Troms-county road segments + 10-min weather series, with a documented heuristic label |
| `model.py` | `IcingRiskNet`: small PyTorch feed-forward regressor, risk-level thresholds |
| `train.py` | Trains the model on the synthetic labels, saves `artifacts/icing_model.pt` |
| `risk.py` | Scores the latest observation per segment, produces risk level + plain-English drivers |
| `app.py` | Dash app: risk map, road detail, gritting priority list, platform health |

## Known limits (say this out loud in the demo)

- Data is synthetic, not live DATEX/NVDB — the model learns a documented heuristic, not real outcomes.
- No historical accident validation yet.
- Swapping in real data: point `generate_data.py`'s output shape (`segments.csv`, `observations.csv`) at a
  real extract from `collector/`, retrain, no dashboard changes needed.
