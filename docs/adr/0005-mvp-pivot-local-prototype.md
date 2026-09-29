# ADR-0005: Replace the Databricks milestone plan with a local PyTorch/Dash prototype track

- Status: Accepted
- Date: 2026-09-29
- Owner: Sohanur Rahman

## Context

The scaffold (ADR-0001 to ADR-0004, `docs/plan/00_README.md` to `13_stretch_review.md`) commits FrostSight
to a Databricks-first build: Unity Catalog, Lakeflow Declarative Pipelines, DATEX II/NVDB live credentials,
AI/BI dashboards, spread across milestones M0 to M7 plus stretch S1 to S3. That plan needs workspace
accounts, DATEX credentials and multi-day setup before anything is demoable, and treats ML (`12_S3_ml.md`)
as the last stretch item, not the core.

The new direction: FrostSight's product is proven fastest as a data-science-first prototype — a trained
model producing an explainable risk score, shown on a dashboard — decoupled from any specific data
platform. Owning this project end to end, product decisions below are mine to make; ADR-0001's domain
framing (icing risk per NVDB road segment, Troms pilot county) and risk-level vocabulary
(`LOW`/`MEDIUM`/`HIGH`/`VERY_HIGH`) still hold, only the delivery path changes.

## Decision

Replace the Databricks milestone plan (`docs/plan/00_README.md` through `13_stretch_review.md`) as the
primary roadmap. The new primary track is `prototype/`: synthetic road-weather data standing in for the
live DATEX II/NVDB feeds, a small PyTorch feed-forward model trained to score icing risk per segment, and
a Plotly Dash dashboard reproducing the four planned views (risk map, road detail, gritting priority list,
platform health) in one local app. See `prototype/README.md` for how to run it.

The old plan files and ADRs 0001 to 0004 are kept, not deleted — they hold real domain research (source
verification, pilot county choice, risk factors) that the prototype and any later real-data integration
still draw on. They are historical/reference material, not the active roadmap.

## Consequences

- Today's deliverable is a same-day, no-credentials, no-cloud MVP: `python -m prototype.train` then
  `python -m prototype.app`, verified running locally.
- Risk scores are trained on synthetic data seeded from a documented heuristic (`prototype/generate_data.py`),
  not on live observations or real accident history — this is explicitly a modeling/UX prototype, not a
  validated forecast, and the dashboard says so.
- Swapping in real data later means replacing `prototype/generate_data.py`'s output with real DATEX/NVDB
  extracts (`collector/`, still present in the repo) and retraining — the model and app code are
  data-source agnostic already.
- The Databricks bundle, pipeline and dashboard code described in `docs/plan/03_M2_foundation.md` onward
  is deprioritized; nothing in `resources/`, `sql/`, or the bundle targets is required for this track.
