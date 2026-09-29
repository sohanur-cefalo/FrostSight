> **Superseded 2026-09-29 (ADR-0005):** this Databricks milestone plan is no longer the active roadmap.
> The active track is `prototype/` — a local PyTorch model and Plotly Dash dashboard on synthetic data.
> Kept here as reference for the domain research and for a later real-data integration.

# FrostSight: technical implementation plan

Audience: a senior engineer with Python, SQL and cloud experience but no Databricks experience. Every
milestone file below is written so you can follow it step by step: what to do, why in one line, the exact
command or code, what you should see, and the usual failure. Read this file first; it holds the conventions
every other file relies on.

Status: MVP scope (M0 to M7) first, then the stretch scope (S1 to S3). Milestones have an order, not dates.

## 1. Files in this folder

| File | Covers | Owner in the team |
|---|---|---|
| `00_README.md` | Conventions, repo layout, dual-target strategy, glossary | Safiul |
| `01_M0_accounts_and_sources.md` | Accounts, CLI, profiles, source access, verification scripts, pilot county, ADRs | Shawon, Rayhan, Safiul |
| `02_M1_history_and_reference_data.md` | Collector package, NVDB extract, last winter's history, landing layout, data contracts | Shawon, Rayhan |
| `03_M2_foundation.md` | Unity Catalog layout, bundle skeleton with two targets, CI, collector deployment, replay harness prototype, M2 gate | Rayhan, Shawon |
| `04_M3_design.md` | Silver schemas, watermark and dedup design, station-to-segment mapping benchmark, risk model v0, dashboard wireframe, design review | Sani, Rayhan, Safiul |
| `05_M4_bronze_and_silver.md` | Lakeflow Declarative Pipeline: Auto Loader bronze, silver with expectations and quarantine, station lookup, batch reference jobs | Shawon, Sani, Rayhan |
| `06_M5_risk_engine.md` | Windows and temperature trends, risk engine package and tests, gold tables, MERGE | Sani, Safiul |
| `07_M6_product.md` | AI/BI dashboards (risk map, road detail, priority list, health), freshness monitoring, bundle deploy from CI | Safiul, Rayhan |
| `08_M7_hardening_and_demo.md` | Outage handling, backfill, benchmarks, runbooks, demo script, definition of done | All |
| `09_MVP_review.md` | Findings from the review of files 01 to 08 and the fixes applied | Safiul |
| `10_S1_replay.md` | Storm replay through the live pipeline | Sani, Shawon |
| `11_S2_baselines_alerts_api.md` | Historical baselines and anomalies, alert delivery, API app | Safiul, Rayhan |
| `12_S3_ml.md` | Labels, features, baseline model, batch prediction | Sohanur |
| `13_stretch_review.md` | Findings from the review of files 10 to 12 | Safiul |

## 1a. What the scaffold already contains

The initial commit provides the layout below with these files filled in: `README.md`, `pyproject.toml`,
`databricks.yml` (variables and targets, no resources yet), `sql/001_catalog_schemas_volume.sql`,
`config/sources.yml`, `.github/CODEOWNERS`, ADRs 0001 to 0004 in `docs/adr/`, and empty packages
`src/frostsight/` and `collector/` with one import test. Tasks below that "create" these files now
review and extend them instead. Everything else (resources, pipeline and job code, workflows, collector
modules, dashboards) is created by its owner in the milestone that names it.

## 2. The two targets: Free Edition and AWS

The code is identical on both. Only configuration differs. This is the rule that makes the dual target cheap.

| Aspect | Free Edition (team workspace and personal workspaces) | AWS (14-day trial from M4, or pay-as-you-go) |
|---|---|---|
| Compute | Serverless only | Serverless only, by choice, so nothing changes |
| Bundle target | `free` (team) and `personal` | `aws` |
| CLI profile | `frostsight-free`, `frostsight-personal` | `frostsight-aws` |
| Outbound internet from jobs | Restricted | Open |
| Collector | Runs outside: GitHub Actions cron, or a static-IP host for DATEX | Same collector code runs as a job task inside the workspace, or keep the external one |
| Streaming | Pipeline triggered every 10 min by a job | Same, plus optional `continuous: true` |
| Pipelines | One active pipeline per account: everything lives in one pipeline | Same single pipeline; replay may run as a second pipeline |
| Jobs | 5 concurrent tasks per account: chain tasks serially | Same chain; parallelism allowed |
| SQL warehouse | The one 2X-Small | Serverless warehouse, 2X-Small is enough |
| CI deploy identity | A user's token (no service principals) | A service principal with OAuth |
| Dashboards | AI/BI | AI/BI |

Rules that follow from the table:

1. Never configure a cluster. Every job task and pipeline is serverless.
2. Everything that reads an external URL lives in `collector/`, which runs outside Databricks on Free Edition
   and inside a job on AWS. Pipelines and jobs read only from the landing volume and Unity Catalog tables.
3. Target-specific values (workspace, schedule on or off, continuous on or off, collector location) are bundle
   variables. Nothing target-specific is hard-coded in Python or SQL.
4. Schema and table names are the same on every target. Personal workspaces have their own metastore, so the
   same names do not collide.
5. The catalog, schemas, landing volume and grants are created with SQL (`sql/001_catalog_schemas_volume.sql`),
   once per workspace, not by the bundle. Reason: `mode: development` renames bundle resources, the docs do not
   promise it leaves schemas alone, and Free Edition has one metastore anyway. The bundle owns only the
   pipeline, the jobs and the dashboards.

### Bundle variables and targets (the authoritative list)

| Variable | Default | free | personal | aws |
|---|---|---|---|---|
| `catalog` | `frostsight` | same | same | same |
| `landing_root` | `/Volumes/frostsight/landing/raw` | same | same | same |
| `pilot_county` | `"55"` | same | same | same |
| `warehouse_id` | `lookup: warehouse: ${var.warehouse_name}` | | | |
| `warehouse_name` | `"Serverless Starter Warehouse"` (verify the name in your workspace) | same | same | your warehouse |
| `schedule_pause_status` | `UNPAUSED` | `UNPAUSED` | `PAUSED` | `UNPAUSED` |
| `pipeline_continuous` | `false` | `false` | `false` | `false`, may be `true` for a continuous-mode test |
| `pipeline_development` | `true` | `true` | `true` | `false` |
| `notification_email` | the team admin | | | |
| `config_dir` | `${workspace.file_path}/config` | | | |
| `slack_destination_id` | `""` (S2 only, 11_S2 S2.2.3; used only if the workspace offers notification destinations) | the destination id | `""` | the destination id |

Targets: all three use `mode: development`. `free` and `aws` set `presets: { name_prefix: "", trigger_pause_status: UNPAUSED }`
so resource names are stable and schedules run; `personal` keeps the default `[dev <user>]` prefix and paused
schedules. The collector job exists only on `aws`, declared under `targets.aws.resources.jobs`. Jobs import
`frostsight` from a wheel built by the bundle `artifacts` block; the pipeline uses `root_path: ../src` (details in 05_M4).

## 3. Repository layout

Repo: `safiulanik-cefalo/FrostSight`, private. Everything lives at the repo root.

```

  databricks.yml                 bundle root: name, variables, targets free / personal / aws
  resources/
    ingest.pipeline.yml          the one Lakeflow Declarative Pipeline (bronze + silver + lookup)
    orchestrate.job.yml          the scheduled job: pipeline update -> gold -> freshness
    reference.job.yml            batch loads of NVDB, MET, elevation, accidents, avalanche
    dashboards.dashboard.yml     AI/BI dashboards
    (the aws-only collector job is declared inline under targets.aws.resources in databricks.yml)
  sql/
    001_catalog_schemas_volume.sql   creates catalog, schemas, volume and grants; run once per workspace
  src/
    frostsight/                  Python package: pure functions, unit-tested
      __init__.py
      config.py                  reads risk_weights.yml, thresholds, pilot county
      geo.py                     UTM33 -> WGS84, H3 helpers, nearest-link
      transforms.py              silver normalisation, units, dedup helpers
      windows.py                 30m/1h/3h/6h aggregations and trends
      risk.py                    icing score, level, drivers
      freshness.py               per-source freshness rows
    pipelines/
      bronze.py                  Auto Loader streaming tables
      silver.py                  streaming tables with expectations, quarantine flows
      lookup.py                  station -> segment lookup (materialized)
    jobs/
      load_reference.py          NVDB / MET / elevation / accidents / avalanche to bronze and silver
      build_gold.py              windows, risk engine, gold MERGE
      build_freshness.py         gold.data_quality_summary and freshness table
    dashboards/
      risk_map.lvdash.json
      road_detail.lvdash.json
      priority_list.lvdash.json
      platform_health.lvdash.json
    sql/                         reusable SQL for dashboards and checks
  collector/
    __init__.py
    common.py                    HTTP session, retries, landing path helper, volume upload
    nvdb.py                      road network, stations, accidents, avalanche extracts
    datex.py                     road weather 10-min and incidents, basic auth
    frost.py                     historical observations, client id
    met.py                       Locationforecast
    trafikkdata.py               GraphQL volumes (post-MVP)
    elevation.py                 Open-Meteo fallback
    run.py                       CLI entry: python -m collector.run --source road_weather --target free
  replay/
    harness.py                   writes historical files into landing in event-time order at xN speed
  tests/
    unit/                        pytest on pure functions (local PySpark)
    data/                        schema, null, uniqueness, freshness checks run inside a job
    fixtures/                    small real samples captured at M0 and M1
  tools/
    verify_sources.py            M0 source check (not `scripts/`: the root .gitignore ignores any `scripts/` folder)
  config/
    risk_weights.yml             factor breakpoints, weights, level thresholds (authoritative copy in 06_M5)
    dq_rules.yml                 data-quality rules consumed by the pipeline (authoritative copy in 03_M2)
    replay_events.yml            storm windows chosen at M1
    sources.yml                  endpoints, headers, cadence, licence, attribution per source
  .github/workflows/
    ci.yml                       ruff + pytest + bundle validate on every PR
    deploy-free.yml              bundle deploy -t free on main
    deploy-aws.yml               bundle deploy -t aws, manual trigger
    collector.yml                cron: run collector for sources that do not need a fixed IP
  README.md
  runbooks/                      one per track, written at M7

docs/adr/                        ADRs; 0001 to 0004 are in the scaffold, 0005 to 0007 come from M3
```

## 4. Naming

| Thing | Convention | Example |
|---|---|---|
| Catalog | `frostsight` | |
| Schemas | `landing`, `bronze`, `silver`, `gold`, `quarantine`, `ml` | |
| Landing volume | `frostsight.landing.raw` | `/Volumes/frostsight/landing/raw/road_weather/2026/11/03/20261103T071000Z.json` |
| Landing path | `raw/<source>/<yyyy>/<mm>/<dd>/<file>`, one flat `<source>` folder per dataset so every Auto Loader or batch reader watches exactly one folder | `raw/nvdb_stations/2026/11/03/20261103T070000Z.jsonl` |
| Landing sources | `road_weather`, `road_weather_xml`, `road_incidents`, `road_incidents_xml`, `nvdb_road_network`, `nvdb_stations`, `nvdb_speed_limits`, `nvdb_accidents`, `nvdb_avalanche`, `nvdb_counties`, `frost_sources`, `frost_history`, `datex_sites`, `elevation`, `source_metadata` | The collector converts DATEX XML to JSON lines (`road_weather`, `road_incidents`) and keeps the raw XML beside it (`*_xml`) |
| Live file name | `<UTC timestamp>.jsonl`, for example `20261103T071000Z.jsonl`; `_batch_id` = the timestamp stem | |
| Replay file name | `replay__<event_id>__<UTC timestamp>__r<UTC run start>.jsonl` written into the same `road_weather` folder; `_batch_id` = `replay:<event_id>` (part 2 of the name). The run suffix makes every harness run a new set of paths, because Auto Loader never re-reads a path it has seen (10_S1 D3) | `replay__storm_2026_01_15__20260115T081000Z__r20261120T091500Z.jsonl` |
| Failed collector run | Marker file `raw/<source>/_failed/<UTC timestamp>.json` with the error; Auto Loader excludes `_failed/` | |
| Silver road-weather flows | `silver.road_weather_observations` is one streaming table fed by two append flows, `road_weather_live` (`_batch_id NOT LIKE 'replay:%'`) and `road_weather_replay` (`LIKE 'replay:%'`), each with its own watermark and dedup state (05_M4 T4.4, 10_S1 D2) | event-log `origin.flow_name` |
| Bronze tables | as in the spec section 7.1 | `bronze.road_weather_events`, `bronze.road_incident_events`, `bronze.road_network` |
| Silver tables | as in the spec section 7.2, plus the lookup | `silver.road_weather_observations`, `silver.road_segments`, `silver.station_segment_lookup` |
| Gold tables | as in the spec section 7.3 | `gold.road_segment_current_risk`, `gold.road_segment_risk_history`, `gold.data_quality_summary` |
| Quarantine tables | `quarantine.invalid_<source>`, `quarantine.unmapped_observations`, `quarantine.schema_errors` | |
| Metadata columns on every bronze row | `_ingested_at`, `_source`, `_source_file`, `_source_event_id`, `_batch_id`, `_schema_version` | |
| Bundle resources | `ingest` (pipeline), `orchestrate`, `reference`, `collector` (jobs), `risk_map` etc. (dashboards) | |
| Python | `snake_case`, type hints, one function per transform, no Spark session creation inside functions | |
| Timestamps | UTC everywhere; column `event_time` for the measurement time, `_ingested_at` for arrival | |
| Units | °C, m/s, mm, metres | |
| Segment key | `road_segment_id` = NVDB `veglenkesekvensid` plus segment number, string | `"1113653-1-9"` |
| Station key | `station_id` = NVDB `Målestasjonsnummer` as string, joined to DATEX or Frost ids through the lookup | `"1900177"` |
| Risk levels | `LOW`, `MEDIUM`, `HIGH`, `VERY_HIGH` | |
| Branches | `feature/<milestone>-<short-name>`; PR into `main`; squash merge | `feature/m4-bronze-road-weather` |

## 5. Sources, fixed by verification on 28 Sep 2026

Details and test commands are in `../source-verification.md`.

| Source | Endpoint | Auth | Cadence | In MVP |
|---|---|---|---|---|
| NVDB v4 | `https://nvdbapiles.atlas.vegvesen.no` | header `X-Client: frostsight` | Reference, weekly | Yes |
| NVDB object types used | 153 Værstasjon (road-weather stations), 482 traffic stations, 570 accidents, 445 avalanche and landslide, 105 speed limits, `vegnett/veglenkesekvenser/segmentert` road links, `omrader/fylker` | | | Yes, except 482 |
| DATEX II road weather | `https://datex-server-get-v3-1.atlas.vegvesen.no/datexapi/GetMeasuredWeatherData/pullsnapshotdata` | Basic auth, account from the form | 10 min | Yes |
| DATEX II incidents | `.../datexapi/GetSituation/pullsnapshotdata` | same account | 10 to 30 min | Yes |
| MET Frost | `https://frost.met.no` | client id | History, once | Yes, for replay data |
| MET Locationforecast | `https://api.met.no/weatherapi/locationforecast/2.0/compact` | User-Agent | Hourly | Post-MVP batch |
| Open-Meteo elevation | `https://api.open-meteo.com/v1/elevation` | none | Once | Fallback for Kartverket |
| trafikkdata | `https://trafikkdata-api.atlas.vegvesen.no/` GraphQL | none | Hourly | Cut from MVP |

Pilot county: Troms, NVDB `fylke=55`. NVDB geometries are EPSG:25833 (UTM zone 33N) with a Z coordinate;
silver stores both the original and WGS84 lat/lon.

## 6. How each milestone file is written

Every task in a milestone file has this shape:

```
### T4.3 Silver road-weather observations with expectations      owner: Sani
Why: one line.
Do:
  1. exact command or file to create, with the code
  2. ...
Expect: what you see when it worked (table exists, row count, dashboard tile)
If it fails: the two or three most likely causes
```

Each milestone ends with a **Done when** checklist and a **Verify** block of commands anyone can run. The
gates at M0 and M2 are hard stops.

## 7. Glossary for engineers new to Databricks

| Term | Meaning here |
|---|---|
| Workspace | One Databricks environment with its own URL. Free Edition gives one per account |
| Unity Catalog | The catalog of catalogs, schemas, tables, volumes and grants. Three-level names: `catalog.schema.table` |
| Volume | A folder in Unity Catalog for files. Our landing zone. Path style `/Volumes/<catalog>/<schema>/<volume>/...` |
| Delta table | The table format. Supports MERGE, time travel and streaming reads and writes |
| Serverless compute | Databricks runs the Spark cluster for you. No cluster config, per-second billing on AWS, quota on Free |
| Auto Loader | A streaming source that watches a folder and ingests new files exactly once |
| Structured Streaming | Spark's streaming engine. Reads a source incrementally, keeps a checkpoint, supports event-time windows and watermarks |
| Watermark | How late an event may arrive and still be counted. Ours is 30 minutes |
| Lakeflow Declarative Pipeline (SDP) | A managed pipeline where you declare tables in Python or SQL and Databricks runs, orders and checkpoints them. Formerly Delta Live Tables |
| Expectation | A row-level rule inside a pipeline. Rows that fail can be dropped, kept with a flag, or fail the pipeline. We route failures to quarantine tables |
| Lakeflow Job | The scheduler. A job has tasks with dependencies; a task can run a notebook, a Python file, a pipeline update or SQL |
| Asset Bundle | Infrastructure as code for Databricks. `databricks.yml` plus `resources/*.yml`, deployed with the CLI. Formerly Databricks Asset Bundles, now Declarative Automation Bundles, same thing |
| Target | A named deployment environment inside a bundle: `free`, `personal`, `aws` |
| SQL warehouse | Compute for SQL and dashboards. Ours is serverless 2X-Small |
| AI/BI dashboard | The built-in dashboard product, stored as JSON, deployable from a bundle |
| Genie | Natural-language questions over tables. Optional here |
| MLflow | Experiment tracking and model registry. Stretch scope |
| H3 | Hexagonal grid index on the globe. We use it to bucket road segments and station points for fast nearest-segment lookup |
| DBU | Databricks Unit, the billing unit for compute on paid workspaces |

## 8. Definition of done for the MVP

From the spec section 31 and the detailed document, narrowed to the pilot county:

- Road-weather and incident files land in the volume every 10 to 30 minutes from the collector.
- The pipeline turns them into bronze and silver tables with expectations and quarantine, keyed to NVDB
  segments through the station lookup.
- The job computes windows, temperature trends, the icing risk score with drivers, and writes gold tables.
- Four dashboards show the risk map, road detail, gritting priority list and platform health with freshness.
- Everything deploys from the repo with `databricks bundle deploy` on both `free` and `aws` targets, and CI
  runs lint, unit tests and bundle validation on every pull request.
- Runbooks exist per track and the demo runs end to end on replayed last-winter data.
