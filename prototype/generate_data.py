"""Synthetic road-segment weather generator for the FrostSight local prototype.

Stands in for the live DATEX II / NVDB feeds so the prototype runs with no
credentials and no network access. Produces:
  - segments.csv: road segments in the Troms pilot county (ADR-0002), with a
    rough lat/lon line and a terrain exposure factor
  - observations.csv: a 10-minute time series per segment for the last N hours,
    with a heuristic icing_risk label used to train the PyTorch model
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)

N_SEGMENTS = 30
HOURS_OF_HISTORY = 8
STEP_MINUTES = 10

# Rough line from Tromsø towards Bardufoss/Setermoen (E8/E6 corridor), Troms county.
START = (69.6496, 18.9560)
END = (69.0500, 18.5000)


def make_segments() -> pd.DataFrame:
    t = np.linspace(0, 1, N_SEGMENTS)
    lat = START[0] + (END[0] - START[0]) * t + RNG.normal(0, 0.01, N_SEGMENTS)
    lon = START[1] + (END[1] - START[1]) * t + RNG.normal(0, 0.02, N_SEGMENTS)
    elevation = 20 + 400 * np.abs(np.sin(t * np.pi)) + RNG.normal(0, 15, N_SEGMENTS)
    exposure = np.clip(RNG.normal(0.5, 0.2, N_SEGMENTS), 0.05, 1.0)  # wind/shade exposure factor
    return pd.DataFrame(
        {
            "road_segment_id": [f"55-{i:03d}" for i in range(N_SEGMENTS)],
            "road_name": [f"E8/{i:03d}" for i in range(N_SEGMENTS)],
            "lat": lat,
            "lon": lon,
            "elevation_m": elevation.round(1),
            "exposure_factor": exposure.round(2),
        }
    )


def _synth_weather_path(n_steps: int, elevation: float, exposure: float, storm_phase: float):
    """One segment's synthetic 10-min weather path, biased toward a storm scenario."""
    t = np.arange(n_steps)
    base_temp = 2.0 - 0.006 * elevation
    storm = np.sin((t / n_steps + storm_phase) * np.pi) ** 3
    air_temp = base_temp - 6 * storm + RNG.normal(0, 0.4, n_steps)
    surface_temp = air_temp - 1.0 - 1.5 * exposure + RNG.normal(0, 0.3, n_steps)
    wind_speed = np.clip(2 + 6 * storm * exposure + RNG.normal(0, 1, n_steps), 0, None)
    precip_mm = np.clip(storm * 1.5 + RNG.normal(0, 0.2, n_steps), 0, None)
    precip_mm[air_temp > 4] *= 0.2  # rain, not relevant to icing, but still logged
    return air_temp, surface_temp, wind_speed, precip_mm


def heuristic_icing_risk(surface_temp, air_temp, wind_speed, precip_mm, exposure) -> np.ndarray:
    """Deterministic v0 heuristic (documents the label the ML model learns to reproduce).

    Not a validated expert model — a documented placeholder per ADR-0001, used here only
    to generate training labels for the synthetic prototype.
    """
    near_freezing = np.clip(1.0 - np.abs(surface_temp - (-1.0)) / 4.0, 0, 1)
    wet = np.clip(precip_mm / 1.5, 0, 1)
    wind_chill = np.clip(wind_speed / 12.0, 0, 1)
    below_zero = (surface_temp < 0.5).astype(float)
    score = below_zero * (0.5 * near_freezing + 0.3 * wet + 0.15 * wind_chill + 0.1 * exposure)
    return np.clip(score, 0, 1)


def make_observations(segments: pd.DataFrame) -> pd.DataFrame:
    n_steps = int(HOURS_OF_HISTORY * 60 / STEP_MINUTES)
    now = pd.Timestamp.utcnow().floor(f"{STEP_MINUTES}min")
    times = [now - pd.Timedelta(minutes=STEP_MINUTES * k) for k in range(n_steps)][::-1]

    rows = []
    for _, seg in segments.iterrows():
        storm_phase = RNG.uniform(0, 1)
        air_temp, surface_temp, wind_speed, precip_mm = _synth_weather_path(
            n_steps, seg["elevation_m"], seg["exposure_factor"], storm_phase
        )
        risk = heuristic_icing_risk(surface_temp, air_temp, wind_speed, precip_mm, seg["exposure_factor"])
        for k, ts in enumerate(times):
            rows.append(
                {
                    "road_segment_id": seg["road_segment_id"],
                    "event_time": ts,
                    "air_temp_c": round(float(air_temp[k]), 2),
                    "surface_temp_c": round(float(surface_temp[k]), 2),
                    "wind_speed_ms": round(float(wind_speed[k]), 2),
                    "precip_mm": round(float(precip_mm[k]), 2),
                    "heuristic_risk": round(float(risk[k]), 4),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    segments = make_segments()
    observations = make_observations(segments)
    segments.to_csv("prototype/artifacts/segments.csv", index=False)
    observations.to_csv("prototype/artifacts/observations.csv", index=False)
    print(f"Wrote {len(segments)} segments and {len(observations)} observations to prototype/artifacts/")


if __name__ == "__main__":
    main()
