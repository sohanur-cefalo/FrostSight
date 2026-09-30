"""Score current risk per road segment with the trained PyTorch model."""

from __future__ import annotations

import pandas as pd
import torch

from prototype.features import add_surface_temp_trend_1h
from prototype.model import FEATURE_COLUMNS, IcingRiskNet, risk_level

ARTIFACTS = "prototype/artifacts"


def _latest_observations(obs: pd.DataFrame) -> pd.DataFrame:
    """The "current condition" row per segment. For past-observation data
    (generate_data.py's synthetic history) that's the max event_time; for a
    forward-looking forecast series (build_live_weather.py, Phase 2) the max
    event_time is the *furthest future* point, not "now" — so pick whichever
    row's event_time is closest to the current wall-clock time instead, which
    is correct for both."""
    obs = add_surface_temp_trend_1h(obs)
    now = pd.Timestamp.now(tz="UTC")
    event_time = obs["event_time"]
    if event_time.dt.tz is None:
        event_time = event_time.dt.tz_localize("UTC")
    obs = obs.assign(_distance_from_now=(event_time - now).abs())
    closest_idx = obs.groupby("road_segment_id")["_distance_from_now"].idxmin()
    return obs.loc[closest_idx].drop(columns="_distance_from_now").copy()


def score_segments() -> pd.DataFrame:
    segments = pd.read_csv(f"{ARTIFACTS}/segments.csv")
    obs = pd.read_csv(f"{ARTIFACTS}/observations.csv", parse_dates=["event_time"])
    obs = obs.merge(segments[["road_segment_id", "exposure_factor"]], on="road_segment_id", how="left")

    latest = _latest_observations(obs)

    checkpoint = torch.load(f"{ARTIFACTS}/icing_model.pt", weights_only=False)
    model = IcingRiskNet()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    x = torch.tensor(latest[FEATURE_COLUMNS].values, dtype=torch.float32)
    x_norm = (x - checkpoint["feature_mean"]) / checkpoint["feature_std"]
    with torch.no_grad():
        scores = model(x_norm).numpy()

    out = segments.merge(
        latest[
            [
                "road_segment_id",
                "event_time",
                "air_temp_c",
                "surface_temp_c",
                "wind_speed_ms",
                "precip_mm",
                "surface_temp_trend_1h",
                "heuristic_risk",
            ]
        ],
        on="road_segment_id",
        how="left",
    )
    out["ml_risk_score"] = scores
    out["risk_level"] = out["ml_risk_score"].apply(risk_level)

    def drivers(row) -> str:
        parts = []
        if row["surface_temp_c"] < 0.5:
            parts.append(f"surface {row['surface_temp_c']:.1f}°C")
        if row["precip_mm"] > 0.3:
            parts.append(f"precip {row['precip_mm']:.1f}mm")
        if row["wind_speed_ms"] > 6:
            parts.append(f"wind {row['wind_speed_ms']:.1f}m/s")
        if row["surface_temp_trend_1h"] < -1:
            parts.append("dropping fast")
        return ", ".join(parts) if parts else "stable"

    out["drivers"] = out.apply(drivers, axis=1)
    return out.sort_values("ml_risk_score", ascending=False)


if __name__ == "__main__":
    df = score_segments()
    print(df[["road_segment_id", "risk_level", "ml_risk_score", "drivers"]].head(10).to_string(index=False))
