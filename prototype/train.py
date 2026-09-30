"""Train the PyTorch icing-risk model on a broad synthetic weather distribution.

Training always uses a freshly generated synthetic segments/observations set
(generate_data.py), independent of whatever's currently in artifacts/segments.csv
/observations.csv for serving. This matters once real data is involved: a live
MET forecast snapshot (build_live_weather.py, Phase 2) covers one moment in
time across 30 segments — often near-zero variance in risk (e.g. a mild day) —
so training on it directly would collapse the model into "always low risk"
instead of learning the general relationship between weather and icing risk.
The trained model is then used to score whatever real or synthetic data is
staged for serving (risk.py), same as any train/serve split.

Usage: python -m prototype.train
"""

from __future__ import annotations

import pandas as pd
import torch
from torch import nn

from prototype.features import add_surface_temp_trend_1h
from prototype.generate_data import make_observations, make_segments
from prototype.model import FEATURE_COLUMNS, IcingRiskNet

ARTIFACTS = "prototype/artifacts"


def _add_features(obs: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    obs = add_surface_temp_trend_1h(obs)
    obs = obs.merge(segments[["road_segment_id", "exposure_factor"]], on="road_segment_id", how="left")
    return obs


def train() -> None:
    import os

    segments = make_segments()
    obs = make_observations(segments)

    # Fresh checkout convenience: give the dashboard something to serve without
    # a separate step. Never overwrites existing serving data — real data staged
    # by build_dataset.py/build_live_weather.py is left alone.
    if not os.path.exists(f"{ARTIFACTS}/segments.csv"):
        from prototype.provenance import write_meta

        segments.to_csv(f"{ARTIFACTS}/segments.csv", index=False)
        obs.to_csv(f"{ARTIFACTS}/observations.csv", index=False)
        write_meta(segments_source="synthetic", observations_source="synthetic")

    obs = _add_features(obs, segments)

    x = torch.tensor(obs[FEATURE_COLUMNS].values, dtype=torch.float32)
    y = torch.tensor(obs["heuristic_risk"].values, dtype=torch.float32)

    mean, std = x.mean(dim=0), x.std(dim=0).clamp_min(1e-6)
    x_norm = (x - mean) / std

    model = IcingRiskNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    n = x_norm.shape[0]
    perm = torch.randperm(n)
    split = int(n * 0.8)
    train_idx, val_idx = perm[:split], perm[split:]

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        pred = model(x_norm[train_idx])
        loss = loss_fn(pred, y[train_idx])
        loss.backward()
        optimizer.step()

        if epoch % 40 == 0 or epoch == 199:
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(x_norm[val_idx]), y[val_idx]).item()
            print(f"epoch {epoch:3d}  train_mse={loss.item():.4f}  val_mse={val_loss:.4f}")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_mean": mean,
            "feature_std": std,
            "feature_columns": FEATURE_COLUMNS,
        },
        f"{ARTIFACTS}/icing_model.pt",
    )
    print(f"Saved model to {ARTIFACTS}/icing_model.pt")


if __name__ == "__main__":
    train()
