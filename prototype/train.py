"""Generate synthetic data (if missing) and train the PyTorch icing-risk model.

Usage: python -m prototype.train
"""

from __future__ import annotations

import pandas as pd
import torch
from torch import nn

from prototype.generate_data import main as generate_data
from prototype.model import FEATURE_COLUMNS, IcingRiskNet

ARTIFACTS = "prototype/artifacts"


def _add_features(obs: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    obs = obs.sort_values(["road_segment_id", "event_time"]).copy()
    obs["surface_temp_trend_1h"] = obs.groupby("road_segment_id")["surface_temp_c"].diff(6).fillna(0.0)
    obs = obs.merge(segments[["road_segment_id", "exposure_factor"]], on="road_segment_id", how="left")
    return obs


def train() -> None:
    import os

    if not os.path.exists(f"{ARTIFACTS}/observations.csv"):
        generate_data()

    segments = pd.read_csv(f"{ARTIFACTS}/segments.csv")
    obs = pd.read_csv(f"{ARTIFACTS}/observations.csv", parse_dates=["event_time"])
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
