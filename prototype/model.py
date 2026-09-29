"""PyTorch icing-risk model: small feed-forward regressor over instantaneous
weather features plus short-term trend, trained to reproduce (and generalize
past) the heuristic label from generate_data.py.
"""

from __future__ import annotations

import torch
from torch import nn

FEATURE_COLUMNS = [
    "air_temp_c",
    "surface_temp_c",
    "wind_speed_ms",
    "precip_mm",
    "exposure_factor",
    "surface_temp_trend_1h",
]

RISK_LEVELS = ["LOW", "MEDIUM", "HIGH", "VERY_HIGH"]
LEVEL_THRESHOLDS = [0.25, 0.5, 0.75]  # upper bound of LOW, MEDIUM, HIGH


class IcingRiskNet(nn.Module):
    def __init__(self, n_features: int = len(FEATURE_COLUMNS), hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def risk_level(score: float) -> str:
    for level, upper in zip(RISK_LEVELS, LEVEL_THRESHOLDS):
        if score <= upper:
            return level
    return RISK_LEVELS[-1]
