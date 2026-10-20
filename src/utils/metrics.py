"""Patch-averaged RMSE and SRE metrics."""

from typing import Dict

import numpy as np
import torch


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """RMSE per patch (B, C, H, W), averaged over patches."""
    return torch.sqrt(torch.mean((pred - target) ** 2, dim=[1, 2, 3])).mean().item()


def sre(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Signal-to-reconstruction error in dB per patch, averaged over patches."""
    signal = torch.mean(target ** 2, dim=[1, 2, 3])
    error = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
    return (10 * torch.log10((signal + eps) / (error + eps))).mean().item()


class MetricsTracker:
    """Averages batch-level RMSE and SRE."""

    def __init__(self):
        self.values = {'rmse': [], 'sre': []}

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        self.values['rmse'].append(rmse(pred, target))
        self.values['sre'].append(sre(pred, target))

    def average(self) -> Dict[str, float]:
        """Mean of each metric (0.0 when empty)."""
        return {k: float(np.mean(v)) if v else 0.0 for k, v in self.values.items()}
