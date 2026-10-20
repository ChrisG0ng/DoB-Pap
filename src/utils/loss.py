"""Reconstruction + concentration loss for stain unmixing."""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .concentration_scaling import ConcentrationScaler


def reconstruct_od(concentration: torch.Tensor, stain_matrix: torch.Tensor) -> torch.Tensor:
    """OD = A @ C for concentration (B, K, H, W) and stain matrix (B, 3, K)."""
    b, k, h, w = concentration.shape
    return torch.bmm(stain_matrix, concentration.reshape(b, k, h * w)).view(b, stain_matrix.shape[1], h, w)


class UnmixingLoss(nn.Module):
    """
    MSE reconstruction loss plus optional MSE concentration loss.

    Predictions live in scaled space, so the stain matrix is multiplied by the
    scaling factors before reconstruction. :meth:`display_losses` reports the
    same terms in the original concentration scale.
    """

    def __init__(self, reconstruction_weight: float = 1.0, concentration_weight: float = 0.0,
                 scaler: Optional[ConcentrationScaler] = None):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.concentration_weight = concentration_weight
        self.factors = None if scaler is None else torch.tensor(scaler.scaling_factors, dtype=torch.float32)
        self.mse = nn.MSELoss()

    def _combine(self, rec: torch.Tensor, conc: torch.Tensor) -> Dict[str, torch.Tensor]:
        total = self.reconstruction_weight * rec + self.concentration_weight * conc
        return {'total_loss': total, 'reconstruction_loss': rec, 'concentration_loss': conc}

    def _concentration_term(self, pred: torch.Tensor, target: Optional[torch.Tensor]) -> torch.Tensor:
        if target is None or self.concentration_weight <= 0:
            return torch.zeros((), device=pred.device)
        return self.mse(pred, target)

    def forward(self, concentration: torch.Tensor, stain_matrix: torch.Tensor,
                target_od: torch.Tensor, target_conc: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Losses in scaled space (used for back-propagation)."""
        if self.factors is not None:
            stain_matrix = stain_matrix * self.factors.to(stain_matrix.device).view(1, 1, -1)
        rec = self.mse(reconstruct_od(concentration, stain_matrix), target_od)
        return self._combine(rec, self._concentration_term(concentration, target_conc))

    @torch.no_grad()
    def display_losses(self, concentration: torch.Tensor, stain_matrix: torch.Tensor,
                       target_od: torch.Tensor, target_conc: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Losses in the original concentration scale (monitoring only)."""
        if self.factors is None:
            return self.forward(concentration, stain_matrix, target_od, target_conc)
        factors = self.factors.to(concentration.device).view(1, -1, 1, 1)
        conc = concentration * factors
        target = None if target_conc is None else target_conc * factors
        rec = self.mse(reconstruct_od(conc, stain_matrix), target_od)
        return self._combine(rec, self._concentration_term(conc, target))
