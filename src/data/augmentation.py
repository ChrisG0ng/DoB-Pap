"""Paired flip/rotation augmentation, concentration scaling and tensor conversion."""

import random
from typing import Optional, Tuple

import numpy as np
import torch


class PairTransform:
    """
    Apply the same random flips / 90-degree rotation to OD and concentration,
    scale the concentration and convert both to (C, H, W) tensors.

    Args:
        augment: Enable random augmentation (training split only).
        scaler: Optional ConcentrationScaler applied to the concentration.
        flip_prob: Probability of each horizontal and vertical flip.
        rotation_prob: Probability of a rotation by 0/90/180/270 degrees.
    """

    def __init__(self, augment: bool = False, scaler=None, flip_prob: float = 0.5, rotation_prob: float = 0.5):
        self.augment = augment
        self.scaler = scaler
        self.flip_prob = flip_prob
        self.rotation_prob = rotation_prob

    def _augment(self, arrays):
        if random.random() < self.flip_prob:
            arrays = [np.flip(a, axis=1) for a in arrays]
        if random.random() < self.flip_prob:
            arrays = [np.flip(a, axis=0) for a in arrays]
        if random.random() < self.rotation_prob:
            k = random.choice([0, 1, 2, 3])
            arrays = [np.rot90(a, k=k, axes=(0, 1)) for a in arrays]
        return [np.ascontiguousarray(a) for a in arrays]

    def __call__(self, od: np.ndarray, conc: Optional[np.ndarray] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        arrays = [od] if conc is None else [od, conc]
        if self.augment:
            arrays = self._augment(arrays)
        od_t = torch.from_numpy(np.ascontiguousarray(arrays[0])).float().permute(2, 0, 1)
        if conc is None:
            return od_t, None
        conc_np = arrays[1] if self.scaler is None else self.scaler.scale_concentrations(arrays[1])
        return od_t, torch.from_numpy(np.ascontiguousarray(conc_np)).float().permute(2, 0, 1)
