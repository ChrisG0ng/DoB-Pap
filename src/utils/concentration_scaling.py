"""Per-channel concentration scaling used for training and fine-tuning."""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

STAIN_NAMES = ('EY', 'H', 'LG', 'OG')
DEFAULT_FIXED_C5 = 0.01


class ConcentrationScaler:
    """Divides concentrations by per-channel scaling factors (and multiplies them back)."""

    def __init__(self, scaling_factors: Sequence[float], means: Optional[Sequence[float]] = None,
                 stds: Optional[Sequence[float]] = None, channel_names: Sequence[str] = STAIN_NAMES):
        self.scaling_factors = np.asarray(scaling_factors)
        n = len(self.scaling_factors)
        self.means = np.zeros(n) if means is None else np.asarray(means)
        self.stds = np.zeros(n) if stds is None else np.asarray(stds)
        self.channel_names = list(channel_names)

    def scale_concentrations(self, conc: np.ndarray) -> np.ndarray:
        """Scale (..., C) concentrations; near-zero factors are treated as 1."""
        safe = np.where(self.scaling_factors > 1e-8, self.scaling_factors, 1.0)
        return conc / safe

    def inverse_scale_concentrations(self, conc: np.ndarray) -> np.ndarray:
        """Map scaled (..., C) concentrations back to the original scale."""
        return conc * self.scaling_factors

    def to_dict(self) -> Dict:
        """Serializable form stored in checkpoints."""
        return {
            'means': self.means,
            'stds': self.stds,
            'scaling_factors': self.scaling_factors,
            'channel_names': self.channel_names,
            'is_fitted': True,
        }

    @classmethod
    def from_dict(cls, info: Dict) -> 'ConcentrationScaler':
        """Rebuild a scaler from :meth:`to_dict` output."""
        return cls(info['scaling_factors'], info.get('means'), info.get('stds'),
                   info.get('channel_names', STAIN_NAMES))

    def __repr__(self) -> str:
        factors = ', '.join(f'{n}={f:.4f}' for n, f in zip(self.channel_names, self.scaling_factors))
        return f'ConcentrationScaler({factors})'


def scaler_from_checkpoint(checkpoint: Dict) -> Optional[ConcentrationScaler]:
    """Return the scaler stored in a checkpoint dict, or None."""
    info = checkpoint.get('scaler')
    return ConcentrationScaler.from_dict(info) if info else None


def normal_statistics(sum_c: np.ndarray, sum_sq_c: np.ndarray, count: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-channel (mean, std, mean + 2*std) from running sums."""
    if count <= 0:
        zeros = np.zeros_like(sum_c, dtype=np.float64)
        return zeros, zeros, np.ones_like(sum_c, dtype=np.float64)
    means = sum_c / count
    stds = np.sqrt(np.maximum(sum_sq_c / count - means ** 2, 0.0))
    return means, stds, means + 2.0 * stds


def c99_per_channel(values: np.ndarray, fixed_c5: float = DEFAULT_FIXED_C5) -> np.ndarray:
    """99th percentile per channel of (N, C) values after dropping values <= fixed_c5."""
    c99 = np.zeros(values.shape[1], dtype=np.float64)
    for i in range(values.shape[1]):
        data = np.sort(values[:, i])
        kept = data[data > fixed_c5]
        c99[i] = kept[int(len(kept) * 0.99)] if len(kept) else data[-1]
    return c99


def fit_scaler_from_targets(loader: DataLoader) -> ConcentrationScaler:
    """Fit a mean + 2*std scaler on ground-truth concentrations of a loader."""
    sum_c = sum_sq = None
    count = 0
    for batch in tqdm(loader, desc='Concentration statistics'):
        flat = batch['concentration'].permute(0, 2, 3, 1).reshape(-1, batch['concentration'].shape[1])
        flat = flat.double().numpy()
        if sum_c is None:
            sum_c, sum_sq = np.zeros(flat.shape[1]), np.zeros(flat.shape[1])
        sum_c += flat.sum(axis=0)
        sum_sq += (flat ** 2).sum(axis=0)
        count += flat.shape[0]
    means, stds, factors = normal_statistics(sum_c, sum_sq, count)
    return ConcentrationScaler(factors, means, stds)


@torch.no_grad()
def fit_scaler_from_predictions(model: torch.nn.Module, loader: DataLoader, device: str,
                                base_scaler: Optional[ConcentrationScaler] = None,
                                mode: str = 'c99',
                                fixed_c5: float = DEFAULT_FIXED_C5) -> ConcentrationScaler:
    """
    Fit a scaler on model predictions (original scale) over a loader of OD patches.

    Args:
        mode: 'c99' (99th percentile above fixed_c5) or 'normal' (mean + 2*std).
        base_scaler: Scaler of the model; predictions are inverse-scaled with it.
    """
    if mode not in ('c99', 'normal'):
        raise ValueError(f"mode must be 'c99' or 'normal', got {mode!r}")
    model = model.to(device).eval()
    predictions = []
    for batch in tqdm(loader, desc=f'Scaler ({mode}) from predictions'):
        conc = model(batch['optical_density'].to(device)).cpu().numpy().transpose(0, 2, 3, 1)
        if base_scaler is not None:
            conc = base_scaler.inverse_scale_concentrations(conc)
        predictions.append(conc.reshape(-1, conc.shape[-1]))
    values = np.concatenate(predictions, axis=0)

    if mode == 'c99':
        return ConcentrationScaler(c99_per_channel(values, fixed_c5).astype(np.float32))
    values = values.astype(np.float64)
    means, stds, factors = normal_statistics(values.sum(axis=0), (values ** 2).sum(axis=0), len(values))
    return ConcentrationScaler(factors, means, stds)
