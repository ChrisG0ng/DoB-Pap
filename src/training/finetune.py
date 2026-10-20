"""Fine-tuning with a fixed stain matrix and training-history plotting."""

import os
from typing import Callable, Dict, Optional, Sequence

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..data.data_loader import create_data_loaders
from ..utils.common import load_weights
from ..utils.concentration_scaling import DEFAULT_FIXED_C5, ConcentrationScaler, fit_scaler_from_predictions
from .trainer import UnmixingTrainer


def plot_training_history(history: Dict, save_path: str,
                          keys: Sequence[str] = ('total_loss', 'reconstruction_loss')) -> None:
    """Plot train/val curves for `keys` plus the learning rate."""
    epochs = range(1, len(history['learning_rates']) + 1)
    fig, axes = plt.subplots(1, len(keys) + 1, figsize=(6 * (len(keys) + 1), 5))
    for ax, key in zip(axes, keys):
        ax.plot(epochs, history[f'train_{key}'], 'b-', label='Train', linewidth=2)
        ax.plot(epochs, history[f'val_{key}'], 'r--', label='Val', linewidth=2)
        ax.set_title(key.replace('_', ' ').title())
        ax.legend()
    axes[-1].plot(epochs, history['learning_rates'], 'g-', linewidth=2)
    axes[-1].set_title('Learning Rate')
    axes[-1].set_yscale('log')
    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def finetune_with_stain_matrix(build_model: Callable[[], torch.nn.Module], checkpoint: Dict,
                               od_dir: str, stain_matrix: np.ndarray, save_dir: str, model_name: str,
                               section, device: str, compute_scaler: bool = True,
                               fixed_c5: float = DEFAULT_FIXED_C5) -> str:
    """
    Fine-tune a source model on target-domain OD patches with a fixed stain matrix.

    Args:
        build_model: Returns a fresh (untrained) model of the source architecture.
        checkpoint: Source checkpoint dict ('model_state_dict', optional 'scaler').
        od_dir: Directory of '*_OD_*.npy' patches.
        stain_matrix: Estimated (4, 3) stain matrix (rows = stains).
        section: [FINETUNE] config section.
        compute_scaler: Fit a new scaler on source-model predictions
            (section 'compute_scaler_mode': c99 or normal); otherwise reuse the source scaler.

    Returns:
        Path to '{model_name}_best.pth'.
    """
    fixed_sm = torch.tensor(stain_matrix.T, dtype=torch.float32)
    batch_size = section.getint('batch_size', 32)
    base_scaler: Optional[ConcentrationScaler] = ConcentrationScaler.from_dict(checkpoint['scaler']) \
        if checkpoint.get('scaler') else None

    def loaders(scaler):
        return create_data_loaders(od_dir, batch_size=batch_size, validation_split=0.0, scaler=scaler,
                                   use_concentration=False, use_gt_stain_matrix=False)

    if compute_scaler:
        mode = section.get('compute_scaler_mode', 'c99').strip().lower()
        source = load_weights(build_model(), checkpoint)
        scaler = fit_scaler_from_predictions(source, loaders(None)[0], device, base_scaler, mode, fixed_c5)
        del source
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        if base_scaler is None:
            raise ValueError("compute_scaler=false but the source checkpoint has no scaler.")
        scaler = base_scaler
    print(f"Fine-tune scaler: {scaler}")

    train_loader, _, info = loaders(scaler)
    print(f"Fine-tune samples: {info['train_samples']}")
    model = load_weights(build_model(), checkpoint)

    trainer = UnmixingTrainer(
        model, train_loader, None,
        reconstruction_weight=section.getfloat('reconstruction_weight', 1.0),
        concentration_weight=0.0,
        learning_rate=section.getfloat('learning_rate', 1e-4),
        weight_decay=section.getfloat('weight_decay', 1e-4),
        optimizer_type=section.get('optimizer', 'adamw'),
        scheduler_patience=section.getint('scheduler_patience', 5),
        scheduler_factor=section.getfloat('scheduler_factor', 0.5),
        min_lr=section.getfloat('min_lr', 1e-8),
        device=device, save_dir=save_dir, model_name=model_name,
        scaler=scaler, fixed_stain_matrix=fixed_sm,
    )
    history = trainer.train(num_epochs=section.getint('num_epochs', 10),
                            early_stopping_patience=section.getint('early_stopping_patience', 5),
                            save_every=section.getint('save_every', 10))
    plot_training_history(history, os.path.join(save_dir, 'training_history.png'))
    return os.path.join(save_dir, f'{model_name}_best.pth')
