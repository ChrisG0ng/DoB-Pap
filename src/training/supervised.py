"""Supervised training on synthetic OD / concentration / stain-matrix triplets."""

import os
from typing import Callable

import torch

from ..data.data_loader import create_data_loaders
from ..utils.common import resolve_device
from ..utils.concentration_scaling import fit_scaler_from_targets
from .finetune import plot_training_history
from .trainer import UnmixingTrainer


def train_supervised(cfg, build_model: Callable[[], torch.nn.Module], model_name: str) -> str:
    """
    Train a model from a config with [DATA], [TRAINING], [DEVICE] and [PATHS] sections.

    Returns:
        Path to '{model_name}_best.pth'.
    """
    data, train, dev = cfg['DATA'], cfg['TRAINING'], cfg['DEVICE']
    save_dir = cfg['PATHS']['model_save_dir']
    gpu_ids = [int(g) for g in dev.get('gpu_ids', '').split(',') if g.strip()]
    device = resolve_device(dev.get('device', 'cuda'), str(gpu_ids[0]) if gpu_ids else '')

    def loaders(scaler):
        return create_data_loaders(data['train_data_dir'], batch_size=data.getint('batch_size', 32),
                                   num_workers=data.getint('num_workers', 0),
                                   validation_split=data.getfloat('validation_split', 0.2),
                                   random_seed=data.getint('random_seed', 42), scaler=scaler)

    scaler = None
    if data.getboolean('use_concentration_scaling', True):
        scaler = fit_scaler_from_targets(loaders(None)[0])
        print(f"Scaler: {scaler}")
    train_loader, val_loader, info = loaders(scaler)
    print(f"Samples: train={info['train_samples']}, val={info['val_samples']}")

    trainer = UnmixingTrainer(
        build_model(), train_loader, val_loader,
        reconstruction_weight=train.getfloat('reconstruction_weight', 1.0),
        concentration_weight=train.getfloat('concentration_weight', 1.0),
        learning_rate=train.getfloat('learning_rate', 1e-4),
        weight_decay=train.getfloat('weight_decay', 1e-4),
        optimizer_type=train.get('optimizer', 'adamw'),
        scheduler_patience=train.getint('scheduler_patience', 5),
        scheduler_factor=train.getfloat('scheduler_factor', 0.5),
        min_lr=train.getfloat('min_lr', 1e-8),
        device=device, save_dir=save_dir, model_name=model_name, scaler=scaler, gpu_ids=gpu_ids,
    )
    history = trainer.train(num_epochs=train.getint('num_epochs', 50),
                            early_stopping_patience=train.getint('early_stopping_patience', 10),
                            save_every=train.getint('save_every', 50))
    plot_training_history(history, os.path.join(save_dir, 'training_history.png'),
                          keys=('total_loss', 'reconstruction_loss', 'concentration_loss', 'rmse', 'sre'))
    return os.path.join(save_dir, f'{model_name}_best.pth')
