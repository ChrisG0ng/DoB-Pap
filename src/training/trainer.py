"""Training loop shared by UNet2 and UniCAS models."""

import os
import time
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.concentration_scaling import ConcentrationScaler
from ..utils.loss import UnmixingLoss
from ..utils.metrics import MetricsTracker


class UnmixingTrainer:
    """
    Optimizes the reconstruction (+ optional concentration) loss.

    The stain matrix is either fixed (domain adaptation) or taken per patch from
    the batch ('stain_matrix'). Only parameters with requires_grad are optimized.
    The checkpoint with the lowest validation total loss is saved as
    '{model_name}_best.pth' together with the concentration scaler.

    Example:
        >>> trainer = UnmixingTrainer(model, train_loader, val_loader, save_dir='./out')
        >>> history = trainer.train(num_epochs=10, early_stopping_patience=5, save_every=10)
    """

    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
                 reconstruction_weight: float = 1.0, concentration_weight: float = 0.0,
                 learning_rate: float = 1e-4, weight_decay: float = 1e-4, optimizer_type: str = 'adamw',
                 scheduler_patience: int = 5, scheduler_factor: float = 0.5, min_lr: float = 1e-8,
                 device: str = 'cuda', save_dir: str = './models', model_name: str = 'model',
                 scaler: Optional[ConcentrationScaler] = None,
                 fixed_stain_matrix: Optional[torch.Tensor] = None,
                 gpu_ids: Optional[Sequence[int]] = None):
        if gpu_ids and len(gpu_ids) > 1:
            device = f'cuda:{gpu_ids[0]}'
            model = nn.DataParallel(model.to(device), device_ids=list(gpu_ids))
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader or train_loader
        self.save_dir = save_dir
        self.model_name = model_name
        self.scaler = scaler
        self.fixed_stain_matrix = None if fixed_stain_matrix is None else fixed_stain_matrix.to(device)
        os.makedirs(save_dir, exist_ok=True)

        self.criterion = UnmixingLoss(reconstruction_weight, concentration_weight, scaler)
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizers = {'adam': optim.Adam, 'adamw': optim.AdamW}
        if optimizer_type.lower() not in optimizers:
            raise ValueError(f"optimizer must be 'adam' or 'adamw', got {optimizer_type!r}")
        self.optimizer = optimizers[optimizer_type.lower()](params, lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=scheduler_factor,
                                           patience=scheduler_patience, min_lr=min_lr)

        keys = ['total_loss', 'reconstruction_loss', 'concentration_loss', 'rmse', 'sre']
        self.history: Dict[str, List[float]] = {f'{s}_{k}': [] for s in ('train', 'val') for k in keys}
        self.history['learning_rates'] = []
        self.best_val_loss = float('inf')
        self.best_epoch = -1

    def _stain_matrix(self, batch: Dict, batch_size: int) -> torch.Tensor:
        if self.fixed_stain_matrix is not None:
            return self.fixed_stain_matrix.unsqueeze(0).expand(batch_size, -1, -1)
        if 'stain_matrix' not in batch:
            raise ValueError("No fixed stain matrix and no per-patch 'stain_matrix' in batch.")
        return batch['stain_matrix'].to(self.device)

    def _unscale(self, conc: torch.Tensor) -> torch.Tensor:
        if self.scaler is None:
            return conc
        factors = torch.tensor(self.scaler.scaling_factors, dtype=conc.dtype, device=conc.device)
        return conc * factors.view(1, -1, 1, 1)

    def _run_epoch(self, loader: DataLoader, train: bool) -> Dict[str, float]:
        self.model.train(train)
        totals = {'total_loss': 0.0, 'reconstruction_loss': 0.0, 'concentration_loss': 0.0}
        tracker = MetricsTracker()
        with torch.set_grad_enabled(train):
            for batch in tqdm(loader, desc='Training' if train else 'Validating'):
                od = batch['optical_density'].to(self.device)
                target = batch['concentration'].to(self.device) if 'concentration' in batch else None
                stain_matrix = self._stain_matrix(batch, od.size(0))
                conc = self.model(od)

                if train:
                    loss = self.criterion(conc, stain_matrix, od, target)['total_loss']
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                display = self.criterion.display_losses(conc.detach(), stain_matrix, od, target)
                for k in totals:
                    totals[k] += display[k].item()
                if target is not None:
                    tracker.update(self._unscale(conc.detach()), self._unscale(target))
        results = {k: v / len(loader) for k, v in totals.items()}
        results.update(tracker.average())
        return results

    def save_checkpoint(self, epoch: int, filename: str) -> str:
        """Save model, optimizer state, history and scaler to save_dir/filename."""
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        path = os.path.join(self.save_dir, filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'history': self.history,
            'scaler': None if self.scaler is None else self.scaler.to_dict(),
        }, path)
        return path

    def train(self, num_epochs: int, early_stopping_patience: int, save_every: int) -> Dict[str, List[float]]:
        """Train with LR scheduling on validation loss and early stopping; returns the history."""
        patience = 0
        for epoch in range(num_epochs):
            start = time.time()
            train_res = self._run_epoch(self.train_loader, train=True)
            val_res = self._run_epoch(self.val_loader, train=False)
            self.scheduler.step(val_res['total_loss'])
            lr = self.optimizer.param_groups[0]['lr']

            for k, v in train_res.items():
                self.history[f'train_{k}'].append(v)
            for k, v in val_res.items():
                self.history[f'val_{k}'].append(v)
            self.history['learning_rates'].append(lr)

            improved = val_res['total_loss'] < self.best_val_loss
            print(f"Epoch {epoch + 1}/{num_epochs} ({time.time() - start:.1f}s, lr={lr:.2e}) "
                  f"train loss={train_res['total_loss']:.6f} val loss={val_res['total_loss']:.6f} "
                  f"val rec={val_res['reconstruction_loss']:.6f}"
                  + (f" val RMSE={val_res['rmse']:.6f} val SRE={val_res['sre']:.2f}dB" if val_res['rmse'] else '')
                  + (' *' if improved else ''))

            if improved:
                self.best_val_loss, self.best_epoch, patience = val_res['total_loss'], epoch, 0
                self.save_checkpoint(epoch, f'{self.model_name}_best.pth')
            else:
                patience += 1
                if (epoch + 1) % save_every == 0:
                    self.save_checkpoint(epoch, f'{self.model_name}_epoch_{epoch + 1:03d}.pth')
            if patience >= early_stopping_patience:
                print(f"Early stopping: no improvement for {early_stopping_patience} epochs.")
                break
        print(f"Best epoch: {self.best_epoch + 1}, val total loss: {self.best_val_loss:.6f}")
        return self.history
