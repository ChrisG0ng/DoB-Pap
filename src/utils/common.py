"""Configuration, device, checkpoint and report helpers shared by the scripts."""

import ast
import configparser
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from .concentration_scaling import STAIN_NAMES, c99_per_channel, normal_statistics
from .stain_estimation import read_stain_matrix


def load_config(path: str) -> configparser.ConfigParser:
    """Read an .ini configuration file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    cfg = configparser.ConfigParser()
    cfg.read(path, encoding='utf-8')
    return cfg


def resolve_device(device: str = 'cuda', gpu_id: str = '') -> str:
    """Return 'cuda:<id>', 'cuda' or 'cpu' depending on availability."""
    if device.startswith('cuda') and not torch.cuda.is_available():
        print("[WARN] CUDA not available, using CPU.")
        return 'cpu'
    gpu_id = str(gpu_id).strip()
    if device == 'cuda' and gpu_id:
        idx = int(gpu_id)
        if not 0 <= idx < torch.cuda.device_count():
            raise ValueError(f"gpu_id {idx} is out of range ({torch.cuda.device_count()} GPUs)")
        torch.cuda.set_device(idx)
        return f'cuda:{idx}'
    return device


def device_from_section(section) -> str:
    """Resolve the device from a config section with 'device' and optional 'gpu_id'."""
    return resolve_device(section.get('device', 'cuda'), section.get('gpu_id', ''))


def unicas_kwargs(section) -> Dict:
    """UniCASStainUnmixing architecture keywords from a [MODEL] section."""
    return {
        'block_indices': tuple(int(v) for v in section.get('block_indices', '6,12,18,24').split(',')),
        'proj_dim': section.getint('proj_dim', 256),
        'decoder_channels': section.getint('decoder_channels', 256),
        'decoder_norm': section.get('decoder_norm', 'group'),
        'use_detail_branch': section.getboolean('use_detail_branch', True),
        'dropout': section.getfloat('dropout', 0.1),
    }


def load_checkpoint(path: str, device: str = 'cpu') -> Dict:
    """Load a training checkpoint dictionary."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return torch.load(path, map_location=device, weights_only=False)


def load_weights(model: torch.nn.Module, checkpoint: Dict) -> torch.nn.Module:
    """Strictly load 'model_state_dict' into `model`; an unused 'od_stains' entry is skipped."""
    state = {k: v for k, v in checkpoint['model_state_dict'].items() if k != 'od_stains'}
    model.load_state_dict(state, strict=True)
    return model


class PixelStats:
    """Accumulates per-pixel concentrations to report mean + 2*std and c99 scalers."""

    def __init__(self, n_channels: int = 4):
        self.sum = np.zeros(n_channels)
        self.sum_sq = np.zeros(n_channels)
        self.chunks: List[np.ndarray] = []

    def update(self, values: np.ndarray) -> None:
        """Add (N, C) values."""
        if values.size == 0:
            return
        v = values.astype(np.float64)
        self.sum += v.sum(axis=0)
        self.sum_sq += (v * v).sum(axis=0)
        self.chunks.append(values.astype(np.float32))

    def write_report(self, path: str, title: str, fixed_c5: float, inverse_scaled: bool) -> None:
        """Write normal (mean + 2*std) and c99 scalers to a text file."""
        if not self.chunks:
            print("[WARN] No pixels accumulated; scaler report not written.")
            return
        values = np.concatenate(self.chunks, axis=0)
        means, stds, normal = normal_statistics(self.sum, self.sum_sq, len(values))
        c99 = c99_per_channel(values, fixed_c5)
        lines = [title, '=' * 60,
                 f'Predictions inverse-scaled with checkpoint scaler: {inverse_scaled}',
                 f'Total pixels: {len(values)}',
                 f'fixed_c5: {fixed_c5:.8f}', '',
                 '# normal mode: channel\tmean\tstd\tmean+2*std']
        lines += [f'{n}\t{means[i]:.8f}\t{stds[i]:.8f}\t{normal[i]:.8f}' for i, n in enumerate(STAIN_NAMES)]
        lines += ['', '# c99 mode: channel\tc99']
        lines += [f'{n}\t{c99[i]:.8f}' for i, n in enumerate(STAIN_NAMES)]
        _write_lines(path, lines)
        print(f"Scaler report: {path}")


def write_background_report(rows: Sequence[Tuple[str, np.ndarray, np.ndarray]], path: str,
                            epsilon: float) -> None:
    """Write per-image I0 and OD_background = -log10(I0/255) with dataset mean/std."""
    if not rows:
        _write_lines(path, [f'# No images processed. epsilon={epsilon}'])
        return
    od_bg = np.stack([r[2] for r in rows])
    mean = od_bg.mean(axis=0)
    std = od_bg.std(axis=0, ddof=1) if len(rows) > 1 else np.zeros(3)
    lines = ['# OD_background = -log10(I0/255); I0 = mean RGB of background pixels.',
             f'# epsilon = {epsilon}',
             f'# OD_background mean (R,G,B): {mean[0]:.6f} {mean[1]:.6f} {mean[2]:.6f}',
             f'# OD_background std  (R,G,B): {std[0]:.6f} {std[1]:.6f} {std[2]:.6f}',
             '# filename\tI0_R\tI0_G\tI0_B\tOD_bg_R\tOD_bg_G\tOD_bg_B']
    lines += ['\t'.join([name, *(f'{v:.6f}' for v in i0), *(f'{v:.6f}' for v in odb)]) for name, i0, odb in rows]
    _write_lines(path, lines)


def reconstruction_stain_matrix(section) -> Optional[np.ndarray]:
    """
    (3, 4) stain matrix for RGB reconstruction, or None when Do_reconstruction is false.

    Read from `fixed_stain_matrix_dir` (an overall_stain_matrix.txt) or an inline
    `fixed_stain_matrix` list.
    """
    if not section.getboolean('Do_reconstruction', True):
        return None
    path = section.get('fixed_stain_matrix_dir', '').strip()
    if path:
        return read_stain_matrix(path)
    inline = section.get('fixed_stain_matrix', '').strip()
    if inline:
        matrix = np.array(ast.literal_eval(inline), dtype=np.float32)
        if matrix.shape != (3, 4):
            raise ValueError(f"fixed_stain_matrix must be 3x4, got {matrix.shape}")
        return matrix
    raise ValueError("Do_reconstruction=true requires fixed_stain_matrix_dir or fixed_stain_matrix.")


def save_unmixing_results(conc_map: np.ndarray, od_bg: np.ndarray, output_dir: str, rel_dir: str,
                          name: str, stain_matrix: Optional[np.ndarray]) -> None:
    """
    Save {name}_concentration.npy under concentration/ and, if a stain matrix is
    given, per-stain and full reconstructed RGB images under reconstruction/.
    """
    conc_dir = os.path.join(output_dir, 'concentration', rel_dir)
    os.makedirs(conc_dir, exist_ok=True)
    np.save(os.path.join(conc_dir, f'{name}_concentration.npy'), conc_map.astype(np.float32))
    if stain_matrix is None:
        return

    recon_dir = os.path.join(output_dir, 'reconstruction', rel_dir)
    os.makedirs(recon_dir, exist_ok=True)

    def to_rgb(od: np.ndarray) -> Image.Image:
        return Image.fromarray(np.clip(255.0 * np.power(10.0, -od), 0.0, 255.0).astype(np.uint8))

    conc = conc_map.astype(np.float64)
    for i, stain in enumerate(STAIN_NAMES):
        od = conc[:, :, i:i + 1] * stain_matrix[:, i].reshape(1, 1, 3)
        to_rgb(od).save(os.path.join(recon_dir, f'{name}_reconstructed_RGB_{stain}.png'))
    h, w, k = conc.shape
    od_all = (conc.reshape(-1, k) @ stain_matrix.T).reshape(h, w, 3) + od_bg.reshape(1, 1, 3)
    to_rgb(od_all).save(os.path.join(recon_dir, f'{name}_reconstructed_RGB_all.png'))


def _write_lines(path: str, lines: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
