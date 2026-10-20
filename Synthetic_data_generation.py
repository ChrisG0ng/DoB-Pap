"""
Generate synthetic training data (reference only; source data not released).

Each sample takes a random ground-truth concentration patch (256x256x4, EY/H/LG/OG),
augments it (flip, rotation, elastic deformation, global and per-stain scaling),
mixes it with a randomly drawn and perturbed stain matrix (OD = C @ M) and saves
  {prefix}_OD_{i}.npy, {prefix}_con_{i}.npy, optional {prefix}_RGB_{i}.png
together with generation_log.csv (augmentation parameters and stain vectors).

Usage:
    python Synthetic_data_generation.py --config config/synthetic_data_generation_config.ini
"""

import argparse
import csv
import glob
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
from tqdm import tqdm

from src.utils.common import load_config


def parse_scale(value: str) -> Tuple[float, ...]:
    """'b' -> log-uniform b^(2U-1); 'lo,hi' -> uniform(lo, hi)."""
    return tuple(float(v) for v in value.split(','))


def sample_scale(param: Tuple[float, ...], rng: np.random.Generator) -> float:
    """Draw a scale factor from a log-uniform (1 value) or uniform (2 values) distribution."""
    if len(param) == 1:
        return float(np.power(param[0], 2 * rng.uniform(0, 1) - 1))
    return float(rng.uniform(param[0], param[1]))


def load_stain_vectors(sm_dir: str, stains: Sequence[str]) -> Dict[str, np.ndarray]:
    """Load (N, 3) stain vectors per stain from '*{stain}*.csv' files."""
    vectors = {}
    for stain in stains:
        files = sorted(glob.glob(os.path.join(sm_dir, f'*{stain}*.csv')))
        if not files:
            raise FileNotFoundError(f"No CSV file for stain {stain} in {sm_dir}")
        vectors[stain] = np.loadtxt(files[0], delimiter=',', dtype=np.float32).reshape(-1, 3)
    return vectors


def spectral_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle in radians between two vectors."""
    cos = np.dot(v1 / (np.linalg.norm(v1) + 1e-10), v2 / (np.linalg.norm(v2) + 1e-10))
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


def draw_stain_matrix(vectors: Dict[str, np.ndarray], stains: Sequence[str], noise_std: float,
                      min_sad: float, rng: np.random.Generator,
                      max_attempts: int = 1000) -> Tuple[np.ndarray, np.ndarray, List[float], int]:
    """
    Draw one vector per stain, add Gaussian noise, clip to >= 0 and L2-normalize; repeat
    until all pairwise spectral angles are >= min_sad.

    Returns:
        (original (K, 3), perturbed (K, 3), pairwise SADs + mean SAD, attempts).
    """
    for attempt in range(1, max_attempts + 1):
        original = np.zeros((len(stains), 3), dtype=np.float32)
        perturbed = np.zeros_like(original)
        for i, stain in enumerate(stains):
            original[i] = vectors[stain][rng.integers(0, len(vectors[stain]))]
            p = np.clip(original[i] + rng.normal(0, noise_std, size=3), 0, None)
            norm = np.linalg.norm(p)
            perturbed[i] = p / norm if norm > 0 else p
        sads = [spectral_angle(perturbed[i], perturbed[j])
                for i in range(len(stains)) for j in range(i + 1, len(stains))]
        if min(sads) >= min_sad:
            break
    else:
        print(f"[WARN] min_SAD={min_sad} not reached after {max_attempts} attempts.")
    return original, perturbed.astype(np.float32), sads + [float(np.mean(sads))], attempt


def elastic_deform(patch: np.ndarray, alpha: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Apply the same smooth random displacement field to all channels."""
    h, w, c = patch.shape
    dx = gaussian_filter(rng.random((h, w)) * 2 - 1, sigma) * alpha
    dy = gaussian_filter(rng.random((h, w)) * 2 - 1, sigma) * alpha
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    coords = [np.clip(y + dy, 0, h - 1), np.clip(x + dx, 0, w - 1)]
    return np.stack([map_coordinates(patch[:, :, i], coords, order=1, mode='reflect') for i in range(c)], axis=2)


def augment_patch(patch: np.ndarray, aug, scale_patch, scale_channels, rng: np.random.Generator):
    """Flip -> rotate -> elastic deformation -> global and per-stain scaling; returns (patch, params)."""
    params = {'flip': False, 'rotate': False, 'angle': 0, 'elastic': False, 'alpha': 0.0, 'sigma': 0.0}
    if rng.random() < aug.getfloat('flip_probability', 0.0):
        patch = np.flip(patch, axis=1 if rng.integers(0, 2) == 0 else 0).copy()
        params['flip'] = True
    if rng.random() < aug.getfloat('rotate_probability', 0.0):
        k = int(rng.integers(1, 4))
        patch = np.rot90(patch, k=k, axes=(0, 1)).copy()
        params.update(rotate=True, angle=k * 90)
    if rng.random() < aug.getfloat('elastic_probability', 0.0):
        alpha = rng.uniform(*parse_scale(aug.get('elastic_alpha', '10,50')))
        sigma = rng.uniform(*parse_scale(aug.get('elastic_sigma', '6,10')))
        patch = elastic_deform(patch, alpha, sigma, rng)
        params.update(elastic=True, alpha=float(alpha), sigma=float(sigma))

    params['scale_patch'] = sample_scale(scale_patch, rng)
    patch = patch * params['scale_patch']
    params['scale_channels'] = [sample_scale(p, rng) for p in scale_channels]
    patch = patch * np.array(params['scale_channels'], dtype=np.float32)
    return patch.astype(np.float32), params


def log_header(stains: Sequence[str]) -> List[str]:
    """Columns of generation_log.csv (the data loader reads Index and *_perturbed_*)."""
    header = ['Index', 'GT_filename', 'Flip_applied', 'Rotate_applied', 'Rotation_angle',
              'Elastic_applied', 'Elastic_alpha', 'Elastic_sigma', 'Scale_patch']
    header += [f'Scale_channel_{s}' for s in stains]
    header += ['Stain_matrix_attempts', 'Gaussian_noise_std', 'Min_SAD']
    header += [f'{s}_original_{c}' for s in stains for c in 'RGB']
    header += [f'{s}_perturbed_{c}' for s in stains for c in 'RGB']
    header += [f'SAD_{stains[i]}_{stains[j]}' for i in range(len(stains)) for j in range(i + 1, len(stains))]
    return header + ['Mean_SAD', 'Background_noise_R', 'Background_noise_G', 'Background_noise_B']


def main() -> None:
    parser = argparse.ArgumentParser(description='Synthetic OD / concentration generation')
    parser.add_argument('--config', default='config/synthetic_data_generation_config.ini')
    cfg = load_config(parser.parse_args().config)
    paths, aug, out = cfg['PATHS'], cfg['AUGMENTATION'], cfg['OUTPUT']

    stains = [s.strip() for s in cfg.get('STAIN_MATRIX', 'stain_order').split(',')]
    n_samples = cfg.getint('GENERATION', 'Generation_number')
    rng = np.random.default_rng(cfg.getint('GENERATION', 'random_seed', fallback=42))
    noise_std = aug.getfloat('gaussian_noise_std', 0.025)
    min_sad = aug.getfloat('min_SAD', 0.25)
    back_std = aug.getfloat('back_gaussian_std', 0.015)
    scale_patch = parse_scale(aug.get('Scale_patch', '2.0'))
    scale_channels = [parse_scale(aug.get(f'Scale_{s}', '2.0')) for s in stains]
    save_rgb = out.getboolean('Save_RGB', False)

    gt_files = sorted(glob.glob(os.path.join(paths['GT_dir'], '*.npy')))
    if not gt_files:
        raise FileNotFoundError(f"No .npy files in {paths['GT_dir']}")
    vectors = load_stain_vectors(paths['SM_dir'], stains)
    od_dir, rgb_dir = paths['OD_output_dir'], paths.get('RGB_output_dir', paths['OD_output_dir'])
    os.makedirs(od_dir, exist_ok=True)
    os.makedirs(rgb_dir, exist_ok=True)

    log_path = os.path.join(od_dir, out.get('log_filename', 'generation_log.csv'))
    with open(log_path, 'w', newline='', encoding='utf-8') as log_file:
        writer = csv.writer(log_file)
        writer.writerow(log_header(stains))
        for i in tqdm(range(1, n_samples + 1), desc='Generating'):
            gt_file = gt_files[rng.integers(0, len(gt_files))]
            gt = np.load(gt_file).astype(np.float32)
            if gt.shape != (256, 256, len(stains)):
                print(f"[WARN] Skipping {gt_file} with shape {gt.shape}")
                continue
            conc, params = augment_patch(gt, aug, scale_patch, scale_channels, rng)
            original, perturbed, sads, attempts = draw_stain_matrix(vectors, stains, noise_std, min_sad, rng)
            od = (conc.reshape(-1, len(stains)) @ perturbed).reshape(256, 256, 3).astype(np.float32)

            name = os.path.splitext(os.path.basename(gt_file))[0].replace('GT_con_', '', 1)
            prefix = f'Synthetic_{n_samples}_{name}'
            np.save(os.path.join(od_dir, f'{prefix}_OD_{i:06d}.npy'), od)
            np.save(os.path.join(od_dir, f'{prefix}_con_{i:06d}.npy'), conc)

            background = np.zeros(3, dtype=np.float32)
            if save_rgb:
                background = np.abs(rng.normal(0, back_std, size=3)).astype(np.float32)
                rgb = 255 * np.power(10, -(np.clip(od, 0, 10) + background))
                Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(
                    os.path.join(rgb_dir, f'{prefix}_RGB_{i:06d}.png'))

            writer.writerow([i, name, params['flip'], params['rotate'], params['angle'],
                             params['elastic'], params['alpha'], params['sigma'], params['scale_patch'],
                             *params['scale_channels'], attempts, noise_std, min_sad,
                             *original.flatten(), *perturbed.flatten(), *sads, *background])
    print(f"Generated {n_samples} samples in {od_dir}; log: {log_path}")


if __name__ == '__main__':
    main()
