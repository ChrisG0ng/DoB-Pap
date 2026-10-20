"""
Dataset of OD patches ('*_OD_*.npy') with optional concentration patches
('*_con_*.npy') and per-patch ground-truth stain matrices (generation_log.csv).
"""

import csv
import glob
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .augmentation import PairTransform

OD_TAG = '_OD_'
CONC_TAG = '_con_'
STAIN_COLUMNS = [f'{s}_perturbed_{c}' for s in ('EY', 'H', 'LG', 'OG') for c in 'RGB']


def find_od_files(data_dir: str, require_concentration: bool) -> List[str]:
    """Relative paths (without .npy) of OD patches, optionally only those with a concentration file."""
    indices = []
    for path in glob.glob(os.path.join(data_dir, '**', f'*{OD_TAG}*.npy'), recursive=True):
        rel = os.path.splitext(os.path.relpath(path, data_dir))[0]
        if require_concentration and not os.path.exists(os.path.join(data_dir, rel.replace(OD_TAG, CONC_TAG) + '.npy')):
            print(f"[WARN] Missing concentration file for {rel}")
            continue
        indices.append(rel)
    return sorted(indices)


def load_gt_stain_matrices(data_dir: str) -> Dict[int, np.ndarray]:
    """Map sample index -> (3, 4) stain matrix from generation_log.csv (empty if absent)."""
    path = os.path.join(data_dir, 'generation_log.csv')
    if not os.path.exists(path):
        return {}
    matrices = {}
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            values = np.array([float(row[c]) for c in STAIN_COLUMNS], dtype=np.float32)
            matrices[int(row['Index'])] = values.reshape(4, 3).T
    return matrices


class StainDataset(Dataset):
    """
    Returns dicts with 'optical_density' (3, H, W) and, when available,
    'concentration' (4, H, W) and 'stain_matrix' (3, 4).
    """

    def __init__(self, data_dir: str, indices: List[str], transform: PairTransform,
                 use_concentration: bool = True, stain_matrices: Optional[Dict[int, np.ndarray]] = None):
        self.data_dir = data_dir
        self.indices = indices
        self.transform = transform
        self.use_concentration = use_concentration
        self.stain_matrices = stain_matrices or {}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict:
        index = self.indices[idx]
        od = np.load(os.path.join(self.data_dir, f'{index}.npy')).astype(np.float32)
        conc = None
        if self.use_concentration:
            conc = np.load(os.path.join(self.data_dir, f'{index.replace(OD_TAG, CONC_TAG)}.npy')).astype(np.float32)
        od_t, conc_t = self.transform(od, conc)

        sample = {'optical_density': od_t, 'file_index': index}
        if conc_t is not None:
            sample['concentration'] = conc_t
        if self.stain_matrices:
            sample['stain_matrix'] = torch.from_numpy(self.stain_matrices[int(index.split('_')[-1])])
        return sample


def _split_by_prefix(indices: List[str], validation_split: float, seed: int) -> Tuple[List[str], List[str]]:
    """Split so that patches sharing a source prefix stay in the same subset."""
    groups: Dict[str, List[str]] = {}
    for index in indices:
        groups.setdefault(re.sub(r'_(con|OD)_\d{6}$', '', index), []).append(index)
    np.random.seed(seed)
    prefixes = np.random.permutation(list(groups.keys()))
    val_prefixes = set(prefixes[:int(len(prefixes) * validation_split)])
    train = [i for p, items in groups.items() if p not in val_prefixes for i in items]
    val = [i for p, items in groups.items() if p in val_prefixes for i in items]
    return train, val


def create_data_loaders(data_dir: str, batch_size: int = 32, num_workers: int = 0,
                        validation_split: float = 0.2, random_seed: int = 42, scaler=None,
                        use_concentration: bool = True,
                        use_gt_stain_matrix: bool = True) -> Tuple[DataLoader, Optional[DataLoader], Dict]:
    """
    Build training (randomly augmented) and validation loaders.

    Returns:
        (train_loader, val_loader or None if no validation samples, info dict).
    """
    indices = find_od_files(data_dir, require_concentration=use_concentration)
    if not indices:
        raise ValueError(f"No OD patches ('*{OD_TAG}*.npy') found in {data_dir}")
    stain_matrices = load_gt_stain_matrices(data_dir) if use_gt_stain_matrix else {}
    train_idx, val_idx = _split_by_prefix(indices, validation_split, random_seed)

    def make_loader(subset: List[str], augment: bool) -> DataLoader:
        dataset = StainDataset(data_dir, subset, PairTransform(augment=augment, scaler=scaler),
                               use_concentration=use_concentration, stain_matrices=stain_matrices)
        return DataLoader(dataset, batch_size=batch_size, shuffle=augment, num_workers=num_workers,
                          pin_memory=torch.cuda.is_available())

    val_loader = make_loader(val_idx, augment=False) if val_idx else None
    info = {'total_samples': len(indices), 'train_samples': len(train_idx), 'val_samples': len(val_idx)}
    return make_loader(train_idx, augment=True), val_loader, info
