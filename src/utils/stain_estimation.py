"""Domain stain matrix estimation from OD patches and predicted concentrations."""

import ast
import os
from typing import List, Optional, Sequence

import numpy as np
from scipy.optimize import minimize, nnls
from scipy.stats import chi2
from tqdm import tqdm

from .concentration_scaling import STAIN_NAMES

# Reference Pap stain vectors (rows EY, H, LG, OG; columns R, G, B) for the SAD filter.
PAP_STAIN_ANCHOR = np.array([
    [0.0, 0.9517, 0.3070],
    [0.7322, 0.6460, 0.2158],
    [0.9447, 0.1839, 0.2716],
    [0.0, 0.2539, 0.9672],
])


def _normalize_columns(a: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(a, axis=0)
    return np.where(norms > 1e-10, a / np.where(norms > 1e-10, norms, 1.0), 1.0 / np.sqrt(a.shape[0]))


def estimate_patch_stain_matrix(conc: np.ndarray, od: np.ndarray, max_iter: int = 1000) -> np.ndarray:
    """
    Solve min ||A C - Y||^2 s.t. A >= 0, ||A[:, j]|| = 1 (NNLS init + SLSQP).

    Args:
        conc: Concentrations C of shape (K, N).
        od: Optical densities Y of shape (3, N).

    Returns:
        Stain matrix A of shape (3, K) with unit-norm columns.
    """
    m, k = od.shape[0], conc.shape[0]
    a_init = _normalize_columns(np.stack([nnls(conc.T, od[i])[0] for i in range(m)]))

    def objective(a_flat):
        return np.sum((a_flat.reshape(m, k) @ conc - od) ** 2)

    def gradient(a_flat):
        return (2.0 * (a_flat.reshape(m, k) @ conc - od) @ conc.T).flatten()

    constraints = [
        {'type': 'eq', 'fun': lambda a_flat, j=j: np.sum(a_flat.reshape(m, k)[:, j] ** 2) - 1.0}
        for j in range(k)
    ]
    result = minimize(objective, a_init.flatten(), method='SLSQP', jac=gradient,
                      bounds=[(0.0, None)] * (m * k), constraints=constraints,
                      options={'maxiter': max_iter, 'ftol': 1e-9})
    return _normalize_columns(np.clip(result.x.reshape(m, k), 0.0, None))


def patch_dominance(conc_patches: Sequence[np.ndarray], threshold: float = 0.1) -> np.ndarray:
    """
    Per-patch dominance score of each stain, shape (n, K).

    Concentrations are normalized by global mean + 2*std. For stain i, pixels whose
    normalized value exceeds `threshold` are selected and the score is the share of
    stain i in the mean concentration of those pixels.
    """
    flat = [p.reshape(-1, p.shape[-1]).T for p in conc_patches]
    all_x = np.concatenate(flat, axis=1)
    scale = np.mean(all_x, axis=1) + 2.0 * np.std(all_x, axis=1)
    scale = np.where(scale <= 1e-10, 1.0, scale)

    scores = np.zeros((len(flat), all_x.shape[0]))
    for n, x in enumerate(flat):
        x_sc = x / scale[:, None]
        for i in range(x.shape[0]):
            mask = x_sc[i] > threshold
            if not np.any(mask):
                continue
            means = x_sc[:, mask].mean(axis=1)
            total = means.sum()
            scores[n, i] = means[i] / total if total > 0 else 0.0
    return scores


def spectral_angle(vectors: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Angle (radians) between each row of `vectors` and `anchor`."""
    denom = np.maximum(np.linalg.norm(vectors, axis=1) * np.linalg.norm(anchor), 1e-12)
    return np.arccos(np.clip(vectors @ anchor / denom, -1.0, 1.0))


def mahalanobis_inliers(data: np.ndarray, confidence_level: float) -> np.ndarray:
    """Indices of rows whose Mahalanobis distance is below the chi-square threshold."""
    if len(data) < 2:
        return np.arange(len(data))
    diff = data - data.mean(axis=0)
    cov = np.cov(data, rowvar=False) + np.eye(data.shape[1]) * 1e-6
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)
    dist = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', diff, cov_inv, diff), 0.0))
    return np.where(dist < np.sqrt(chi2.ppf(confidence_level, data.shape[1])))[0]


def write_stain_matrix_report(path: str, wm: np.ndarray, lines: List[str]) -> None:
    """Write the estimated (4, 3) matrix; the last line is parsed by :func:`read_stain_matrix`."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    out = ['Overall Stain Matrix Estimation', '=' * 60, *lines, '',
           'Stain matrix (rows = EY/H/LG/OG, columns = R/G/B, unit row norm):',
           '       R         G         B']
    out += [f"{name}:  {r:.6f}  {g:.6f}  {b:.6f}" for name, (r, g, b) in zip(STAIN_NAMES, wm)]
    out += ['', 'Transposed (3x4, rows = R/G/B):', f"fixed_stain_matrix = {wm.T.tolist()}", '']
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))


def read_stain_matrix(path: str) -> np.ndarray:
    """Read the (3, 4) matrix from the last 'fixed_stain_matrix = ...' line of a report."""
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in reversed(lines):
        if line.strip().startswith('fixed_stain_matrix'):
            matrix = np.array(ast.literal_eval(line.split('=', 1)[1].strip()), dtype=np.float32)
            if matrix.shape != (3, 4):
                raise ValueError(f"Stain matrix must be 3x4, got {matrix.shape}")
            return matrix
    raise ValueError(f"No 'fixed_stain_matrix = ...' line found in {path}")


def estimate_domain_stain_matrix(od_patches: Sequence[np.ndarray], conc_patches: Sequence[np.ndarray],
                                 report_path: str, top_percentage: float = 0.05,
                                 confidence_level: float = 0.95,
                                 sad_threshold: Optional[float] = None) -> np.ndarray:
    """
    Estimate a domain stain matrix of shape (4, 3) with unit row norms.

    For each stain, per-patch matrices of the top `top_percentage` patches ranked by
    stain dominance are filtered by spectral angle to the Pap anchor (if
    `sad_threshold` is set) and by Mahalanobis distance, then averaged.
    """
    if not od_patches:
        raise RuntimeError("No OD patches available for stain matrix estimation.")
    matrices, valid_conc = [], []
    for od, conc in tqdm(zip(od_patches, conc_patches), total=len(od_patches), desc='Per-patch stain matrix'):
        try:
            matrices.append(estimate_patch_stain_matrix(conc.reshape(-1, 4).T, od.reshape(-1, 3).T))
            valid_conc.append(conc)
        except Exception as exc:
            print(f"  [WARN] Patch skipped: {exc}")
    if not matrices:
        raise RuntimeError("No stain matrices could be estimated.")

    scores = patch_dominance(valid_conc)
    vectors = np.stack(matrices).transpose(0, 2, 1)  # (n, K, 3)
    n_top = max(1, int(np.ceil(len(matrices) * top_percentage)))

    wm = np.zeros((4, 3))
    lines = [f"Patches: {len(matrices)}", f"Top percentage: {top_percentage * 100:.1f}% ({n_top} per stain)",
             f"Confidence level: {confidence_level * 100:.1f}%"]
    if sad_threshold is not None:
        lines.append(f"SAD threshold: {sad_threshold:.4f} rad")
    for i, name in enumerate(STAIN_NAMES):
        top = vectors[np.argsort(scores[:, i])[::-1][:n_top], i]
        if sad_threshold is not None:
            sads = spectral_angle(top, PAP_STAIN_ANCHOR[i])
            keep = sads <= sad_threshold
            top = top[keep] if np.any(keep) else top[[int(np.argmin(sads))]]
        inliers = mahalanobis_inliers(top, confidence_level)
        wm[i] = top[inliers].mean(axis=0)
        lines.append(f"  {name}: {len(inliers)} inliers / {len(top)} candidates")
        print(lines[-1])

    wm = wm / np.maximum(np.linalg.norm(wm, axis=1, keepdims=True), 1e-10)
    write_stain_matrix_report(report_path, wm, lines)
    return wm
