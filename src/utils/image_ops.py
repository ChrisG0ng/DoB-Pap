"""Image I/O, background detection, optical density and patch tiling."""

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from skimage.filters import threshold_otsu

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
OD_EPSILON = 0.5
OTSU_BRIGHT_SHIFT = 0.75


def load_rgb_image(path: str) -> np.ndarray:
    """Load an image as an (H, W, 3) uint8 RGB array."""
    with Image.open(path) as img:
        return np.array(img.convert('RGB'), dtype=np.uint8)


def collect_image_files(root: str, extensions: Iterable[str] = IMAGE_EXTENSIONS,
                        selection_mode: str = 'sequence', random_seed: int = 42) -> List[Path]:
    """
    List images under a file or directory (recursively).

    Args:
        selection_mode: 'sequence' (sorted by relative path) or 'random' (seeded shuffle).
    """
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")
    if root_path.is_file():
        return [root_path]

    exts = {e.lower() if e.startswith('.') else f'.{e.lower()}' for e in extensions}
    files = sorted(
        (p for p in root_path.rglob('*') if p.is_file() and p.suffix.lower() in exts),
        key=lambda p: p.relative_to(root_path).as_posix(),
    )
    mode = selection_mode.strip().lower()
    if mode == 'random':
        order = np.random.default_rng(random_seed).permutation(len(files))
        return [files[i] for i in order]
    if mode != 'sequence':
        raise ValueError(f"selection_mode must be 'sequence' or 'random', got {selection_mode!r}")
    return files


def relative_path(path: Path, root: str) -> Path:
    """Path of an image relative to the input root (its file name if root is a file)."""
    root_path = Path(root)
    return Path(path.name) if root_path.is_file() else path.relative_to(root_path)


def background_mask_auto(image: np.ndarray, tolerance: float = 0.01) -> np.ndarray:
    """
    Background = pixels whose normalized RGB sum is within `tolerance` of the most
    frequent RGB sum (falls back to the maximum RGB sum when that peak is < 1.8).
    """
    rgb_sum = np.round(np.sum(np.round(image.astype(np.float32) / 255.0, 2), axis=2), 2)
    values, counts = np.unique(rgb_sum, return_counts=True)
    peak = values[np.argmax(counts)]
    if peak < 1.8:
        peak = np.max(rgb_sum)
    return rgb_sum >= (peak - tolerance)


def background_mask_otsu(image: np.ndarray) -> np.ndarray:
    """
    Otsu threshold T on luminance (pure-white pixels excluded), shifted toward the
    bright-class mean: T' = T + 0.75 * (mu_bright - T). Pure white is always background.
    """
    pure_white = np.all(image == 255, axis=2)
    rgb = image.astype(np.float64)
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    pixels = gray[~pure_white]
    if pixels.size == 0:
        return np.ones(gray.shape, dtype=bool)
    try:
        thresh = float(threshold_otsu(pixels))
    except ValueError:
        thresh = float(np.max(pixels))
    bright = pixels[pixels >= thresh]
    if bright.size > 0:
        thresh += OTSU_BRIGHT_SHIFT * (float(np.mean(bright)) - thresh)
    return (gray >= thresh) | pure_white


def background_masks(image: np.ndarray, method: str = 'auto') -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (mask_i0, mask_patch): the mask used to estimate I0 and the mask used
    to measure the background fraction of patches.
    """
    method = method.strip().lower()
    if method == 'auto':
        return background_mask_auto(image, 0.01), background_mask_auto(image, 0.1)
    if method == 'otsu':
        mask = background_mask_otsu(image)
        return mask, mask
    raise ValueError(f"Background method must be 'auto' or 'Otsu', got {method!r}")


def background_intensity(image: np.ndarray, mask: np.ndarray, epsilon: float = OD_EPSILON) -> np.ndarray:
    """Mean RGB of background pixels (I0), float64 (3,); 255 when no background."""
    if np.any(mask):
        i0 = np.mean(image[mask].astype(np.float64), axis=0)
    else:
        i0 = np.full(3, 255.0)
    return np.maximum(i0, epsilon)


def optical_density(image: np.ndarray, i0: np.ndarray, epsilon: float = OD_EPSILON) -> np.ndarray:
    """OD = -log10((I + eps) / I0), clipped at 0, float32 (H, W, 3)."""
    od = -np.log10((image.astype(np.float32) + epsilon) / i0.reshape(1, 1, 3))
    return np.maximum(od, 0.0).astype(np.float32)


def od_background(i0: np.ndarray) -> np.ndarray:
    """Background optical density -log10(I0 / 255), shape (3,)."""
    return -np.log10(i0 / 255.0)


def ceil_to_multiple(length: int, n: int) -> int:
    """Smallest multiple of n that is >= length (at least n)."""
    return max(n, -(-length // n) * n)


def reflect_pad(image: np.ndarray, target_h: int, target_w: int, center: bool = False) -> np.ndarray:
    """Reflect-pad (H, W[, C]) to at least (target_h, target_w); repeats when the pad exceeds the size."""
    img = image
    while img.shape[0] < target_h or img.shape[1] < target_w:
        h, w = img.shape[:2]
        ph = min(max(target_h - h, 0), max(h - 1, 1))
        pw = min(max(target_w - w, 0), max(w - 1, 1))
        top = ph // 2 if center else 0
        left = pw // 2 if center else 0
        pad = [(top, ph - top), (left, pw - left)] + [(0, 0)] * (img.ndim - 2)
        img = np.pad(img, pad, mode='reflect' if h > 1 and w > 1 else 'edge')
    return img[:target_h, :target_w]


def pad_to_multiple(image: np.ndarray, n: int) -> np.ndarray:
    """Reflect-pad bottom/right so that H and W are multiples of n."""
    h, w = image.shape[:2]
    th, tw = ceil_to_multiple(h, n), ceil_to_multiple(w, n)
    return image if (th, tw) == (h, w) else reflect_pad(image, th, tw)


def fit_to_square(image: np.ndarray, size: int) -> np.ndarray:
    """Downscale so the long edge is <= size (LANCZOS), then center reflect-pad to size x size."""
    h, w = image.shape[:2]
    if h > size or w > size:
        scale = size / float(max(h, w))
        new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        image = np.array(Image.fromarray(image).resize((new_w, new_h), Image.Resampling.LANCZOS))
    if image.shape[:2] == (size, size):
        return image
    return reflect_pad(image, size, size, center=True)


def parse_patch_mode(value: str, single_size: int) -> Tuple[str, int]:
    """
    'single' -> ('single', single_size): each image is fitted to one square patch.
    Integer n -> ('grid', n): reflect-pad to multiples of n and split into n x n patches.
    """
    value = value.strip().lower()
    if value == 'single':
        return 'single', single_size
    if not value.isdigit() or int(value) <= 0:
        raise ValueError(f"patch_mode must be a positive integer or 'single', got {value!r}")
    return 'grid', int(value)


def prepare_image(rgb: np.ndarray, mode: str, size: int) -> np.ndarray:
    """Fit the image to one square patch ('single') or pad it to a multiple of `size` ('grid')."""
    return fit_to_square(rgb, size) if mode == 'single' else pad_to_multiple(rgb, size)


def _axis_starts(length: int, patch: int, stride: int) -> List[int]:
    if length <= patch:
        return [0]
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


def tile_positions(height: int, width: int, patch: int, stride: Optional[int] = None) -> List[Tuple[int, int]]:
    """Top-left corners of tiles covering the image; the last row/column is aligned to the border."""
    stride = stride or patch
    return [(r, c) for r in _axis_starts(height, patch, stride) for c in _axis_starts(width, patch, stride)]


def extract_tiles(array: np.ndarray, positions: Sequence[Tuple[int, int]], patch: int) -> List[np.ndarray]:
    """Crop (patch, patch[, C]) tiles at the given positions."""
    return [array[r:r + patch, c:c + patch] for r, c in positions]


def stitch_tiles(tiles: Sequence[np.ndarray], positions: Sequence[Tuple[int, int]],
                 shape: Tuple[int, int]) -> np.ndarray:
    """Average overlapping (h, w, C) tiles into an (H, W, C) map."""
    height, width = shape
    out = np.zeros((height, width, tiles[0].shape[2]), dtype=np.float32)
    count = np.zeros((height, width, 1), dtype=np.float32)
    for tile, (r, c) in zip(tiles, positions):
        th, tw = min(tile.shape[0], height - r), min(tile.shape[1], width - c)
        out[r:r + th, c:c + tw] += tile[:th, :tw]
        count[r:r + th, c:c + tw] += 1
    return out / np.maximum(count, 1)


@torch.no_grad()
def predict_concentrations(model: torch.nn.Module, od_patches: Sequence[np.ndarray], device: str,
                           batch_size: int = 32, scaler=None) -> List[np.ndarray]:
    """Predict (h, w, 4) concentrations (original scale if `scaler` is given) for OD patches."""
    model.eval()
    results = []
    for i in range(0, len(od_patches), batch_size):
        batch = torch.from_numpy(np.stack(od_patches[i:i + batch_size])).float().permute(0, 3, 1, 2)
        conc = model(batch.to(device)).cpu().numpy().transpose(0, 2, 3, 1).astype(np.float32)
        if scaler is not None:
            conc = scaler.inverse_scale_concentrations(conc).astype(np.float32)
        results.extend(conc)
    return results
