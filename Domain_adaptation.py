"""
Domain adaptation of the UNet2 stain unmixing model.

  Step 1  Unmix target-domain RGB images with the source model
          -> patch_output/ (concentration), od_output/ (OD), background/background.txt
  Step 2  Estimate the domain stain matrix -> stain_matrix/overall_stain_matrix.txt
  Step 3  Fine-tune the source model with that matrix -> finetuned_model/unet2_stain_best.pth

Usage:
    python Domain_adaptation.py --config config/domain_adaptation_config.ini
"""

import argparse
import os
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from src.models import UNet2
from src.training import finetune_with_stain_matrix
from src.utils import image_ops as imops
from src.utils.common import (device_from_section, load_checkpoint, load_config, load_weights,
                              write_background_report)
from src.utils.concentration_scaling import scaler_from_checkpoint
from src.utils.stain_estimation import estimate_domain_stain_matrix

PATCH_SIZE = 256


def _pad_mask(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    return np.pad(mask, ((0, h - mask.shape[0]), (0, w - mask.shape[1])))


def unmix_images(model, scaler, input_dir: str, out_dir: str, device: str, batch_size: int,
                 extensions: List[str], max_background: float) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Step 1: tile each image into 256x256 patches (border-aligned), predict concentrations
    (background pixels set to 0) and keep OD patches with background fraction <= max_background.

    Returns:
        (od_patches, concentration_patches) of the kept patches.
    """
    files = imops.collect_image_files(input_dir, extensions)
    if not files:
        raise FileNotFoundError(f"No images found in {input_dir}")
    print(f"[Step 1] {len(files)} image(s) in {input_dir}")

    kept_od, kept_conc, bg_rows = [], [], []
    for path in tqdm(files, desc='Step 1 - Stain unmixing'):
        rel = imops.relative_path(path, input_dir)
        rgb = imops.load_rgb_image(str(path))
        mask_i0, mask_patch = imops.background_masks(rgb, 'auto')
        i0 = imops.background_intensity(rgb, mask_i0)
        bg_rows.append((rel.as_posix(), i0, imops.od_background(i0)))

        od = imops.optical_density(rgb, i0)
        od[mask_i0] = 0.0
        h, w = max(od.shape[0], PATCH_SIZE), max(od.shape[1], PATCH_SIZE)
        od = imops.reflect_pad(od, h, w)
        mask_i0, mask_patch = _pad_mask(mask_i0, h, w), _pad_mask(mask_patch, h, w)

        positions = imops.tile_positions(h, w, PATCH_SIZE)
        od_tiles = imops.extract_tiles(od, positions, PATCH_SIZE)
        conc_tiles = imops.predict_concentrations(model, od_tiles, device, batch_size, scaler)
        i0_tiles = imops.extract_tiles(mask_i0, positions, PATCH_SIZE)
        bg_tiles = imops.extract_tiles(mask_patch, positions, PATCH_SIZE)

        conc_dir = os.path.join(out_dir, 'patch_output', rel.parent)
        od_dir = os.path.join(out_dir, 'od_output', rel.parent)
        os.makedirs(conc_dir, exist_ok=True)
        os.makedirs(od_dir, exist_ok=True)
        for idx, (od_t, conc_t, i0_t, bg_t) in enumerate(zip(od_tiles, conc_tiles, i0_tiles, bg_tiles)):
            conc_t[i0_t] = 0.0
            np.save(os.path.join(conc_dir, f'{path.stem}_{idx}.npy'), conc_t)
            if bg_t.mean() <= max_background:
                np.save(os.path.join(od_dir, f'{path.stem}_OD_{idx}.npy'), od_t)
                kept_od.append(od_t)
                kept_conc.append(conc_t)

    write_background_report(bg_rows, os.path.join(out_dir, 'background', 'background.txt'), imops.OD_EPSILON)
    print(f"[Step 1] Kept {len(kept_od)} OD patches.")
    return kept_od, kept_conc


def main() -> None:
    parser = argparse.ArgumentParser(description='UNet2 domain adaptation')
    parser.add_argument('--config', default='config/domain_adaptation_config.ini')
    cfg = load_config(parser.parse_args().config)
    da, inf, est, ft = cfg['DOMAIN_ADAPTATION'], cfg['INFERENCE'], cfg['ESTIMATION'], cfg['FINETUNE']

    out_dir = da['output_base_dir']
    device = device_from_section(da)
    checkpoint = load_checkpoint(da['model_path'])
    scaler = scaler_from_checkpoint(checkpoint)
    model = load_weights(UNet2(), checkpoint).to(device)
    print(f"Device: {device} | source model: {da['model_path']} | {scaler}")

    extensions = inf.get('image_extensions', ','.join(imops.IMAGE_EXTENSIONS))
    od_patches, conc_patches = unmix_images(
        model, scaler, da['input_dir'], out_dir, device,
        batch_size=inf.getint('batch_size', 64),
        extensions=[e.strip() for e in extensions.split(',')],
        max_background=inf.getfloat('max_background_fraction', 0.8),
    )
    del model

    print("[Step 2] Estimating domain stain matrix")
    stain_matrix = estimate_domain_stain_matrix(
        od_patches, conc_patches, os.path.join(out_dir, 'stain_matrix', 'overall_stain_matrix.txt'),
        top_percentage=est.getfloat('top_percentage', 0.05),
        confidence_level=est.getfloat('confidence_level', 0.95),
    )
    print(np.array2string(stain_matrix, precision=4))

    print("[Step 3] Fine-tuning")
    dropout_keep_prob = ft.getfloat('dropout_keep_prob', 0.95)
    best = finetune_with_stain_matrix(
        lambda: UNet2(dropout_keep_prob=dropout_keep_prob), checkpoint,
        od_dir=os.path.join(out_dir, 'od_output'), stain_matrix=stain_matrix,
        save_dir=os.path.join(out_dir, 'finetuned_model'), model_name='unet2_stain',
        section=ft, device=device, compute_scaler=True,
    )
    print(f"Done. Fine-tuned model: {best}")


if __name__ == '__main__':
    main()
