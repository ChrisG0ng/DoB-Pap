"""
Domain adaptation of the UniCAS stain unmixing model.

  Step 1  Unmix target-domain RGB images with the source model
          -> patch_output/, od_output/, background/background.txt, scaler/scaler.txt
  Step 2  Estimate the domain stain matrix -> stain_matrix/overall_stain_matrix.txt
  Step 3  Fine-tune the source model with that matrix -> finetuned_model/uniCAS_stain_best.pth

Usage:
    python UniCAS_domain_adaptation.py --config config/uniCAS_domain_adaptation_config.ini
"""

import argparse
import os
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from src.models import UniCASStainUnmixing
from src.training import finetune_with_stain_matrix
from src.utils import image_ops as imops
from src.utils.common import (PixelStats, device_from_section, load_checkpoint, load_config, load_weights,
                              unicas_kwargs, write_background_report)
from src.utils.concentration_scaling import DEFAULT_FIXED_C5, scaler_from_checkpoint
from src.utils.stain_estimation import estimate_domain_stain_matrix


def unmix_images(model, scaler, inf, input_dir: str, out_dir: str, device: str,
                 fixed_c5: float) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Step 1: predict concentrations for every patch and keep OD patches whose background
    fraction is <= max_background_fraction, until max_od_patches have been kept.

    Returns:
        (od_patches, concentration_patches) of the kept patches.
    """
    mode, size = imops.parse_patch_mode(inf.get('patch_mode', '256'), model.img_size)
    method = inf.get('Background_method', 'Otsu')
    max_background = inf.getfloat('max_background_fraction', 0.75)
    max_patches = inf.getint('max_od_patches', 0)
    batch_size = inf.getint('batch_size', 32)
    extensions = [e.strip() for e in inf.get('image_extensions', ','.join(imops.IMAGE_EXTENSIONS)).split(',')]
    files = imops.collect_image_files(input_dir, extensions, inf.get('selection_mode', 'sequence'),
                                      inf.getint('random_seed', 42))
    if not files:
        raise FileNotFoundError(f"No images found in {input_dir}")
    print(f"[Step 1] {len(files)} image(s) | patch_mode={mode}:{size} | background={method}")

    kept_od, kept_conc, bg_rows = [], [], []
    stats = PixelStats()
    for path in tqdm(files, desc='Step 1 - Stain unmixing'):
        rel = imops.relative_path(path, input_dir)
        rgb = imops.prepare_image(imops.load_rgb_image(str(path)), mode, size)
        mask_i0, mask_patch = imops.background_masks(rgb, method)
        i0 = imops.background_intensity(rgb, mask_i0)
        bg_rows.append((rel.as_posix(), i0, imops.od_background(i0)))

        od = imops.optical_density(rgb, i0)
        positions = imops.tile_positions(od.shape[0], od.shape[1], size)
        od_tiles = imops.extract_tiles(od, positions, size)
        conc_tiles = imops.predict_concentrations(model, od_tiles, device, batch_size, scaler)
        i0_tiles = imops.extract_tiles(mask_i0, positions, size)
        bg_tiles = imops.extract_tiles(mask_patch, positions, size)

        conc_dir = os.path.join(out_dir, 'patch_output', rel.parent)
        od_dir = os.path.join(out_dir, 'od_output', rel.parent)
        os.makedirs(conc_dir, exist_ok=True)
        os.makedirs(od_dir, exist_ok=True)
        for idx, (od_t, conc_t, i0_t, bg_t) in enumerate(zip(od_tiles, conc_tiles, i0_tiles, bg_tiles)):
            np.save(os.path.join(conc_dir, f'{path.stem}_{idx}.npy'), conc_t)
            stats.update(conc_t[~i0_t])
            if bg_t.mean() <= max_background:
                np.save(os.path.join(od_dir, f'{path.stem}_OD_{idx}.npy'), od_t)
                kept_od.append(od_t)
                kept_conc.append(conc_t)

        if 0 < max_patches <= len(kept_od):
            print(f"[Step 1] Reached {len(kept_od)} OD patches (max_od_patches={max_patches}).")
            break

    write_background_report(bg_rows, os.path.join(out_dir, 'background', 'background.txt'), imops.OD_EPSILON)
    stats.write_report(os.path.join(out_dir, 'scaler', 'scaler.txt'), 'Concentration scaler statistics (Step 1)',
                       fixed_c5, inverse_scaled=scaler is not None)
    print(f"[Step 1] Kept {len(kept_od)} OD patches.")
    return kept_od, kept_conc


def main() -> None:
    parser = argparse.ArgumentParser(description='UniCAS domain adaptation')
    parser.add_argument('--config', default='config/uniCAS_domain_adaptation_config.ini')
    cfg = load_config(parser.parse_args().config)
    da, inf, est, ft = cfg['DOMAIN_ADAPTATION'], cfg['INFERENCE'], cfg['ESTIMATION'], cfg['FINETUNE']
    fixed_c5 = cfg.getfloat('SCALER', 'fixed_c5', fallback=DEFAULT_FIXED_C5)
    kwargs = unicas_kwargs(cfg['MODEL'])

    out_dir = da['output_base_dir']
    device = device_from_section(da)
    checkpoint = load_checkpoint(da['model_path'])
    scaler = scaler_from_checkpoint(checkpoint)
    model = load_weights(UniCASStainUnmixing(**kwargs), checkpoint).to(device)
    print(f"Device: {device} | source model: {da['model_path']} | {scaler}")

    od_patches, conc_patches = unmix_images(model, scaler, inf, da['input_dir'], out_dir, device, fixed_c5)
    del model

    print("[Step 2] Estimating domain stain matrix")
    stain_matrix = estimate_domain_stain_matrix(
        od_patches, conc_patches, os.path.join(out_dir, 'stain_matrix', 'overall_stain_matrix.txt'),
        top_percentage=est.getfloat('top_percentage', 0.05),
        confidence_level=est.getfloat('confidence_level', 0.95),
        sad_threshold=est.getfloat('sad_outlier_threshold', 0.4),
    )
    print(np.array2string(stain_matrix, precision=4))

    print("[Step 3] Fine-tuning")
    best = finetune_with_stain_matrix(
        lambda: UniCASStainUnmixing(**kwargs), checkpoint,
        od_dir=os.path.join(out_dir, 'od_output'), stain_matrix=stain_matrix,
        save_dir=os.path.join(out_dir, 'finetuned_model'), model_name='uniCAS_stain',
        section=ft, device=device, compute_scaler=ft.getboolean('compute_scaler', False),
        fixed_c5=fixed_c5,
    )
    print(f"Done. Fine-tuned model: {best}")


if __name__ == '__main__':
    main()
