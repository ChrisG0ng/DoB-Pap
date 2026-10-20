"""
Stain unmixing of RGB images with a (fine-tuned) UniCAS model.

Outputs under output_dir:
    concentration/    {name}_concentration.npy, (H, W, 4) EY/H/LG/OG
    reconstruction/   per-stain and full reconstructed RGB (Do_reconstruction = true)
    background/       background.txt (I0 and OD_background per image)
    scaler/           scaler.txt (mean + 2*std and c99 of foreground predictions)

Usage:
    python UniCAS_inference.py --config config/uniCAS_inference_config.ini
"""

import argparse
import os

from tqdm import tqdm

from src.models import UniCASStainUnmixing
from src.utils import image_ops as imops
from src.utils.common import (PixelStats, device_from_section, load_checkpoint, load_config, load_weights,
                              reconstruction_stain_matrix, save_unmixing_results, unicas_kwargs,
                              write_background_report)
from src.utils.concentration_scaling import DEFAULT_FIXED_C5, scaler_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description='UniCAS stain unmixing inference')
    parser.add_argument('--config', default='config/uniCAS_inference_config.ini')
    cfg = load_config(parser.parse_args().config)
    test = cfg['TESTING']
    input_dir = cfg['DATA']['test_data_dir']
    output_dir = cfg['PATHS']['output_dir']
    method = test.get('Background_method', 'Otsu')
    batch_size = test.getint('batch_size', 32)
    crop_to_input = test.getboolean('tile_to_initial', True)

    device = device_from_section(cfg['DEVICE'])
    checkpoint = load_checkpoint(test['model_path'])
    scaler = scaler_from_checkpoint(checkpoint)
    model = load_weights(UniCASStainUnmixing(**unicas_kwargs(cfg['MODEL'])), checkpoint).to(device)
    mode, size = imops.parse_patch_mode(test.get('patch_mode', '256'), model.img_size)
    stain_matrix = reconstruction_stain_matrix(test)

    files = imops.collect_image_files(input_dir, selection_mode=test.get('selection_mode', 'sequence'),
                                      random_seed=test.getint('random_seed', 42))
    limit = test.getint('inference_number', 0)
    files = files[:limit] if limit > 0 else files
    print(f"Device: {device} | model: {test['model_path']} | {scaler} | {len(files)} image(s) | "
          f"patch_mode={mode}:{size} | background={method}")

    stats, bg_rows = PixelStats(), []
    for path in tqdm(files, desc='Inference'):
        rel = imops.relative_path(path, input_dir)
        rgb = imops.load_rgb_image(str(path))
        height, width = rgb.shape[:2]
        rgb = imops.prepare_image(rgb, mode, size)
        mask, _ = imops.background_masks(rgb, method)
        i0 = imops.background_intensity(rgb, mask)
        od_bg = imops.od_background(i0)
        bg_rows.append((rel.as_posix(), i0, od_bg))

        od = imops.optical_density(rgb, i0)
        positions = imops.tile_positions(od.shape[0], od.shape[1], size)
        conc_tiles = imops.predict_concentrations(
            model, imops.extract_tiles(od, positions, size), device, batch_size, scaler)
        conc_map = imops.stitch_tiles(conc_tiles, positions, od.shape[:2])
        if mode == 'grid' and crop_to_input:
            conc_map, mask = conc_map[:height, :width], mask[:height, :width]
        stats.update(conc_map[~mask])

        save_unmixing_results(conc_map, od_bg, output_dir, str(rel.parent), path.stem, stain_matrix)

    write_background_report(bg_rows, os.path.join(output_dir, 'background', 'background.txt'), imops.OD_EPSILON)
    stats.write_report(os.path.join(output_dir, 'scaler', 'scaler.txt'), 'Concentration scaler statistics (inference)',
                       cfg.getfloat('SCALER', 'fixed_c5', fallback=DEFAULT_FIXED_C5), scaler is not None)
    print(f"Results saved to {output_dir}")


if __name__ == '__main__':
    main()
