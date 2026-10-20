"""Unit tests: python -m unittest discover tests"""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.models import UNet2, UniCASStainUnmixing
from src.utils import image_ops as imops
from src.utils.common import PixelStats, write_background_report
from src.utils.concentration_scaling import ConcentrationScaler, c99_per_channel, normal_statistics
from src.utils.loss import UnmixingLoss, reconstruct_od
from src.utils.stain_estimation import (PAP_STAIN_ANCHOR, estimate_patch_stain_matrix, mahalanobis_inliers,
                                        read_stain_matrix, spectral_angle, write_stain_matrix_report)


class ImageOpsTest(unittest.TestCase):
    def test_tile_positions_cover_and_align_to_border(self):
        positions = imops.tile_positions(600, 520, 256)
        self.assertEqual(sorted({r for r, _ in positions}), [0, 256, 344])
        self.assertEqual(sorted({c for _, c in positions}), [0, 256, 264])
        self.assertEqual(imops.tile_positions(100, 100, 256), [(0, 0)])

    def test_stitch_inverts_tiling_with_overlap(self):
        data = np.random.default_rng(0).random((300, 280, 4)).astype(np.float32)
        positions = imops.tile_positions(300, 280, 128, stride=96)
        stitched = imops.stitch_tiles(imops.extract_tiles(data, positions, 128), positions, (300, 280))
        np.testing.assert_allclose(stitched, data, rtol=1e-6)

    def test_pad_to_multiple_and_fit_to_square(self):
        img = np.random.default_rng(1).integers(0, 255, (452, 300, 3), dtype=np.uint8)
        padded = imops.pad_to_multiple(img, 256)
        self.assertEqual(padded.shape, (512, 512, 3))
        np.testing.assert_array_equal(padded[:452, :300], img)
        self.assertEqual(imops.fit_to_square(img, 224).shape, (224, 224, 3))
        self.assertEqual(imops.fit_to_square(img[:10, :5], 224).shape, (224, 224, 3))

    def test_optical_density_of_background_is_zero(self):
        img = np.full((8, 8, 3), 240, dtype=np.uint8)
        img[:2, :2] = 60
        i0 = imops.background_intensity(img, imops.background_mask_auto(img))
        od = imops.optical_density(img, i0)
        self.assertAlmostEqual(float(od[5, 5].max()), 0.0, places=2)
        self.assertGreater(float(od[0, 0].min()), 0.5)
        np.testing.assert_allclose(imops.od_background(np.full(3, 255.0)), 0.0, atol=1e-7)

    def test_background_masks(self):
        img = np.full((20, 20, 3), 250, dtype=np.uint8)
        img[5:15, 5:15] = (120, 60, 140)
        for method in ('auto', 'Otsu'):
            mask_i0, _ = imops.background_masks(img, method)
            self.assertTrue(mask_i0[0, 0])
            self.assertFalse(mask_i0[10, 10])
        with self.assertRaises(ValueError):
            imops.background_masks(img, 'unknown')

    def test_parse_patch_mode(self):
        self.assertEqual(imops.parse_patch_mode('256', 224), ('grid', 256))
        self.assertEqual(imops.parse_patch_mode('single', 224), ('single', 224))
        with self.assertRaises(ValueError):
            imops.parse_patch_mode('-1', 224)

    def test_collect_image_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, 'sub'))
            for name in ('b.png', 'a.jpg', 'sub/c.PNG'):
                Image.new('RGB', (4, 4)).save(os.path.join(tmp, name))
            Path(tmp, 'note.txt').write_text('x')
            files = imops.collect_image_files(tmp)
            self.assertEqual([p.relative_to(tmp).as_posix() for p in files], ['a.jpg', 'b.png', 'sub/c.PNG'])
            self.assertEqual(len(imops.collect_image_files(tmp, selection_mode='random')), 3)
            self.assertEqual(imops.relative_path(files[2], tmp).as_posix(), 'sub/c.PNG')


class StainEstimationTest(unittest.TestCase):
    def test_recovers_known_stain_matrix(self):
        true = PAP_STAIN_ANCHOR.T  # (3, 4)
        conc = np.random.default_rng(2).random((4, 2000))
        estimated = estimate_patch_stain_matrix(conc, true @ conc)
        np.testing.assert_allclose(estimated, true, atol=1e-3)

    def test_report_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'overall_stain_matrix.txt')
            write_stain_matrix_report(path, PAP_STAIN_ANCHOR, ['test'])
            np.testing.assert_allclose(read_stain_matrix(path), PAP_STAIN_ANCHOR.T, atol=1e-6)

    def test_outlier_helpers(self):
        self.assertEqual(list(mahalanobis_inliers(np.ones((1, 3)), 0.95)), [0])
        data = np.vstack([np.random.default_rng(3).normal(0, 0.01, (30, 3)), [[5.0, 5.0, 5.0]]])
        self.assertNotIn(30, mahalanobis_inliers(data, 0.95))
        angles = spectral_angle(np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]), np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(angles, [0.0, np.pi / 2], atol=1e-6)


class ScalingAndLossTest(unittest.TestCase):
    def test_scaler_round_trip(self):
        scaler = ConcentrationScaler([0.5, 2.0, 1.0, 0.0])
        conc = np.random.default_rng(4).random((5, 5, 4))
        restored = ConcentrationScaler.from_dict(scaler.to_dict())
        np.testing.assert_allclose(restored.inverse_scale_concentrations(restored.scale_concentrations(conc))[..., :3],
                                   conc[..., :3])

    def test_statistics(self):
        values = np.array([[0.0, 1.0], [0.5, 2.0], [1.0, 3.0]])
        means, stds, factors = normal_statistics(values.sum(0), (values ** 2).sum(0), 3)
        np.testing.assert_allclose(means, [0.5, 2.0])
        np.testing.assert_allclose(factors, means + 2 * values.std(0))
        np.testing.assert_allclose(c99_per_channel(np.linspace(0, 1, 101)[:, None], 0.5), [1.0])

    def test_pixel_stats_and_background_report(self):
        stats = PixelStats()
        stats.update(np.ones((10, 4)))
        with tempfile.TemporaryDirectory() as tmp:
            stats.write_report(os.path.join(tmp, 'scaler.txt'), 'title', 0.01, False)
            write_background_report([('a.png', np.full(3, 255.0), np.zeros(3))], os.path.join(tmp, 'bg.txt'), 0.5)
            self.assertIn('EY\t1.00000000', Path(tmp, 'scaler.txt').read_text())
            self.assertIn('a.png', Path(tmp, 'bg.txt').read_text())

    def test_loss_uses_scaled_stain_matrix(self):
        conc = torch.rand(2, 4, 8, 8)
        stain = torch.rand(2, 3, 4)
        factors = [0.5, 1.0, 2.0, 4.0]
        target = reconstruct_od(conc * torch.tensor(factors).view(1, 4, 1, 1), stain)
        loss = UnmixingLoss(scaler=ConcentrationScaler(factors))
        self.assertAlmostEqual(loss(conc, stain, target)['total_loss'].item(), 0.0, places=10)
        self.assertAlmostEqual(loss.display_losses(conc, stain, target)['reconstruction_loss'].item(), 0.0, places=10)


class ModelTest(unittest.TestCase):
    def test_unet2_output(self):
        conc = UNet2().eval()(torch.rand(1, 3, 64, 64))
        self.assertEqual(tuple(conc.shape), (1, 4, 64, 64))
        self.assertTrue(bool((conc >= 0).all()))

    def test_unicas_output_and_frozen_encoder(self):
        model = UniCASStainUnmixing().eval()
        with torch.no_grad():
            conc = model(torch.rand(1, 3, 256, 256) * 0.3)
        self.assertEqual(tuple(conc.shape), (1, 4, 256, 256))
        self.assertTrue(bool((conc >= 0).all()))
        self.assertFalse(any(p.requires_grad for p in model.encoder.parameters()))
        model.train()
        self.assertFalse(model.encoder.backbone.training)


if __name__ == '__main__':
    unittest.main()
