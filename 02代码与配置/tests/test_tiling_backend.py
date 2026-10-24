import unittest

import numpy as np
import torch

from scripts.tiling_backend import OverlapTiler, Prediction, sampled_patch_indices, starts, windows


class IdentityModel:
    memory_bank = torch.empty(1)

    def __call__(self, image):
        return Prediction(image[:, 0].amax().reshape(1), image[:, :1])


class TilingTests(unittest.TestCase):
    def test_full_coverage_and_exact_edge_anchors(self):
        self.assertEqual(starts(700), [0, 224, 444])
        coverage = np.zeros((700, 700), dtype=int)
        for y, x, yy, xx in windows(700, 700):
            coverage[y:yy, x:xx] += 1
        self.assertEqual(len(windows(700, 700)), 9)
        self.assertGreater(coverage.min(), 0)
        self.assertEqual(coverage.max(), 4)
        self.assertEqual(starts(256), [0])

    def test_reconstruction_original_coordinates_borders_and_seams(self):
        # A two-dimensional nonuniform ramp catches axis swaps and edge shifts.
        ramp = torch.arange(700 * 700, dtype=torch.float32).reshape(700, 700) / 490000
        image = ramp[None, None].repeat(1, 3, 1, 1)
        tiler = OverlapTiler(IdentityModel())
        with torch.inference_mode():
            result = tiler(image)
        torch.testing.assert_close(result.anomaly_map[0, 0], ramp, atol=2e-7, rtol=1e-6)
        self.assertEqual(float(result.pred_score), float(ramp.max()))

    def test_overlap_is_weighted_blend_not_sum(self):
        class TileConstant(IdentityModel):
            def __call__(self, image):
                value = image[0, 0, 0, 0]
                return Prediction(value.reshape(1), torch.ones_like(image[:, :1]) * value)
        image = torch.arange(700, dtype=torch.float32)[None, None, None].expand(1, 3, 700, 700)
        tiler = OverlapTiler(TileConstant())
        result = tiler(image)
        self.assertEqual(float(result.anomaly_map[0, 0, 100, 100]), 0)
        self.assertAlmostEqual(float(result.anomaly_map[0, 0, 100, 699]), 444, places=4)
        self.assertTrue(0 < float(result.anomaly_map[0, 0, 100, 240]) < 224)
        self.assertEqual(float(result.pred_score), 444)

    def test_sampling_reproducible_balanced_distinct(self):
        first = sampled_patch_indices(9, seed=42)
        self.assertEqual(first, sampled_patch_indices(9, seed=42))
        self.assertNotEqual(first, sampled_patch_indices(9, seed=43))
        self.assertEqual(sum(map(len, first)), 1024)
        self.assertLessEqual(max(map(len, first)) - min(map(len, first)), 1)
        for indices in first:
            self.assertEqual(len(indices), len(set(indices)))
            self.assertTrue(all(0 <= i < 1024 for i in indices))

    def test_reject_bad_geometry_and_shape(self):
        for arguments in ((255, 256, 224), (700, 256, 257), (700, 0, 1), (700, 256, 0)):
            with self.assertRaises(ValueError):
                starts(*arguments)
        with self.assertRaises(ValueError):
            OverlapTiler(IdentityModel())(torch.zeros(1, 3, 256, 256))
        with self.assertRaises(ValueError):
            sampled_patch_indices(9, budget=8)


if __name__ == '__main__':
    unittest.main()
