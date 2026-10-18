"""Check channel/spatial meaning using hand-computable feature maps."""

import unittest
import torch

from scripts.feature_steps import organize_features


class FeatureStepsTests(unittest.TestCase):
    def test_known_pooling_values_and_channel_order(self):
        # Padded 3x3 averaging divides by 9, including padding zeros.
        fine = torch.arange(1, 10, dtype=torch.float32).reshape(1, 1, 3, 3)
        coarse = torch.full((1, 1, 1, 1), 9.0)
        rows = organize_features(fine, coarse)["patch_vectors"]
        sums = torch.tensor([12, 21, 16, 27, 45, 33, 24, 39, 28], dtype=torch.float32)
        expected = torch.stack((sums / 9, torch.ones(9)), dim=1)
        torch.testing.assert_close(rows, expected)

    def test_two_images_keep_separate_spatial_rows(self):
        fine = torch.tensor([9.0, 18.0]).reshape(2, 1, 1, 1)
        coarse = torch.tensor([27.0, 36.0]).reshape(2, 1, 1, 1)
        torch.testing.assert_close(organize_features(fine, coarse)["patch_vectors"],
                                   torch.tensor([[1.0, 3.0], [2.0, 4.0]]))


if __name__ == "__main__":
    unittest.main()
