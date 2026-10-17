"""Check that threshold exploration exposes the FP/FN trade-off."""

import unittest

import numpy as np

from scripts.analyze_thresholds import compare_thresholds


class ThresholdAnalysisTests(unittest.TestCase):
    def test_changes_decisions_without_changing_scores_and_exposes_misses(self):
        calibration = np.array([0.0, 1.0, 2.0, 3.0])
        development = np.array([2.90, 2.94, 2.95])
        before = development.copy()
        rows = compare_thresholds(calibration, [0, 0, 0, 0], development, [0, 0, 1])
        self.assertEqual([(row["FP"], row["FN"]) for row in rows], [(2, 0), (1, 0), (0, 1)])
        self.assertAlmostEqual(rows[0]["image_threshold"], 2.85)
        np.testing.assert_array_equal(development, before)

    def test_rejects_anomalous_calibration_scores(self):
        with self.assertRaisesRegex(ValueError, "Anomalous labels"):
            compare_thresholds([1, 9], [0, 1], [2, 3], [0, 1])


if __name__ == "__main__":
    unittest.main()
