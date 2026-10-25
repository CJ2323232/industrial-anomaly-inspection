import unittest

import numpy as np

from scripts.matched_recall_metrics import sorted_scores, cutoff_for_tp, counts_at


class MatchedRecallTests(unittest.TestCase):
    def test_strict_cutoff_and_highest_float32_boundary(self):
        pos = sorted_scores(np.array([1, 2, 3, 4], dtype=np.float32))
        neg = sorted_scores(np.array([1, 2, 2.5, 4], dtype=np.float32))
        threshold = cutoff_for_tp(pos, 2)
        self.assertEqual(counts_at(pos, neg, threshold), {'TP': 2, 'FP': 1, 'FN': 2, 'TN': 3})
        self.assertEqual(counts_at(pos, neg, float(np.nextafter(np.float32(threshold), np.float32(np.inf))))['TP'], 1)

    def test_ties_are_included_without_interpolation(self):
        pos = sorted_scores(np.array([1, 2, 2, 2, 3], dtype=np.float32))
        threshold = cutoff_for_tp(pos, 2)
        self.assertEqual(counts_at(pos, np.array([], dtype=np.float32), threshold)['TP'], 4)
        self.assertEqual(counts_at(pos, np.array([], dtype=np.float32), 2)['TP'], 1)

    def test_all_tied_and_full_recall(self):
        pos = sorted_scores(np.ones(4, dtype=np.float32))
        self.assertEqual(counts_at(pos, pos, cutoff_for_tp(pos, 1)), {'TP': 4, 'FP': 4, 'FN': 0, 'TN': 0})

    def test_empty_classes(self):
        empty = np.array([], dtype=np.float32)
        self.assertEqual(counts_at(empty, empty, 0), {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0})
        with self.assertRaises(ValueError):
            cutoff_for_tp(empty, 1)

    def test_invalid_scores_and_targets(self):
        for values in (np.array([1.]), np.array([np.nan], dtype=np.float32), np.ones((1, 1), dtype=np.float32)):
            with self.assertRaises(ValueError):
                sorted_scores(values)
        for k in (0, 3, 1.5, True):
            with self.assertRaises(ValueError):
                cutoff_for_tp(np.array([1, 2], dtype=np.float32), k)

    def test_exact_counts_against_direct_masks(self):
        rng = np.random.default_rng(7)
        values = np.round(rng.normal(size=900), 1).astype(np.float32)
        labels = rng.random(900) < .3
        pos, neg = sorted_scores(values[labels]), sorted_scores(values[~labels])
        previous_fp = -1
        for k in range(1, len(pos) + 1):
            threshold = cutoff_for_tp(pos, k)
            c = counts_at(pos, neg, threshold)
            pred = values > threshold
            self.assertEqual(c, {'TP': int((pred & labels).sum()), 'FP': int((pred & ~labels).sum()),
                                 'FN': int((~pred & labels).sum()), 'TN': int((~pred & ~labels).sum())})
            self.assertGreaterEqual(c['TP'], k)
            self.assertGreaterEqual(c['FP'], previous_fp)
            previous_fp = c['FP']


if __name__ == '__main__':
    unittest.main()
