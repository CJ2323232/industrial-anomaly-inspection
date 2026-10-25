import unittest
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from scripts.threshold_scan_metrics import pixel_metrics, rank_metrics, cutoff_at_count, oracle_image_iou
from scripts.matched_recall_metrics import counts_at, sorted_scores


class ThresholdScanTests(unittest.TestCase):
    def test_area_denominators_are_distinct(self):
        r = pixel_metrics({'TP': 6, 'FP': 2, 'FN': 4, 'TN': 8})
        self.assertEqual(r['Pixel_Recall'], .6)
        self.assertEqual(r['Pixel_IoU'], .5)
        self.assertEqual(r['Pixel_Precision'], .75)
        self.assertEqual(r['FP_area_fraction'], .1)
        self.assertEqual(r['normal_pixel_FPR'], .2)
        self.assertEqual(r['predicted_contamination'], .25)

    def test_undefined_zero_predictions_and_empty_ground_truth(self):
        r = pixel_metrics({'TP': 0, 'FP': 0, 'FN': 0, 'TN': 10})
        self.assertIsNone(r['Pixel_Recall'])
        self.assertIsNone(r['Pixel_IoU'])
        self.assertIsNone(r['Pixel_Precision'])
        self.assertEqual(r['FP_area_fraction'], 0)

    def test_rank_metrics_match_sklearn_with_ties(self):
        rng = np.random.default_rng(12)
        labels = rng.random(1000) < .3
        values = np.round(rng.normal(size=1000), 1).astype(np.float32)
        r = rank_metrics(sorted_scores(values[labels]), sorted_scores(values[~labels]))
        self.assertAlmostEqual(r['AP'], average_precision_score(labels, values), places=13)
        self.assertAlmostEqual(r['AUROC'], roc_auc_score(labels, values), places=13)

    def test_all_scores_tied(self):
        r = rank_metrics(np.ones(2, dtype=np.float32), np.ones(6, dtype=np.float32))
        self.assertEqual(r, {'AP': .25, 'AUROC': .5})

    def test_single_class_rank_metrics(self):
        self.assertEqual(rank_metrics(np.array([], dtype=np.float32), np.ones(3, dtype=np.float32)), {'AP': None, 'AUROC': None})
        self.assertEqual(rank_metrics(np.ones(3, dtype=np.float32), np.array([], dtype=np.float32)), {'AP': 1., 'AUROC': None})

    def test_recall_endpoints_are_reachable(self):
        pos = np.array([2, 3], dtype=np.float32)
        neg = np.array([1, 4], dtype=np.float32)
        self.assertEqual(counts_at(pos, neg, cutoff_at_count(pos, neg, 0)), {'TP': 0, 'FP': 0, 'FN': 2, 'TN': 2})
        self.assertEqual(counts_at(pos, neg, cutoff_at_count(pos, neg, 2))['TP'], 2)

    def test_oracle_matches_exhaustive_unique_thresholds(self):
        rng = np.random.default_rng(18)
        scores = np.round(rng.normal(size=150), 1).astype(np.float32)
        labels = rng.random(150) < .2
        pos, neg = sorted_scores(scores[labels]), sorted_scores(scores[~labels])
        candidates = np.unique(np.r_[scores, np.nextafter(scores.min(), np.float32(-np.inf))])
        exhaustive = [pixel_metrics(counts_at(pos, neg, float(t)))['Pixel_IoU'] for t in candidates]
        oracle = oracle_image_iou(pos, neg)
        self.assertEqual(oracle['max_IoU'], max(exhaustive))
        self.assertEqual(pixel_metrics(counts_at(pos, neg, oracle['threshold']))['Pixel_IoU'], oracle['max_IoU'])


if __name__ == '__main__':
    unittest.main()
