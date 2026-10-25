import unittest
import numpy as np
from scripts.select_guarded_threshold import best_constrained_cutoff
from scripts.matched_recall_metrics import counts_at, sorted_scores
from scripts.threshold_scan_metrics import pixel_metrics


class GuardedThresholdTests(unittest.TestCase):
    def test_exact_constrained_search_matches_exhaustive_thresholds(self):
        rng = np.random.default_rng(24)
        values = np.round(rng.normal(size=250), 1).astype(np.float32)
        gt = rng.random(250) < .3
        pos, neg = sorted_scores(values[gt]), sorted_scores(values[~gt])
        minimum_tp = int(np.ceil(.8*len(pos)))
        maximum_threshold = float(np.nextafter(pos[4], np.float32(-np.inf)))
        best = best_constrained_cutoff(pos, neg, minimum_tp, maximum_threshold)
        candidates = np.unique(np.r_[values, np.nextafter(values.min(), np.float32(-np.inf))])
        legal = [pixel_metrics(counts_at(pos, neg, float(t)))['Pixel_IoU'] for t in candidates
                 if t <= maximum_threshold and counts_at(pos, neg, float(t))['TP'] >= minimum_tp]
        self.assertEqual(best['Pixel_IoU'], max(legal))
        self.assertLessEqual(best['threshold'], maximum_threshold)


if __name__ == '__main__':
    unittest.main()
