"""Synthetic counting and pairing checks; no dataset images are opened."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from resolution_metrics import evaluate_records, paired_localization


def record(path, tp, fp, fn, tn, *, label=1, prediction=1, defect="scratch", ap=0.7):
    return {"image_path": path, "label": label, "pred_label": prediction,
            "defect_type": defect, "area_ratio": (tp + fn) / (tp + fp + fn + tn),
            "raw_score": 14.0 if label else 12.0,
            "pixel_counts": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
            "pixel_AP": ap if label else None, "pixel_AUROC": 0.9 if label else None}


class ResolutionMetricsTests(unittest.TestCase):
    def test_exact_area_boundaries_and_normal_exclusion(self):
        rows = [record("tiny", 1, 0, 0, 1999), record("lower", 1, 0, 0, 999),
                record("upper", 1, 0, 0, 99),
                record("normal", 0, 10, 0, 90, label=0, prediction=1, defect="good")]
        before = deepcopy(rows)
        result = evaluate_records(rows)
        self.assertEqual([group["n"] for group in result["area_groups"].values()], [1, 1, 1])
        self.assertEqual(result["image_counts"], {"TP": 3, "FP": 1, "FN": 0, "TN": 0})
        self.assertEqual(result["normal_false_positive_pixels"],
                         {"images": 1, "FP": 10, "total_pixels": 100, "rate": 0.1})
        self.assertEqual(result["per_defect"]["good"]["macro"]["n"], 0)
        self.assertIsNone(result["per_defect"]["good"]["macro"]["pixel_AP"])
        self.assertEqual(rows, before)
        json.dumps(result, allow_nan=False)

    def test_macro_is_not_pooled_and_normal_pixels_do_not_inflate_localization(self):
        # Tiny defect localized perfectly; larger defect completely missed.
        rows = [record("small", 1, 0, 0, 99), record("large", 0, 0, 99, 1, prediction=0),
                record("normal", 0, 0, 0, 10000, label=0, prediction=0, defect="good")]
        result = evaluate_records(rows)
        self.assertEqual(result["anomaly_macro"]["pixel_IoU"], 0.5)
        self.assertEqual(result["anomaly_macro"]["pixel_F1"], 0.5)
        self.assertEqual(result["pixel_metrics"]["IoU"], 0.01)
        self.assertEqual(result["pixel_counts"]["TN"], 10100)
        self.assertEqual(result["image_counts"], {"TP": 1, "FP": 0, "FN": 1, "TN": 1})

    def test_empty_bins_and_empty_input_have_null_rates(self):
        result = evaluate_records([record("large", 1, 0, 0, 99)])
        empty = result["area_groups"]["lt_0.1_percent"]
        self.assertEqual(empty["n"], 0)
        self.assertIsNone(empty["macro"]["pixel_IoU"])
        self.assertIsNone(empty["pixel_metrics"]["IoU"])
        all_empty = evaluate_records([])
        self.assertEqual(all_empty["total_images"], 0)
        self.assertIsNone(all_empty["normal_false_positive_pixels"]["rate"])
        paired = paired_localization([], [])
        self.assertIsNone(paired["overall"]["delta_macro"]["pixel_IoU"])
        self.assertEqual(paired["overall"]["pixel_IoU_change_counts"]["equal"], 0)
        json.dumps(paired, allow_nan=False)

    def test_undefined_metric_has_explicit_valid_count(self):
        full_mask = record("full", 8, 0, 2, 0, ap=0.8)
        full_mask["pixel_AUROC"] = None
        result = evaluate_records([full_mask, record("mixed", 1, 0, 1, 98, ap=0.6)])
        self.assertEqual(result["anomaly_macro"]["n"], 2)
        self.assertEqual(result["anomaly_macro"]["pixel_AUROC_count"], 1)
        self.assertEqual(result["anomaly_macro"]["pixel_AUROC"], 0.9)
        self.assertEqual(result["anomaly_macro"]["pixel_AP_count"], 2)
        self.assertAlmostEqual(result["anomaly_macro"]["pixel_AP"], 0.7)

    def test_pairing_uses_paths_not_input_order_and_reports_signed_changes(self):
        baseline = [record("a", 1, 0, 1, 98), record("b", 2, 0, 0, 98),
                    record("c", 0, 0, 2, 98),
                    record("normal", 0, 1, 0, 99, label=0, prediction=1, defect="good")]
        candidate = [record("normal", 0, 0, 0, 100, label=0, prediction=0, defect="good"),
                     record("c", 0, 0, 2, 98), record("b", 1, 0, 1, 98),
                     record("a", 2, 0, 0, 98)]
        result = paired_localization(baseline, candidate)
        changes = result["overall"]["pixel_IoU_change_counts"]
        self.assertEqual(changes, {"improved": 1, "worse": 1, "equal": 1})
        self.assertEqual(result["overall"]["delta_macro"]["pixel_IoU"], 0)
        self.assertEqual(result["image_prediction_changes"], 1)
        by_path = {row["image_path"]: row for row in result["per_image"]}
        self.assertEqual(by_path["a"]["delta_pixel_IoU"], 0.5)
        self.assertEqual(by_path["b"]["delta_pixel_IoU"], -0.5)
        self.assertIsNone(by_path["normal"]["delta_pixel_IoU"])
        self.assertIsNone(by_path["normal"]["area_group"])
        self.assertTrue(any("n=0" in text for text in result["small_sample_limits"]))
        self.assertTrue(any("n=3" in text for text in result["small_sample_limits"]))

    def test_invalid_record_counts_scores_and_duplicate_paths_rejected(self):
        base = record("a", 1, 0, 1, 98)
        for change in ({"raw_score": float("nan")}, {"area_ratio": 0.3},
                       {"pixel_AP": 1.1}, {"defect_type": "good"}, {"label": 2},
                       {"pixel_counts": {"TP": -1, "FP": 0, "FN": 3, "TN": 98}},
                       {"pixel_counts": {"TP": 1.0, "FP": 0, "FN": 1, "TN": 98}},
                       {"pixel_counts": {"TP": 0, "FP": 0, "FN": 0, "TN": 0}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                evaluate_records([{**base, **change}])
        with self.assertRaises(ValueError):
            evaluate_records([base, deepcopy(base)])
        normal = record("n", 0, 0, 0, 100, label=0, prediction=0, defect="good")
        normal["pixel_AP"] = 0.0
        with self.assertRaises(ValueError):
            evaluate_records([normal])

    def test_unaligned_pairs_and_different_metric_availability_rejected(self):
        base = record("a", 1, 0, 1, 98)
        candidates = [[record("b", 1, 0, 1, 98)],
                      [record("a", 0, 0, 0, 100, label=0, defect="good")],
                      [record("a", 1, 0, 1, 98, defect="bent")],
                      [record("a", 1, 0, 2, 97)],
                      [record("a", 2, 0, 2, 196)],
                      [{**base, "pixel_AP": None}], [base, deepcopy(base)]]
        for candidate in candidates:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                paired_localization([base], candidate)


if __name__ == "__main__":
    unittest.main()
