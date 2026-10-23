"""Check calibration isolation, raw-score metrics and checkpoint correctness."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.baseline_utils import (area_group, binary_metrics, load_manifest, normal_threshold,
                                    verify_consumed_files)


class BaselineStatisticsTests(unittest.TestCase):
    def test_metrics_use_continuous_scores(self):
        result = binary_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], 0.5)
        self.assertEqual([result[k] for k in ("TP", "FP", "FN", "TN")], [1, 0, 1, 2])
        self.assertAlmostEqual(result["AUROC"], 0.75)
        self.assertAlmostEqual(result["AP"], 5 / 6)

    def test_threshold_ties_and_tied_auc(self):
        result = binary_metrics([0, 1], [0.5, 0.5], 0.5)
        self.assertEqual((result["FP"], result["FN"]), (0, 1))
        self.assertEqual(result["AUROC"], 0.5)

    def test_all_normal_does_not_produce_valid_auc(self):
        result = binary_metrics([0, 0], [0.1, 0.2], 0.3)
        self.assertIsNone(result["AUROC"])
        self.assertIsNone(result["AP"])
        self.assertIsNone(result["recall"])

    def test_rejects_anomalous_calibration(self):
        with self.assertRaisesRegex(ValueError, "Anomalous labels"):
            normal_threshold([1.0, 9.0], [0, 1])

    def test_normal_quantile_and_invalid_scores(self):
        self.assertAlmostEqual(normal_threshold([0, 1, 2, 3], [0, 0, 0, 0]), 2.85)
        with self.assertRaises(ValueError):
            normal_threshold([0, np.nan], [0, 0])

    def test_area_boundaries(self):
        self.assertEqual(area_group(0.0009), "lt_0.1_percent")
        self.assertEqual(area_group(0.001), "0.1_to_lt_1_percent")
        self.assertEqual(area_group(0.01), "ge_1_percent")


class BaselineManifestTests(unittest.TestCase):
    def test_rejects_edited_identity_and_reassigned_calibration(self):
        source = Path(__file__).resolve().parents[1] / "configs/splits/metal_nut-v1.json"
        original = load_manifest(source)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            edited = copy.deepcopy(original)
            edited["split_seed"] = 99
            path.write_text(json.dumps(edited), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                load_manifest(path)
            edited = copy.deepcopy(original)
            development = next(p for p in edited["splits"]["development"] if "/good/" in p)
            calibration = edited["splits"]["normal_calibration"][0]
            edited["splits"]["normal_calibration"].remove(calibration)
            edited["splits"]["normal_calibration"].append(development)
            edited["splits"]["development"].remove(development)
            edited["splits"]["development"].append(calibration)
            payload = {k: v for k, v in edited.items() if k != "manifest_sha256"}
            edited["manifest_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            path.write_text(json.dumps(edited), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unexpected split/source/defect"):
                load_manifest(path)

    def test_hash_verification_rejects_changes_and_never_opens_held_out(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {"splits": {}, "files": {}}
            for split in ("fit", "normal_calibration", "development", "held_out"):
                relative = f"{split}.png"
                content = split.encode()
                manifest["splits"][split] = [relative]
                manifest["files"][relative] = {"image_sha256": hashlib.sha256(content).hexdigest(), "mask_path": None}
                if split != "held_out":
                    (root / relative).write_bytes(content)
            result = verify_consumed_files(manifest, root)
            self.assertEqual(result, {"verified_images": 3, "verified_masks": 0, "held_out_images_opened": 0})
            (root / "fit.png").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                verify_consumed_files(manifest, root)


class BaselineModelTests(unittest.TestCase):
    def test_original_coordinate_bilinear_mapping(self):
        import torch
        from types import SimpleNamespace
        from scripts.run_trusted_baseline import predict

        class TinyPrediction:
            def __call__(self, image):
                return SimpleNamespace(pred_score=torch.tensor([2.0]),
                                       anomaly_map=torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]]))

        score, result = predict(TinyPrediction(), torch.zeros(3, 2, 2), (4, 4))
        self.assertEqual(score, 2.0)
        self.assertEqual(result.shape, (4, 4))
        self.assertAlmostEqual(float(result[1, 1]), 0.75)
        self.assertEqual((float(result[0, 0]), float(result[-1, -1])), (0.0, 3.0))

    def test_random_weight_checkpoint_reload_without_training_or_download(self):
        import torch
        from scripts.run_trusted_baseline import load_checkpoint, make_model, predict

        torch.set_num_threads(2)
        torch.manual_seed(0)
        config = {"backbone": "resnet18.a1_in1k", "layers": ["layer2", "layer3"], "num_neighbors": 9}
        model = make_model(config, pretrained=False).eval()
        model.memory_bank = torch.randn(9, 384)
        image = torch.randn(3, 64, 64)
        score, score_map = predict(model, image, (80, 90))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({"model_config": config, "model_state_dict": model.state_dict()}, path)
            loaded, _ = load_checkpoint(path, "cpu")
            loaded_score, loaded_map = predict(loaded, image, (80, 90))
        self.assertEqual(score, loaded_score)
        np.testing.assert_array_equal(score_map, loaded_map)


if __name__ == "__main__":
    unittest.main()
