"""Validate integrated retrieval with synthetic tensors; no dataset inference."""
import unittest
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from scripts.compare_retrieval import compare_outputs
from scripts.retrieval_backend import blocked_knn, use_retrieval


class RetrievalBackendTests(unittest.TestCase):
    def test_k1_and_k9_with_uneven_chunks_match_direct_double_oracle(self):
        generator = torch.Generator().manual_seed(23)
        q = torch.randn(7, 10, generator=generator, dtype=torch.float64)[:, ::2]
        b = torch.randn(13, 10, generator=generator, dtype=torch.float64)[:, ::2]
        original_q, original_b = q.clone(), b.clone()
        oracle = torch.linalg.vector_norm(q[:, None] - b[None], dim=-1)
        for k in (1, 9, 13):
            expected, _ = oracle.topk(k, dim=1, largest=False, sorted=True)
            for chunks in ((1, 1), (3, 4), (100, 100)):
                distances, indices = blocked_knn(q, b, k, *chunks)
                if k == 1:
                    distances, indices = distances[:, None], indices[:, None]
                torch.testing.assert_close(distances, expected, atol=1e-12, rtol=1e-12)
                torch.testing.assert_close(oracle.gather(1, indices), expected, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(q, original_q, atol=0, rtol=0)
        torch.testing.assert_close(b, original_b, atol=0, rtol=0)

    def test_ties_and_single_vector_bank(self):
        q = torch.tensor([[0., 0.]])
        b = torch.tensor([[3., 4.], [1., 0.], [-1., 0.], [1., 0.]])
        d, i = blocked_knn(q, b, 1, 1, 2)
        self.assertEqual((d.item(), i.item()), (1., 1))
        d, i = blocked_knn(q, b, 3, 1, 2)
        self.assertEqual(d.tolist(), [[1., 1., 1.]])
        self.assertEqual(set(i[0].tolist()), {1, 2, 3})
        d, i = blocked_knn(q, torch.tensor([[3., 4.]]))
        self.assertEqual((d.item(), i.item()), (5., 0))

    def test_invalid_arguments(self):
        q = torch.ones(2, 3)
        for options in ({"k": 3}, {"query_chunk": 0}, {"bank_chunk": -1}, {"k": True}):
            with self.assertRaises(ValueError):
                blocked_knn(q, q, **options)
        for bank in (torch.ones(0, 3), torch.ones(2, 4), q.double(), q.long()):
            with self.assertRaises(ValueError):
                blocked_knn(q, bank)

    def test_context_restores_backend_after_exception(self):
        class FakeModel:
            memory_bank = torch.zeros(2, 3)

            def nearest_neighbors(self, embedding, n_neighbors):
                return "reference"
        model = FakeModel()
        with self.assertRaises(RuntimeError):
            with use_retrieval(model, "blocked"):
                self.assertNotEqual(model.nearest_neighbors(torch.ones(1, 3), 1), "reference")
                raise RuntimeError("intentional")
        self.assertEqual(model.nearest_neighbors(None, 1), "reference")
        self.assertNotIn("nearest_neighbors", model.__dict__)

    def test_small_numeric_change_can_cross_fixed_thresholds(self):
        result = compare_outputs(1., np.array([[4.]]), 1.00001, np.array([[4.00001]]),
                                 {"fixed": 1.000005}, 4.000005)
        self.assertTrue(result["score_close"])
        self.assertTrue(result["map_close"])
        self.assertTrue(result["image_decision_changes"]["fixed"])
        self.assertEqual(result["pixel_decision_changes"], 1)
        with self.assertRaises(ValueError):
            compare_outputs(float("nan"), np.zeros((1, 1)), 0., np.zeros((1, 1)), {"fixed": 1.}, 1.)

    def test_complete_synthetic_forward_preserves_weighting_and_maps(self):
        from anomalib.models.image.patchcore.torch_model import PatchcoreModel
        torch.manual_seed(23)
        model = PatchcoreModel(backbone="resnet18.a1_in1k", layers=["layer2", "layer3"],
                              pre_trained=False, num_neighbors=9).double().eval()
        model.requires_grad_(False)
        model.memory_bank = torch.randn(13, 384, dtype=torch.float64) * .1 + 1
        image = torch.randn(1, 3, 32, 32, dtype=torch.float64)
        before_bank = model.memory_bank.clone()
        with torch.inference_mode():
            reference_trace, blocked_trace = [], []
            with use_retrieval(model, "reference", trace=reference_trace):
                expected = model(image)
            with use_retrieval(model, "blocked", 3, 4, blocked_trace):
                actual = model(image)
        self.assertEqual([r["k"] for r in reference_trace], [1, 9])
        self.assertEqual([r["k"] for r in blocked_trace], [1, 9])
        torch.testing.assert_close(actual.pred_score, expected.pred_score, atol=1e-5, rtol=1e-6)
        torch.testing.assert_close(actual.anomaly_map, expected.anomaly_map, atol=1e-5, rtol=1e-6)
        torch.testing.assert_close(model.memory_bank, before_bank, atol=0, rtol=0)
        self.assertNotIn("nearest_neighbors", model.__dict__)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_float32_topk(self):
        q = torch.tensor([[0., 4.], [3., 0.]], device="cuda")
        b = torch.tensor([[0., 0.], [3., 4.], [9., 9.]], device="cuda")
        for k in (1, 3):
            actual, indices = blocked_knn(q, b, k, 1, 2)
            expected, _ = torch.linalg.vector_norm(q[:, None] - b[None], dim=-1).topk(k, dim=1, largest=False, sorted=True)
            torch.testing.assert_close(actual.reshape(2, k), expected)
            self.assertEqual(indices.dtype, torch.long)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_synthetic_runner_writes_correctness_timing_and_final_reports(self):
        from scripts import compare_retrieval as runner
        from scripts.baseline_utils import file_sha256
        from scripts.run_trusted_baseline import state_sha256

        class FakeDetector(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.feature_extractor = torch.nn.Identity()
                self.register_buffer("memory_bank", torch.arange(26, device="cuda", dtype=torch.float32).reshape(13, 2) / 10)

            def nearest_neighbors(self, embedding, n_neighbors):
                matrix = (embedding.pow(2).sum(1, keepdim=True) - 2 * embedding @ self.memory_bank.T
                          + self.memory_bank.pow(2).sum(1)[None]).clamp_min(0).sqrt()
                return matrix.min(dim=1)

            def forward(self, image):
                score, _ = self.nearest_neighbors(image[0, :2, 0, 0].reshape(1, 2), 1)
                return SimpleNamespace(pred_score=score, anomaly_map=score.reshape(1, 1, 1, 1).expand(1, 1, 4, 4))

        cache = Path(__file__).resolve().parents[2] / ".cache/tmp"
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="synthetic_retrieval_test_", dir=cache) as temp:
            root = Path(temp)
            baseline, dataset = root / "baseline", root / "data"
            baseline.mkdir()
            dataset.mkdir()
            identities = {}
            for name in ("model.pt", "report.json", "calibration.json"):
                (baseline / name).write_bytes(b"synthetic test fixture, not a real checkpoint")
                identities[name] = file_sha256(baseline / name)
            policy = root / "policy.json"
            policy.write_text("{}", encoding="utf-8")
            model = FakeDetector().eval()
            paths = ["metal_nut/test/good/synthetic.png", "metal_nut/test/scratch/synthetic.png"]
            records = {p: {"label": i, "height": 4, "width": 4, "image_sha256": "synthetic"} for i, p in enumerate(paths)}
            manifest = {"manifest_sha256": "synthetic", "splits": {"development": paths}, "files": records}
            tensors = {p: torch.full((3, 4, 4), float(1 + 3 * i)) for i, p in enumerate(paths)}
            old_rows = [{"image_path": p, "raw_score": float(model(tensors[p].unsqueeze(0).cuda()).pred_score.item())} for p in paths]
            report = {"config": {}, "frozen_backbone_state_sha256": state_sha256(model.feature_extractor.state_dict()),
                      "thresholds": {"image_threshold": 1., "pixel_threshold": 1.}}
            argv = ["compare_retrieval", "--baseline-dir", str(baseline), "--data-root", str(dataset),
                    "--policy", str(policy), "--output-root", str(root / "output"),
                    "--warmup", "1", "--repeats", "1", "--query-chunk", "1", "--bank-chunk", "4"]
            with patch("sys.argv", argv), patch.object(runner, "inspect_inputs", return_value=(identities, report, manifest, old_rows, 1.5)), \
                    patch("scripts.run_trusted_baseline.load_checkpoint", return_value=(model, {"manifest_sha256": "synthetic", "model_config": {}})), \
                    patch("scripts.run_trusted_baseline.make_transform", return_value=None), \
                    patch("scripts.run_trusted_baseline.load_tensor", side_effect=lambda root, path, transform: tensors[path]), \
                    patch("scripts.run_trusted_baseline.load_mask", side_effect=lambda root, record: np.full((4, 4), record["label"], dtype=np.uint8)):
                runner.main()
            output = next((root / "output").iterdir())
            result = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertTrue(result["compatibility_gate_passed"])
            self.assertEqual(result["development_images"], 2)
            self.assertEqual(result["held_out_images_opened"], 0)
            self.assertFalse(result["backend_automatically_adopted"])
            for backend in ("reference", "blocked"):
                self.assertEqual(result["performance"][backend]["samples"], 2)
                self.assertGreater(result["performance"][backend]["mean_ms"], 0)
                self.assertGreater(result["performance"][backend]["max_peak_allocated_bytes"], 0)
            self.assertEqual(len(list((output / "maps").rglob("*.npz"))), 2)
            self.assertTrue((output / "结果摘要.txt").is_file())


if __name__ == "__main__":
    unittest.main()
