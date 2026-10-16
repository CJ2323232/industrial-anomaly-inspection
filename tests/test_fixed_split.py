"""Regression checks for leakage, provenance and mask integrity."""

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from scripts.prepare_fixed_split import build_manifest, resolve_file


class FixedSplitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        train = [f"metal_nut/train/good/{i:03d}.png" for i in range(10)]
        development = ["metal_nut/test/good/000.png", "metal_nut/test/scratch/000.png"]
        held_out = ["metal_nut/test/good/001.png", "metal_nut/test/scratch/001.png"]
        self.previous = {"train": train, "development": development, "held_out": held_out}
        for index, relative in enumerate(train + development + held_out):
            path = self.data / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), (index, 30, 50)).save(path)
        for index in range(2):
            path = self.data / f"metal_nut/ground_truth/scratch/{index:03d}_mask.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            mask = Image.new("L", (8, 8), 0)
            mask.putpixel((0, 0), 255)
            mask.save(path)
        self.source = self.root / "previous.json"
        self.save_previous()

    def save_previous(self):
        self.source.write_text(json.dumps(self.previous), encoding="utf-8")

    def test_preserves_seen_and_unseen_membership_and_is_deterministic(self):
        manifest, report = build_manifest(self.data, self.source)
        repeated, _ = build_manifest(self.data, self.source)
        self.assertEqual(manifest, repeated)
        self.assertEqual(len(manifest["splits"]["fit"]), 8)
        self.assertEqual(len(manifest["splits"]["normal_calibration"]), 2)
        self.assertEqual(manifest["splits"]["development"], sorted(self.previous["development"]))
        self.assertEqual(manifest["splits"]["held_out"], sorted(self.previous["held_out"]))
        flattened = [p for paths in manifest["splits"].values() for p in paths]
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertTrue(all(report["checks"].values()))

    def test_rejects_cross_split_path_leakage(self):
        self.previous["held_out"].append(self.previous["development"][0])
        self.save_previous()
        with self.assertRaisesRegex(ValueError, "Repeated image paths"):
            build_manifest(self.data, self.source)

    def test_rejects_same_image_with_different_name(self):
        source = self.data / self.previous["train"][0]
        target = self.data / self.previous["development"][0]
        target.write_bytes(source.read_bytes())
        with self.assertRaisesRegex(ValueError, "Exact image duplicates"):
            build_manifest(self.data, self.source)

    def test_rejects_missing_assignment(self):
        self.previous["held_out"].pop()
        self.save_previous()
        with self.assertRaisesRegex(ValueError, "exactly cover"):
            build_manifest(self.data, self.source)

    def test_rejects_misaligned_mask(self):
        path = self.data / "metal_nut/ground_truth/scratch/000_mask.png"
        Image.new("L", (7, 8), 255).save(path)
        with self.assertRaisesRegex(ValueError, "Mask shape/mode mismatch"):
            build_manifest(self.data, self.source)

    def test_rejects_empty_anomaly_mask(self):
        path = self.data / "metal_nut/ground_truth/scratch/000_mask.png"
        Image.new("L", (8, 8), 0).save(path)
        with self.assertRaisesRegex(ValueError, "nonempty binary"):
            build_manifest(self.data, self.source)

    def test_rejects_path_escape(self):
        with self.assertRaisesRegex(ValueError, "Unsafe relative path"):
            resolve_file(self.data.resolve(), "../previous.json")


if __name__ == "__main__":
    unittest.main()
