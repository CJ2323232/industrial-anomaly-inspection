"""Freeze metal_nut splits while preserving the evaluated and held-out images.

This script audits files and labels; it never loads a model or evaluates scores.
Run from the repository root with the first-run Python environment.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import random

from PIL import Image


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_file(root, relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative or ":" in relative:
        raise ValueError(f"Unsafe relative path: {relative}")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f"Missing file or path outside dataset: {relative}")
    return resolved


def build_manifest(data_root, previous_manifest, category="metal_nut", seed=0, calibration_ratio=0.2):
    root = data_root.resolve()
    if category != "metal_nut":
        raise ValueError("v1 only supports metal_nut; other categories need their own frozen evaluation split")
    if not 0 < calibration_ratio < 1:
        raise ValueError("calibration_ratio must be between 0 and 1")
    previous = json.loads(previous_manifest.read_text(encoding="utf-8"))
    if set(previous) != {"train", "development", "held_out"}:
        raise ValueError("Unexpected keys in the first-run manifest")
    original = [p for paths in previous.values() for p in paths]
    if len(original) != len(set(original)):
        raise ValueError("Repeated image paths in the first-run manifest")
    actual = {p.relative_to(root).as_posix() for source in ("train", "test")
              for p in (root / category / source).rglob("*.png")}
    if set(original) != actual:
        raise ValueError("First-run paths must exactly cover the current category's train and test PNG files")
    train_prefix = f"{category}/train/good/"
    test_prefix = f"{category}/test/"
    if any(not p.startswith(train_prefix) for p in previous["train"]):
        raise ValueError("Only original train/good images may enter fitting or calibration")
    if any(not p.startswith(test_prefix) for split in ("development", "held_out") for p in previous[split]):
        raise ValueError("Development and held-out images must come from original test")
    normal_paths = sorted(previous["train"])
    calibration_count = int(len(normal_paths) * calibration_ratio)
    if not 1 <= calibration_count < len(normal_paths):
        raise ValueError("The calibration ratio leaves an empty fitting or calibration set")
    shuffled = normal_paths.copy()
    random.Random(seed).shuffle(shuffled)
    calibration_paths = set(shuffled[:calibration_count])
    split_paths = {
        "fit": sorted(set(normal_paths) - calibration_paths),
        "normal_calibration": sorted(calibration_paths),
        "development": sorted(previous["development"]),
        "held_out": sorted(previous["held_out"]),
    }
    assigned = [p for paths in split_paths.values() for p in paths]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(original):
        raise ValueError("New splits must cover every original image exactly once")
    records = {}
    duplicate_groups = defaultdict(list)
    expected_masks = set()
    for relative in sorted(original):
        path = resolve_file(root, relative)
        parts = PurePosixPath(relative).parts
        if len(parts) != 4 or parts[0] != category or parts[1] not in ("train", "test"):
            raise ValueError(f"Unexpected image path: {relative}")
        defect_type = parts[2]
        label = int(defect_type != "good")
        with Image.open(path) as image:
            image.load()
            if image.mode != "RGB":
                raise ValueError(f"Expected RGB image: {relative}")
            size = image.size
        mask_relative = None
        mask_sha = None
        if label:
            mask_relative = f"{category}/ground_truth/{defect_type}/{path.stem}_mask.png"
            mask_path = resolve_file(root, mask_relative)
            with Image.open(mask_path) as mask:
                mask.load()
                if mask.size != size or mask.mode != "L":
                    raise ValueError(f"Mask shape/mode mismatch: {mask_relative}")
                histogram = mask.histogram()
                if not histogram[255] or sum(histogram[1:255]):
                    raise ValueError(f"Expected nonempty binary 0/255 mask: {mask_relative}")
            expected_masks.add(mask_relative)
            mask_sha = sha256(mask_path)
        image_sha = sha256(path)
        duplicate_groups[image_sha].append(relative)
        records[relative] = {
            "image_sha256": image_sha,
            "label": label,
            "defect_type": defect_type,
            "width": size[0],
            "height": size[1],
            "mask_path": mask_relative,
            "mask_sha256": mask_sha,
        }
    actual_masks = {p.relative_to(root).as_posix() for p in (root / category / "ground_truth").rglob("*.png")}
    if actual_masks != expected_masks:
        raise ValueError("Mask directory contains missing or unassigned PNG files")
    duplicates = [paths for paths in duplicate_groups.values() if len(paths) > 1]
    if duplicates:
        raise ValueError(f"Exact image duplicates require review before freezing: {duplicates}")
    manifest = {
        "schema_version": 1,
        "protocol": "metal_nut-v1-independent-normal-calibration",
        "category": category,
        "previous_manifest_sha256": sha256(previous_manifest),
        "split_seed": seed,
        "normal_calibration_ratio": calibration_ratio,
        "normal_calibration_rounding": "floor(n * ratio)",
        "splits": split_paths,
        "files": records,
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    report = {
        "purpose": "split_and_file_audit_only_no_model_evaluation",
        "manifest_sha256": manifest["manifest_sha256"],
        "generator_sha256": sha256(Path(__file__)),
        "image_count": len(records),
        "mask_count": len(expected_masks),
        "counts": {name: dict(sorted(Counter(records[p]["defect_type"] for p in paths).items()))
                   for name, paths in split_paths.items()},
        "checks": {
            "all_paths_disjoint": len(assigned) == len(set(assigned)),
            "original_images_covered_once": set(original) == set(assigned),
            "fit_and_calibration_normal_only": all(records[p]["label"] == 0
                for name in ("fit", "normal_calibration") for p in split_paths[name]),
            "development_membership_preserved": split_paths["development"] == sorted(previous["development"]),
            "held_out_membership_preserved": split_paths["held_out"] == sorted(previous["held_out"]),
            "images_readable_rgb": True,
            "masks_nonempty_binary_and_aligned": True,
            "no_exact_image_duplicates": not duplicates,
        },
        "limitations": [
            "Byte-level duplicate checks do not establish independence of objects, scenes or capture batches.",
            "Calibration images were used in the old 220-image smoke feature bank; rebuild using only fit paths.",
            "Development membership is historical, not newly sampled per-defect stratification.",
            "Held-out files are audited for integrity and labels; no model predictions or performance are computed.",
            "Splits are frozen; threshold calibration and manifest-consuming model runners are not implemented yet.",
        ],
    }
    return manifest, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[2] / "03原始数据/解压数据/MVTecAD")
    parser.add_argument("--previous-manifest", type=Path, default=Path(__file__).resolve().parents[2] / "05项目文档/results/2026-10-13-split_manifest.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "configs/splits/metal_nut-v1.json")
    parser.add_argument("--report", type=Path, default=Path(__file__).resolve().parents[2] / "04实验结果/03早期运行与原理核验/split_audit/metal_nut-v1.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calibration-ratio", type=float, default=0.2)
    args = parser.parse_args()
    if args.output.resolve() == args.report.resolve():
        parser.error("--output and --report must be different paths")
    try:
        manifest, report = build_manifest(args.data_root, args.previous_manifest,
                                         seed=args.seed, calibration_ratio=args.calibration_ratio)
        outputs = [(args.output, manifest), (args.report, report)]
        texts = [(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n") for path, value in outputs]
        for path, content in texts:
            if path.exists() and path.read_text(encoding="utf-8") != content:
                raise ValueError(f"Refusing to overwrite different frozen content: {path}; use a new version/path")
        for path, content in texts:
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8", newline="\n")
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps({"manifest_sha256": manifest["manifest_sha256"],
                      "split_sizes": {name: len(paths) for name, paths in manifest["splits"].items()},
                      "checks": report["checks"]}, indent=2))


if __name__ == "__main__":
    main()
