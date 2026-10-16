"""Manifest checks and evaluation helpers for the reference baseline."""

import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np


SPLITS = ("fit", "normal_calibration", "development", "held_out")
CONSUMED_SPLITS = SPLITS[:3]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_file(root, relative):
    root = Path(root).resolve()
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative or ":" in relative:
        raise ValueError(f"Unsafe dataset path: {relative}")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f"Missing file or path outside dataset: {relative}")
    return resolved


def load_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    content = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != manifest.get("manifest_sha256"):
        raise ValueError("Manifest identity mismatch; do not edit a frozen manifest")
    if manifest.get("schema_version") != 1 or manifest.get("category") != "metal_nut":
        raise ValueError("This baseline supports the metal_nut v1 schema")
    splits = manifest["splits"]
    if set(splits) != set(SPLITS) or any(not splits[name] for name in SPLITS):
        raise ValueError("All four nonempty splits must be explicitly assigned")
    assigned = [relative for name in SPLITS for relative in splits[name]]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(manifest["files"]):
        raise ValueError("Split overlap or unassigned manifest files")
    hashes = [manifest["files"][relative]["image_sha256"] for relative in assigned]
    if len(hashes) != len(set(hashes)):
        raise ValueError("Repeated image contents in the manifest")
    for name in SPLITS:
        for relative in splits[name]:
            record = manifest["files"][relative]
            parts = PurePosixPath(relative).parts
            expected_source = "train" if name in ("fit", "normal_calibration") else "test"
            if (len(parts) != 4 or parts[0] != "metal_nut" or parts[1] != expected_source
                    or parts[2] != record["defect_type"]):
                raise ValueError(f"Unexpected split/source/defect: {relative}")
            if record["label"] not in (0, 1) or record["label"] != int(parts[2] != "good"):
                raise ValueError(f"Label disagrees with defect directory: {relative}")
            if name in ("fit", "normal_calibration") and record["label"] != 0:
                raise ValueError("Fitting and calibration must contain only normal images")
            if record["label"] == 0 and (record["mask_path"] is not None or record["mask_sha256"] is not None):
                raise ValueError(f"Normal image must not have an anomaly mask: {relative}")
            if record["label"] == 1:
                expected_mask = f"metal_nut/ground_truth/{parts[2]}/{PurePosixPath(relative).stem}_mask.png"
                if record["mask_path"] != expected_mask or not record["mask_sha256"]:
                    raise ValueError(f"Incorrect anomaly mask identity: {relative}")
            if record["width"] < 1 or record["height"] < 1:
                raise ValueError(f"Invalid original image dimensions: {relative}")
    return manifest


def verify_consumed_files(manifest, data_root):
    """Hash only fitting, calibration and development files, never held-out data."""
    image_count = mask_count = 0
    for name in CONSUMED_SPLITS:
        for relative in manifest["splits"][name]:
            record = manifest["files"][relative]
            if file_sha256(safe_file(data_root, relative)) != record["image_sha256"]:
                raise ValueError(f"Image SHA-256 mismatch: {relative}")
            image_count += 1
            if record["mask_path"] is not None:
                if file_sha256(safe_file(data_root, record["mask_path"])) != record["mask_sha256"]:
                    raise ValueError(f"Mask SHA-256 mismatch: {record['mask_path']}")
                mask_count += 1
    return {"verified_images": image_count, "verified_masks": mask_count,
            "held_out_images_opened": 0}


def normal_threshold(scores, labels, quantile=0.95):
    values = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    if values.ndim != 1 or not values.size or values.shape != labels.shape:
        raise ValueError("Normal calibration needs aligned nonempty score and label vectors")
    if not np.all(labels == 0):
        raise ValueError("Anomalous labels cannot enter normal-only calibration")
    if not np.all(np.isfinite(values)) or not 0 < quantile < 1:
        raise ValueError("Invalid calibration scores or quantile")
    return float(np.quantile(values, quantile, method="linear"))


def binary_counts(labels, scores, threshold):
    labels, scores = np.asarray(labels).reshape(-1), np.asarray(scores).reshape(-1)
    if labels.shape != scores.shape or not labels.size:
        raise ValueError("Labels and scores must be nonempty and aligned")
    if not np.all((labels == 0) | (labels == 1)) or not np.all(np.isfinite(scores)):
        raise ValueError("Labels must be binary and scores must be finite")
    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    positive = labels == 1
    prediction = scores > threshold
    tp = int(np.count_nonzero(positive & prediction))
    fp = int(np.count_nonzero(~positive & prediction))
    fn = int(np.count_nonzero(positive & ~prediction))
    tn = int(labels.size - tp - fp - fn)
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


def summarize_counts(counts):
    tp, fp, fn, tn = (counts[key] for key in ("TP", "FP", "FN", "TN"))
    ratio = lambda numerator, denominator: numerator / denominator if denominator else None
    return {**counts, "normal_count": tn + fp, "anomaly_count": tp + fn,
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
            "F1": ratio(2 * tp, 2 * tp + fp + fn), "IoU": ratio(tp, tp + fp + fn),
            "false_positive_rate": ratio(fp, fp + tn), "false_negative_rate": ratio(fn, fn + tp)}


def binary_metrics(labels, scores, threshold):
    from sklearn.metrics import average_precision_score, roc_auc_score

    summary = summarize_counts(binary_counts(labels, scores, threshold))
    if summary["normal_count"] and summary["anomaly_count"]:
        summary["AUROC"] = float(roc_auc_score(labels, scores))
        summary["AP"] = float(average_precision_score(labels, scores))
    else:
        summary["AUROC"] = summary["AP"] = None
    return summary


def area_group(ratio):
    if not 0 < ratio <= 1:
        raise ValueError("Anomaly area ratio must be in (0, 1]")
    return "lt_0.1_percent" if ratio < 0.001 else "0.1_to_lt_1_percent" if ratio < 0.01 else "ge_1_percent"
