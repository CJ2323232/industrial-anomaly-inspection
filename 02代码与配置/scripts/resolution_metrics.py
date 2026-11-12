"""Per-image, original-coordinate metrics for controlled resolution experiments.

Area bins use total ground-truth defect area per image, not component size.
Macro localization weights anomalous images equally; pooled metrics weight pixels
equally. Outputs are JSON serializable and empty-group rates remain undefined.
"""

import math
from numbers import Integral, Real

if __package__:
    from .baseline_utils import area_group, summarize_counts
else:
    from baseline_utils import area_group, summarize_counts

AREA_GROUPS = ("lt_0.1_percent", "0.1_to_lt_1_percent", "ge_1_percent")
COUNT_KEYS = ("TP", "FP", "FN", "TN")
LOCALIZATION_KEYS = ("pixel_IoU", "pixel_F1", "pixel_AP", "pixel_AUROC")


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _validated(rows):
    """Copy and validate records without changing caller-owned data."""
    checked, seen = [], set()
    for source in rows:
        path = source.get("image_path")
        if not isinstance(path, str) or not path or path in seen:
            raise ValueError("Every image_path must be nonempty and unique")
        seen.add(path)
        label, prediction = source.get("label"), source.get("pred_label")
        if label not in (0, 1) or prediction not in (0, 1):
            raise ValueError(f"Labels and predictions must be binary: {path}")
        defect = source.get("defect_type")
        if not isinstance(defect, str) or not defect or (defect == "good") != (label == 0):
            raise ValueError(f"defect_type is missing or disagrees with label: {path}")
        counts = source.get("pixel_counts", {})
        if set(counts) != set(COUNT_KEYS) or any(
            isinstance(counts[key], bool) or not isinstance(counts[key], Integral)
            or counts[key] < 0 for key in COUNT_KEYS
        ):
            raise ValueError(f"pixel_counts must contain four nonnegative integers: {path}")
        counts = {key: int(counts[key]) for key in COUNT_KEYS}
        total = sum(counts.values())
        if total == 0:
            raise ValueError(f"An image must contain at least one pixel: {path}")
        ratio = _finite_number(source.get("area_ratio"), "area_ratio")
        expected = (counts["TP"] + counts["FN"]) / total
        if not 0 <= ratio <= 1 or not math.isclose(ratio, expected, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"area_ratio disagrees with original-coordinate counts: {path}")
        if (ratio == 0) != (label == 0):
            raise ValueError(f"Mask area disagrees with image label: {path}")
        metrics = summarize_counts(counts)
        row = {"image_path": path, "label": int(label), "pred_label": int(prediction),
               "defect_type": defect, "area_ratio": ratio, "pixel_counts": counts,
               "raw_score": _finite_number(source.get("raw_score"), "raw_score"),
               "pixel_IoU": metrics["IoU"] if label else None,
               "pixel_F1": metrics["F1"] if label else None}
        for key in ("pixel_AP", "pixel_AUROC"):
            value = source.get(key)
            if value is not None:
                value = _finite_number(value, key)
                if not 0 <= value <= 1:
                    raise ValueError(f"{key} must lie in [0, 1]: {path}")
            if label == 0 and value is not None:
                raise ValueError(f"Normal-image {key} must be None: {path}")
            row[key] = value
        checked.append(row)
    return checked


def _macro(rows):
    anomalies = [row for row in rows if row["label"] == 1]
    result = {"n": len(anomalies)}
    for key in LOCALIZATION_KEYS:
        values = [row[key] for row in anomalies if row[key] is not None]
        result[key] = sum(values) / len(values) if values else None
        result[f"{key}_count"] = len(values)
    return result


def _summary(rows):
    pixels = {key: sum(row["pixel_counts"][key] for row in rows) for key in COUNT_KEYS}
    images = {key: 0 for key in COUNT_KEYS}
    for row in rows:
        key = ("TP" if row["pred_label"] else "FN") if row["label"] else (
            "FP" if row["pred_label"] else "TN")
        images[key] += 1
    return {"n": len(rows), "image_counts": images,
            "image_metrics": summarize_counts(images), "pixel_counts": pixels,
            "pixel_metrics": summarize_counts(pixels), "macro": _macro(rows)}


def evaluate_records(rows):
    """Summarize thresholded records, including null rates for empty groups.

    Required fields: image_path, label, defect_type, area_ratio, raw_score,
    pred_label, pixel_counts (TP/FP/FN/TN). Optional pixel_AP/pixel_AUROC are
    per-image threshold-independent values; use None when undefined.
    """
    rows = _validated(rows)
    total = _summary(rows)
    normal = [row for row in rows if row["label"] == 0]
    normal_fp = sum(row["pixel_counts"]["FP"] for row in normal)
    normal_pixels = sum(sum(row["pixel_counts"].values()) for row in normal)
    anomalies = [row for row in rows if row["label"] == 1]
    return {"total_images": len(rows), "normal_images": len(normal),
            "anomalous_images": len(anomalies),
            "image_counts": total["image_counts"], "image_metrics": total["image_metrics"],
            "pixel_counts": total["pixel_counts"], "pixel_metrics": total["pixel_metrics"],
            "anomaly_macro": total["macro"],
            "normal_false_positive_pixels": {"images": len(normal), "FP": normal_fp,
                                             "total_pixels": normal_pixels,
                                             "rate": normal_fp / normal_pixels if normal_pixels else None},
            "area_groups": {group: _summary([row for row in anomalies
                                            if area_group(row["area_ratio"]) == group])
                            for group in AREA_GROUPS},
            "per_defect": {defect: _summary([row for row in rows if row["defect_type"] == defect])
                           for defect in sorted({row["defect_type"] for row in rows})}}


def _paired_summary(pairs):
    baseline = _macro([left for left, _ in pairs])
    candidate = _macro([right for _, right in pairs])
    result = {"n": len(pairs), "baseline_macro": baseline, "candidate_macro": candidate,
              "delta_macro": {key: (candidate[key] - baseline[key]
                                     if candidate[key] is not None and baseline[key] is not None else None)
                              for key in LOCALIZATION_KEYS}}
    for key in ("pixel_IoU", "pixel_F1"):
        changes = {"improved": 0, "worse": 0, "equal": 0}
        for left, right in pairs:
            delta = right[key] - left[key]
            changes["improved" if delta > 0 else "worse" if delta < 0 else "equal"] += 1
        result[f"{key}_change_counts"] = changes
    return result


def paired_localization(rows256, rows512):
    """Align by path and report 512-minus-256 per-image localization changes.

    Requires identical paths, labels, ground-truth pixel counts, areas, defect
    types and metric availability. Input order has no effect. Raw-score changes
    are diagnostic only because score scales may differ with resolution.
    """
    left = {row["image_path"]: row for row in _validated(rows256)}
    right = {row["image_path"]: row for row in _validated(rows512)}
    if set(left) != set(right):
        raise ValueError("Paired resolutions must contain exactly the same image paths")
    pairs, per_image = [], []
    for path in sorted(left):
        first, second = left[path], right[path]
        if any(first[key] != second[key] for key in ("label", "defect_type", "area_ratio")):
            raise ValueError(f"Paired labels, defect types or areas differ: {path}")
        first_counts, second_counts = first["pixel_counts"], second["pixel_counts"]
        if (sum(first_counts.values()) != sum(second_counts.values()) or
                first_counts["TP"] + first_counts["FN"] != second_counts["TP"] + second_counts["FN"]):
            raise ValueError(f"Paired original-coordinate ground-truth sizes differ: {path}")
        if any((first[key] is None) != (second[key] is None) for key in LOCALIZATION_KEYS):
            raise ValueError(f"Paired metric availability differs: {path}")
        result = {key: first[key] for key in ("image_path", "label", "defect_type", "area_ratio")}
        result["area_group"] = area_group(first["area_ratio"]) if first["label"] else None
        result["image_prediction_changed"] = first["pred_label"] != second["pred_label"]
        for key in (*LOCALIZATION_KEYS, "raw_score"):
            result[f"{key}_256"], result[f"{key}_512"] = first[key], second[key]
            result[f"delta_{key}"] = second[key] - first[key] if first[key] is not None else None
        result["pred_label_256"], result["pred_label_512"] = first["pred_label"], second["pred_label"]
        per_image.append(result)
        if first["label"]:
            pairs.append((first, second))
    groups = {group: _paired_summary([(first, second) for first, second in pairs
                                      if area_group(first["area_ratio"]) == group])
              for group in AREA_GROUPS}
    limits = ["按整张原图中的缺陷总面积占比分组，不等同于逐个连通缺陷的尺寸分组。",
              "开发集上的成对描述性对照不能作为最终测试或泛化能力结论；原始异常分数不能跨分辨率直接比较高低。"]
    for group, summary in groups.items():
        if summary["n"] == 0:
            limits.append(f"{group}: n=0，不能评价该尺寸组的定位能力。")
        elif summary["n"] < 10:
            limits.append(f"{group}: n={summary['n']}，样本少，仅作探索性观察，不能据此宣称稳定提升。")
    return {"total_images": len(left), "anomalous_images": len(pairs),
            "image_prediction_changes": sum(row["image_prediction_changed"] for row in per_image),
            "per_image": per_image, "overall": _paired_summary(pairs),
            "area_groups": groups, "small_sample_limits": limits}
