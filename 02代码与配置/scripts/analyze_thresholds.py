"""Explore image-threshold sensitivity using saved calibration/development scores.

This is post-hoc development analysis, not a new fitted model or final test.
The original baseline, its pixel threshold and all predictions remain unchanged.
"""

import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np

if __package__:
    from .baseline_utils import binary_counts, file_sha256, load_manifest, normal_threshold, summarize_counts
else:
    from baseline_utils import binary_counts, file_sha256, load_manifest, normal_threshold, summarize_counts


QUANTILES = (0.95, 0.975, 0.99)
SOURCE_FILES = ("report.json", "calibration.json", "config.json", "split_manifest.json", "model.pt",
                "normal_calibration_predictions.csv", "development_predictions.csv")


def compare_thresholds(calibration_scores, calibration_labels, development_scores, development_labels):
    rows = []
    for quantile in QUANTILES:
        threshold = normal_threshold(calibration_scores, calibration_labels, quantile)
        counts = summarize_counts(binary_counts(development_labels, development_scores, threshold))
        rows.append({"normal_quantile": quantile, "image_threshold": threshold,
                     "calibration_exceedances": int(np.count_nonzero(np.asarray(calibration_scores) > threshold)),
                     **counts})
    return rows


def read_predictions(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["raw_score"], row["label"] = float(row["raw_score"]), int(row["label"])
    return rows


def checked_inputs(root):
    identities = {name: file_sha256(root / name) for name in SOURCE_FILES}
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    calibration = json.loads((root / "calibration.json").read_text(encoding="utf-8"))
    manifest = load_manifest(root / "split_manifest.json")
    if report["status"] != "completed_development_reference_baseline_not_final_test":
        raise ValueError("A completed development baseline is required")
    if (report["manifest_sha256"] != manifest["manifest_sha256"]
            or report["calibration_file_sha256"] != identities["calibration.json"]
            or report["checkpoint"]["sha256"] != identities["model.pt"]
            or report["held_out_images_opened"] != 0):
        raise ValueError("Baseline identities or held-out separation failed")
    normal_rows = read_predictions(root / "normal_calibration_predictions.csv")
    development_rows = read_predictions(root / "development_predictions.csv")
    for split, rows in (("normal_calibration", normal_rows), ("development", development_rows)):
        paths = [row["image_path"] for row in rows]
        if len(paths) != len(set(paths)) or set(paths) != set(manifest["splits"][split]):
            raise ValueError(f"Prediction paths do not match frozen {split} membership")
        for row in rows:
            record = manifest["files"][row["image_path"]]
            if row["label"] != record["label"] or row["image_sha256"] != record["image_sha256"]:
                raise ValueError("Prediction label or image identity differs from manifest")
    scores = [row["raw_score"] for row in normal_rows]
    labels = [row["label"] for row in normal_rows]
    reference_threshold = normal_threshold(scores, labels, 0.95)
    if not np.isclose(reference_threshold, calibration["image_threshold"], atol=1e-8, rtol=0):
        raise ValueError("The original 95% threshold cannot be reproduced from saved normal scores")
    counts = binary_counts([row["label"] for row in development_rows],
                           [row["raw_score"] for row in development_rows], reference_threshold)
    if any(counts[key] != report["image_metrics"][key] for key in counts):
        raise ValueError("Saved raw predictions do not reproduce the original confusion counts")
    return identities, report, normal_rows, development_rows


def plot_scores(path, normal_rows, development_rows, comparisons):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    normal_development = [row for row in development_rows if row["label"] == 0]
    anomalous_development = [row for row in development_rows if row["label"] == 1]
    groups = [(f"Normal calibration ({len(normal_rows)})", normal_rows),
              (f"Normal development ({len(normal_development)})", normal_development),
              (f"Anomalous development ({len(anomalous_development)})", anomalous_development)]
    fig, axis = plt.subplots(figsize=(11, 4.5))
    random = np.random.default_rng(0)
    for index, (label, rows) in enumerate(groups):
        values = [row["raw_score"] for row in rows]
        axis.scatter(values, index + random.uniform(-0.12, 0.12, len(values)), s=27, alpha=0.75,
                     color=("tab:blue", "tab:green", "tab:orange")[index])
    for row, color in zip(comparisons, ("tab:red", "tab:purple", "tab:brown")):
        axis.axvline(row["image_threshold"], color=color, linestyle="--", linewidth=1.2,
                     label=f"Normal q={row['normal_quantile']:.3f}: threshold={row['image_threshold']:.3f}")
    axis.set_yticks(range(3), [label for label, _ in groups])
    axis.set_xlabel("Raw image anomaly score (not probability)")
    axis.set_title("Image-threshold sensitivity: exploratory development analysis")
    axis.grid(axis="x", alpha=0.2)
    axis.legend(loc="upper right", fontsize=8)
    axis.set_ylim(-0.4, 2.65)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = args.baseline_dir.resolve()
    identities, report, normal_rows, development_rows = checked_inputs(root)
    if args.check_only:
        print("INPUT_CHECK_PASSED | baseline reproduced; no thresholds changed or analysis outputs written")
        return
    comparisons = compare_thresholds([row["raw_score"] for row in normal_rows], [row["label"] for row in normal_rows],
                                     [row["raw_score"] for row in development_rows], [row["label"] for row in development_rows])
    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output = (args.output_root or Path(__file__).resolve().parents[2] / "04实验结果/02阈值分析") / f"thresholds_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    with (output / "threshold_comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    plot_scores(output / "score_distribution.png", normal_rows, development_rows, comparisons)
    result = {"purpose": "post_hoc_development_image_threshold_sensitivity_not_final_evaluation",
              "baseline_run": root.name, "baseline_source_sha256": identities,
              "analysis_script_sha256": file_sha256(Path(__file__)),
              "manifest_sha256": report["manifest_sha256"], "comparisons": comparisons,
              "normal_calibration_count": len(normal_rows), "development_count": len(development_rows),
              "held_out_images_opened": 0, "new_model_fitted": False, "pixel_threshold_changed": False,
              "baseline_threshold_overwritten": False, "candidate_threshold_selected": False,
              "limitations": ["Quantile policies are explored after observing development errors.",
                              "Performance on these 45 seen development images is not a final test estimate.",
                              "Changing image thresholds does not change AUROC, raw anomaly maps or pixel predictions.",
                              "A higher threshold may reduce false positives while increasing false negatives.",
                              "The 70 held-out images are not read or scored."]}
    if any(file_sha256(root / name) != digest for name, digest in identities.items()):
        raise ValueError("Source baseline changed during analysis")
    (output / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Normal quantile | Image threshold | FP / normal | FN / anomalous")
    for row in comparisons:
        print(f"{row['normal_quantile']:.3%} | {row['image_threshold']:.6f} | "
              f"{row['FP']}/{row['normal_count']} | {row['FN']}/{row['anomaly_count']}")
    print("Exploratory development results only; no baseline/model/pixel threshold changed.")
    print(f"THRESHOLD_ANALYSIS_COMPLETED | {output}")


if __name__ == "__main__":
    main()
