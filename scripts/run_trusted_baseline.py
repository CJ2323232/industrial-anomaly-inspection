"""Run a manifest-controlled anomalib reference baseline on development data.

Fit on normal fit images, calibrate on separate normal images, then evaluate
development images at original resolution. Held-out images are never opened.
This remains an anomalib reference, not an independent PatchCore implementation.
"""

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timedelta, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image

if __package__:
    from .baseline_utils import (area_group, binary_counts, binary_metrics, file_sha256,
                                 load_manifest, normal_threshold, safe_file, summarize_counts,
                                 verify_consumed_files)
else:
    from baseline_utils import (area_group, binary_counts, binary_metrics, file_sha256,
                                load_manifest, normal_threshold, safe_file, summarize_counts,
                                verify_consumed_files)


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def state_sha256(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def make_model(config, pretrained):
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel

    return PatchcoreModel(backbone=config["backbone"], layers=config["layers"],
                          pre_trained=pretrained, num_neighbors=config["num_neighbors"])


def make_transform(config):
    import torch
    from torchvision.transforms import v2

    return v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                       v2.Resize(tuple(config["image_size"]), antialias=True),
                       v2.Normalize(mean=config["mean"], std=config["std"])])


def load_tensor(data_root, relative, transform):
    with Image.open(safe_file(data_root, relative)) as image:
        if image.mode != "RGB":
            raise ValueError(f"Unexpected non-RGB image: {relative}")
        return transform(image)


def predict(model, image, size):
    import torch
    from torch.nn import functional as F

    with torch.no_grad():
        result = model(image.unsqueeze(0))
        original_map = F.interpolate(result.anomaly_map, size=size, mode="bilinear", align_corners=False)
    score = float(result.pred_score[0].item())
    array = original_map[0, 0].detach().cpu().numpy().astype(np.float32)
    if not np.isfinite(score) or not np.all(np.isfinite(array)):
        raise ValueError("Non-finite raw prediction")
    return score, array


def load_checkpoint(path, device):
    import torch

    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = make_model(saved["model_config"], pretrained=False)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    return model.to(device).eval(), saved


def load_mask(data_root, record):
    if record["mask_path"] is None:
        return np.zeros((record["height"], record["width"]), dtype=np.uint8)
    with Image.open(safe_file(data_root, record["mask_path"])) as image:
        array = np.asarray(image)
    if array.shape != (record["height"], record["width"]) or not np.all((array == 0) | (array == 255)):
        raise ValueError("Mask is not binary at original image resolution")
    return (array > 0).astype(np.uint8)


def save_visualization(path, data_root, relative, mask, score_map, row, thresholds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with Image.open(safe_file(data_root, relative)) as image:
        original = np.asarray(image)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    axes[0].imshow(original)
    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[2].imshow(original)
    axes[2].imshow(score_map, cmap="turbo", alpha=0.55, vmin=0,
                   vmax=max(2 * thresholds["pixel_threshold"], 1e-6))
    axes[3].imshow(score_map > thresholds["pixel_threshold"], cmap="gray", vmin=0, vmax=1)
    for axis, title in zip(axes, ("Original", "Ground truth", "Raw anomaly map", "Predicted mask")):
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(f"{relative} | score={row['raw_score']:.4f} > threshold={thresholds['image_threshold']:.4f} "
                 f"is {bool(row['pred_label'])}", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run(args):
    import torch

    if version("anomalib") != "2.2.0":
        raise ValueError("This reference runner is verified against anomalib 2.2.0 only")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in the selected Python environment")
    if not 0 < args.coreset_ratio <= 1 or args.batch_size < 1:
        raise ValueError("Invalid coreset ratio or fitting batch size")
    manifest = load_manifest(args.manifest)
    verified = verify_consumed_files(manifest, args.data_root)
    from huggingface_hub import hf_hub_download
    pretrained_file = Path(hf_hub_download("timm/resnet18.a1_in1k", "model.safetensors", local_files_only=True))
    config = {
        "implementation": "anomalib-2.2.0-PatchcoreModel",
        "backbone": "resnet18.a1_in1k", "pretrained_source": "timm/resnet18.a1_in1k",
        "layers": ["layer2", "layer3"], "num_neighbors": 9,
        "image_size": [256, 256], "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
        "resize_antialias": True, "center_crop": None,
        "reference_map_upsampling": "anomalib 2.2.0 default nearest followed by sigma=4 Gaussian",
        "original_coordinate_mapping": "bilinear, align_corners=False",
        "coreset_ratio": args.coreset_ratio, "fit_batch_size": args.batch_size,
        "eval_batch_size": 1, "seed": args.seed, "image_quantile": 0.95, "pixel_quantile": 0.995,
        "quantile_method": "linear", "decision_rule": "raw_score > threshold",
        "pixel_calibration": "pooled original-resolution normal calibration pixels",
        "score_normalization": "none; retain raw distances, not probabilities",
    }
    preflight = {"status": "preflight_passed_not_trained", "checked_at": now(), **verified,
                 "manifest_sha256": manifest["manifest_sha256"],
                 "split_counts": {name: len(paths) for name, paths in manifest["splits"].items()},
                 "pretrained_sha256": file_sha256(pretrained_file), "gpu": torch.cuda.get_device_name(0)}
    print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
    if args.check_only:
        return
    if args.output is None:
        raise ValueError("--output is required for an actual run")
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output
    started_at, started = now(), time.perf_counter()
    write_json(output / "preflight.json", preflight)
    write_json(output / "config.json", config)
    write_json(output / "split_manifest.json", manifest)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda:0")
    transform = make_transform(config)
    model = make_model(config, pretrained=True).to(device)
    model.requires_grad_(False)
    backbone_before = state_sha256(model.feature_extractor.state_dict())
    model.train()
    model.feature_extractor.eval()
    torch.cuda.reset_peak_memory_stats(device)
    fit_paths = manifest["splits"]["fit"]
    print(f"[1/3] FIT: {len(fit_paths)} normal images; calibration/development excluded", flush=True)
    candidate_count = 0
    with torch.no_grad():
        for start in range(0, len(fit_paths), args.batch_size):
            paths = fit_paths[start:start + args.batch_size]
            batch = torch.stack([load_tensor(args.data_root, path, transform) for path in paths]).to(device)
            embedding = model(batch)
            candidate_count += embedding.shape[0]
            if start % (args.batch_size * 11) == 0 or start + len(paths) == len(fit_paths):
                print(f"  features {start + len(paths)}/{len(fit_paths)}", flush=True)
        model.subsample_embedding(args.coreset_ratio)
    model.eval()
    if model.memory_bank.ndim != 2 or model.memory_bank.shape[0] < 1 or not torch.isfinite(model.memory_bank).all():
        raise ValueError("Invalid fitted memory bank")
    backbone_after = state_sha256(model.feature_extractor.state_dict())
    if backbone_before != backbone_after:
        raise ValueError("Frozen backbone parameters or batch-normalization statistics changed")
    print(f"  memory bank: {list(model.memory_bank.shape)}", flush=True)

    def infer_and_save(relative, split):
        record = manifest["files"][relative]
        image = load_tensor(args.data_root, relative, transform).to(device)
        score, score_map = predict(model, image, (record["height"], record["width"]))
        map_relative = Path("maps") / split / Path(relative).with_suffix(".npz")
        map_path = output / map_relative
        map_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(map_path, anomaly_map=score_map)
        row = {"image_path": relative, "image_sha256": record["image_sha256"], "label": record["label"],
               "defect_type": record["defect_type"], "raw_score": score, "map_path": map_relative.as_posix()}
        return row, score_map

    print("[2/3] CALIBRATE: normal calibration images only", flush=True)
    cal_paths = manifest["splits"]["normal_calibration"]
    cal_pixels = np.empty(sum(manifest["files"][p]["height"] * manifest["files"][p]["width"] for p in cal_paths), dtype=np.float32)
    offset, calibration_rows = 0, []
    for index, relative in enumerate(cal_paths):
        row, score_map = infer_and_save(relative, "normal_calibration")
        count = score_map.size
        cal_pixels[offset:offset + count] = score_map.reshape(-1)
        offset += count
        calibration_rows.append(row)
        if (index + 1) % 11 == 0 or index + 1 == len(cal_paths):
            print(f"  calibration {index + 1}/{len(cal_paths)}", flush=True)
    image_threshold = normal_threshold([row["raw_score"] for row in calibration_rows],
                                       [row["label"] for row in calibration_rows], config["image_quantile"])
    pixel_threshold = float(np.quantile(cal_pixels, config["pixel_quantile"], method="linear"))
    thresholds = {"frozen_at": now(), "normal_image_count": len(cal_paths), "normal_pixel_count": int(cal_pixels.size),
                  "image_threshold": image_threshold, "pixel_threshold": pixel_threshold,
                  "image_quantile": config["image_quantile"], "pixel_quantile": config["pixel_quantile"],
                  "method": "linear", "decision_rule": config["decision_rule"],
                  "normal_calibration_image_exceedances": sum(row["raw_score"] > image_threshold for row in calibration_rows),
                  "normal_calibration_pixel_exceedance_fraction": float(np.count_nonzero(cal_pixels > pixel_threshold) / cal_pixels.size),
                  "limitations": "Empirical quantiles on 44 correlated product images do not guarantee future false-positive rates."}
    del cal_pixels
    write_json(output / "calibration.json", thresholds)
    calibration_sha = file_sha256(output / "calibration.json")
    for row in calibration_rows:
        row["pred_label"] = int(row["raw_score"] > image_threshold)
    write_csv(output / "normal_calibration_predictions.csv", calibration_rows)
    checkpoint = {"format_version": 1, "model_config": config,
                  "model_state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
                  "manifest_sha256": manifest["manifest_sha256"], "thresholds": thresholds}
    torch.save(checkpoint, output / "model.pt")
    del checkpoint

    print("[3/3] DEVELOPMENT: raw scores, fixed thresholds, original coordinates; held-out excluded", flush=True)
    dev_paths = manifest["splits"]["development"]
    pixel_count = sum(manifest["files"][p]["height"] * manifest["files"][p]["width"] for p in dev_paths)
    pixel_scores, pixel_labels = np.empty(pixel_count, dtype=np.float32), np.empty(pixel_count, dtype=np.uint8)
    offset, development_rows, pixel_totals = 0, [], {key: 0 for key in ("TP", "FP", "FN", "TN")}
    for index, relative in enumerate(dev_paths):
        record = manifest["files"][relative]
        row, score_map = infer_and_save(relative, "development")
        mask = load_mask(args.data_root, record)
        count = score_map.size
        pixel_scores[offset:offset + count], pixel_labels[offset:offset + count] = score_map.reshape(-1), mask.reshape(-1)
        offset += count
        counts = binary_counts(mask, score_map, pixel_threshold)
        for key in pixel_totals:
            pixel_totals[key] += counts[key]
        row.update({"pred_label": int(row["raw_score"] > image_threshold),
                    "mask_sha256": record["mask_sha256"], "area_ratio": float(mask.mean()),
                    "area_group": area_group(float(mask.mean())) if record["label"] else "normal",
                    "pixel_IoU": summarize_counts(counts)["IoU"], "pixel_F1": summarize_counts(counts)["F1"],
                    "pixel_FP": counts["FP"], "pixel_FN": counts["FN"]})
        development_rows.append(row)
        save_visualization(output / "visualizations" / relative, args.data_root, relative, mask, score_map, row, thresholds)
        if (index + 1) % 10 == 0 or index + 1 == len(dev_paths):
            print(f"  development {index + 1}/{len(dev_paths)}", flush=True)
    write_csv(output / "development_predictions.csv", development_rows)
    labels = [row["label"] for row in development_rows]
    scores = [row["raw_score"] for row in development_rows]
    image_metrics = binary_metrics(labels, scores, image_threshold)
    print("  computing exact original-resolution pixel AUROC/AP (CPU)", flush=True)
    pixel_metrics = binary_metrics(pixel_labels, pixel_scores, pixel_threshold)
    if any(pixel_metrics[key] != pixel_totals[key] for key in pixel_totals):
        raise ValueError("Per-image and pooled pixel counts disagree")
    del pixel_labels, pixel_scores
    grouped = defaultdict(list)
    for row in development_rows:
        grouped[f"defect/{row['defect_type']}"].append(row)
        if row["label"]:
            grouped[f"area/{row['area_group']}"].append(row)
    group_metrics = {name: binary_metrics([r["label"] for r in rows], [r["raw_score"] for r in rows], image_threshold)
                     for name, rows in grouped.items()}
    errors = [{"image_path": row["image_path"], "type": "false_positive" if row["label"] == 0 else "false_negative",
               "raw_score": row["raw_score"]} for row in development_rows if row["pred_label"] != row["label"]]
    localization_cases = [{key: row[key] for key in ("image_path", "defect_type", "area_ratio", "pixel_IoU", "pixel_F1")}
                          for row in development_rows if row["label"]]
    localization_cases.sort(key=lambda row: row["pixel_IoU"] if row["pixel_IoU"] is not None else -1)
    write_json(output / "error_cases.json", {"image_errors": errors, "lowest_localization_IoU": localization_cases[:10]})

    sample = dev_paths[0]
    record = manifest["files"][sample]
    image = load_tensor(args.data_root, sample, transform).to(device)
    before_score, before_map = predict(model, image, (record["height"], record["width"]))
    fit_peak = int(torch.cuda.max_memory_allocated(device))
    reloaded, saved = load_checkpoint(output / "model.pt", device)
    after_score, after_map = predict(reloaded, image, (record["height"], record["width"]))
    if not np.isclose(before_score, after_score, rtol=1e-6, atol=1e-6) or not np.allclose(before_map, after_map, rtol=1e-6, atol=1e-6):
        raise ValueError("Checkpoint reload predictions disagree")
    if saved["manifest_sha256"] != manifest["manifest_sha256"] or file_sha256(output / "calibration.json") != calibration_sha:
        raise ValueError("Frozen manifest or thresholds changed during evaluation")
    report = {
        "status": "completed_development_reference_baseline_not_final_test", "started_at": started_at, "completed_at": now(),
        "duration_seconds": time.perf_counter() - started, "duration_is_inference_latency": False,
        "manifest_sha256": manifest["manifest_sha256"], "calibration_file_sha256": calibration_sha,
        "config": config, "split_counts": preflight["split_counts"], "held_out_images_opened": 0,
        "candidate_feature_count": int(candidate_count), "memory_bank_shape": list(model.memory_bank.shape),
        "memory_bank_bytes": model.memory_bank.numel() * model.memory_bank.element_size(),
        "frozen_backbone_state_sha256": backbone_before, "backbone_unchanged": True,
        "pretrained_sha256": preflight["pretrained_sha256"],
        "image_metrics": image_metrics, "pixel_metrics": pixel_metrics, "group_metrics": group_metrics,
        "thresholds": thresholds, "image_errors": errors,
        "checkpoint": {"filename": "model.pt", "sha256": file_sha256(output / "model.pt"),
                       "size_bytes": (output / "model.pt").stat().st_size, "reload_verified": True,
                       "reload_sample": sample, "score_abs_difference": abs(before_score - after_score),
                       "map_max_abs_difference": float(np.max(np.abs(before_map - after_map)))},
        "environment": {"python": os.sys.version.split()[0], "torch": str(torch.__version__), "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(0), "anomalib": version("anomalib"),
                        "numpy": version("numpy"), "sklearn": version("scikit-learn"),
                        "torch_peak_allocated_bytes_before_reload": fit_peak},
        "source_sha256": {Path(__file__).name: file_sha256(Path(__file__)),
                          "baseline_utils.py": file_sha256(Path(__file__).with_name("baseline_utils.py"))},
        "limitations": ["Development is used for inspection and later selection; it is not an independent final test.",
                        "Only 8 normal and 37 anomalous development images; rates are uncertain at this sample size.",
                        "Custom data split and reduced ResNet18/1% configuration; not original-paper reproduction.",
                        "Normal-only empirical pixel quantiles treat pixels as pooled observations, not independent trials.",
                        "One category and one algorithm seed; no external data or formal latency benchmark yet.",
                        "Peak CUDA allocation is a PyTorch run statistic, not total VRAM or system memory."],
    }
    write_json(output / "report.json", report)
    print(json.dumps({"status": report["status"], "image_metrics": image_metrics,
                      "pixel_metrics": pixel_metrics, "held_out_images_opened": 0}, indent=2), flush=True)
    print(f"BASELINE_COMPLETED | {output.resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/MVTecAD"))
    parser.add_argument("--manifest", type=Path, default=Path("configs/splits/metal_nut-v1.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--coreset-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check-only", action="store_true", help="Verify inputs and cached weights without fitting or scoring")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
