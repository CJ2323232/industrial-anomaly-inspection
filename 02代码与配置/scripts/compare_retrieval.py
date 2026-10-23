"""Compare two retrieval backends inside full, frozen PatchCore inference.

Manual development experiment only: 45 fixed development images, two fixed
image thresholds, unchanged pixel threshold, no fitting and no held-out data.
"""
import argparse
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np

if __package__:
    from .analyze_thresholds import checked_inputs
    from .baseline_utils import binary_counts, binary_metrics, file_sha256, load_manifest, normal_threshold, safe_file, summarize_counts
else:
    from analyze_thresholds import checked_inputs
    from baseline_utils import binary_counts, binary_metrics, file_sha256, load_manifest, normal_threshold, safe_file, summarize_counts


ATOL = 1e-4
RTOL = 1e-5
BACKENDS = ("reference", "blocked")


def inspect_inputs(args):
    identities, report, cal_rows, dev_rows = checked_inputs(args.baseline_dir)
    manifest = load_manifest(args.baseline_dir / "split_manifest.json")
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    candidate = normal_threshold([r["raw_score"] for r in cal_rows], [r["label"] for r in cal_rows], .975)
    if (policy["checkpoint_sha256"] != identities["model.pt"] or policy["manifest_sha256"] != manifest["manifest_sha256"]
            or policy["image_quantile"] != .975 or policy["quantile_method"] != "linear"
            or policy["image_threshold_for_this_checkpoint"] != candidate
            or policy["pixel_threshold"] != report["thresholds"]["pixel_threshold"]):
        raise ValueError("Candidate policy does not match this frozen baseline and calibration")
    if version("anomalib") != report["environment"]["anomalib"]:
        raise ValueError("Use the baseline anomalib version")
    for relative in manifest["splits"]["development"]:
        record = manifest["files"][relative]
        for key, hash_key in (("image_path", "image_sha256"), ("mask_path", "mask_sha256")):
            path = relative if key == "image_path" else record[key]
            if path is not None and file_sha256(safe_file(args.data_root, path)) != record[hash_key]:
                raise ValueError(f"Development file changed: {path}")
    return identities, report, manifest, dev_rows, candidate


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def compare_outputs(reference_score, reference_map, blocked_score, blocked_map, thresholds, pixel_threshold):
    """Keep continuous differences and threshold changes as separate evidence."""
    if not all(np.isfinite(x).all() for x in (np.asarray(reference_score), reference_map,
                                             np.asarray(blocked_score), blocked_map)):
        raise ValueError("Non-finite detector output")
    if reference_map.shape != blocked_map.shape:
        raise ValueError("Output map shape differs")
    return {
        "score_abs_difference": abs(reference_score - blocked_score),
        "score_close": bool(np.isclose(reference_score, blocked_score, atol=ATOL, rtol=RTOL)),
        "map_max_abs_difference": float(np.abs(reference_map - blocked_map).max()),
        "map_close": bool(np.allclose(reference_map, blocked_map, atol=ATOL, rtol=RTOL)),
        "image_decision_changes": {name: bool((reference_score > value) != (blocked_score > value)) for name, value in thresholds.items()},
        "pixel_decision_changes": int(np.count_nonzero((reference_map > pixel_threshold) != (blocked_map > pixel_threshold))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--query-chunk", type=int, default=128)
    parser.add_argument("--bank-chunk", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if min(args.query_chunk, args.bank_chunk, args.warmup, args.repeats) < 1:
        parser.error("Chunks, warmup and repeats must be positive")
    args.baseline_dir, args.data_root = args.baseline_dir.resolve(), args.data_root.resolve()
    identities, baseline_report, manifest, old_rows, candidate = inspect_inputs(args)
    policy_sha = file_sha256(args.policy)
    import torch
    if not torch.cuda.is_available():
        raise ValueError("This performance comparison requires CUDA")
    if args.check_only:
        print("RETRIEVAL_PREFLIGHT_PASSED | 45 development images and masks verified; frozen policy/checkpoint verified")
        print("No model forward, fitting, benchmark or held-out evaluation run.")
        return
    if args.output_root is None:
        parser.error("--output-root is required")
    output_root = args.output_root.resolve()
    if any(output_root == p or output_root.is_relative_to(p) for p in (args.data_root, args.baseline_dir)):
        raise ValueError("Write results outside frozen baseline and dataset")
    if __package__:
        from .retrieval_backend import use_retrieval
        from .run_trusted_baseline import load_checkpoint, load_tensor, make_transform, predict, load_mask, state_sha256
    else:
        from retrieval_backend import use_retrieval
        from run_trusted_baseline import load_checkpoint, load_tensor, make_transform, predict, load_mask, state_sha256
    from torch.nn import functional as F
    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output = output_root / f"retrieval_comparison_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone(timedelta(hours=8))).isoformat()
    run_config = {"status": "started_not_completed", "started_at": started,
                  "query_chunk": args.query_chunk, "bank_chunk": args.bank_chunk, "warmup_each_backend": args.warmup,
                  "repeats": args.repeats, "atol": ATOL, "rtol": RTOL, "dtype": "float32",
                  "checkpoint_sha256": identities["model.pt"], "manifest_sha256": manifest["manifest_sha256"],
                  "policy_sha256": policy_sha, "baseline_source_sha256": identities,
                  "source_sha256": {name: file_sha256(Path(__file__).with_name(name)) for name in
                                    ("compare_retrieval.py", "retrieval_backend.py", "analyze_thresholds.py", "baseline_utils.py", "run_trusted_baseline.py")},
                  "environment": {"torch": str(torch.__version__), "anomalib": version("anomalib"),
                                  "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}}
    write_json(output / "config.json", run_config)
    write_json(output / "split_manifest.json", manifest)
    print(f"OUTPUT: {output}", flush=True)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda:0")
    model, saved = load_checkpoint(args.baseline_dir / "model.pt", device)
    model.requires_grad_(False)
    if (saved["manifest_sha256"] != manifest["manifest_sha256"] or saved["model_config"] != baseline_report["config"]
            or state_sha256(model.feature_extractor.state_dict()) != baseline_report["frozen_backbone_state_sha256"]):
        raise ValueError("Loaded model identity mismatch")
    bank_before = state_sha256({"bank": model.memory_bank})
    if not torch.isfinite(model.memory_bank).all():
        raise ValueError("Invalid memory bank")
    transform = make_transform(saved["model_config"])
    paths = manifest["splits"]["development"]
    images = [load_tensor(args.data_root, relative, transform) for relative in paths]
    old_scores = {r["image_path"]: r["raw_score"] for r in old_rows}
    thresholds = {"original_q95": baseline_report["thresholds"]["image_threshold"], "candidate_q975": candidate}
    pixel_threshold = baseline_report["thresholds"]["pixel_threshold"]
    scores = {name: [] for name in BACKENDS}
    pixel_counts = {name: {key: 0 for key in ("TP", "FP", "FN", "TN")} for name in BACKENDS}
    rows = []
    print("[1/3] CORRECTNESS: compare both full pipelines on 45 fixed development images", flush=True)
    for number, (relative, cpu_image) in enumerate(zip(paths, images)):
        record = manifest["files"][relative]
        image = cpu_image.to(device)
        mask = load_mask(args.data_root, record)
        predictions, traces, pixel_per_image = {}, {}, {}
        for backend in BACKENDS:
            trace = []
            with use_retrieval(model, backend, args.query_chunk, args.bank_chunk, trace):
                score, score_map = predict(model, image, (record["height"], record["width"]))
            predictions[backend] = (score, score_map)
            traces[backend] = trace
            scores[backend].append(score)
            counts = binary_counts(mask, score_map, pixel_threshold)
            pixel_per_image[backend] = summarize_counts(counts)
            for key in counts:
                pixel_counts[backend][key] += counts[key]
        row = {"image_path": relative, "image_sha256": record["image_sha256"], "label": record["label"],
               "reference_score": predictions["reference"][0], "blocked_score": predictions["blocked"][0],
               "reference_matches_saved_score": bool(np.isclose(predictions["reference"][0], old_scores[relative], atol=ATOL, rtol=RTOL)),
               **compare_outputs(*predictions["reference"], *predictions["blocked"], thresholds, pixel_threshold),
               "pixel_metrics": pixel_per_image, "search_calls": []}
        if len(traces["reference"]) != len(traces["blocked"]):
            raise ValueError("Different search-call structure")
        for ref, blocked in zip(traces["reference"], traces["blocked"]):
            if ref["k"] != blocked["k"] or ref["distances"].shape != blocked["distances"].shape:
                raise ValueError("Different search dimensions")
            row["search_calls"].append({"k": ref["k"], "distance_shape": list(ref["distances"].shape),
                                        "max_abs_difference": float(np.abs(ref["distances"] - blocked["distances"]).max()),
                                        "distances_close": bool(np.allclose(ref["distances"], blocked["distances"], atol=ATOL, rtol=RTOL)),
                                        "different_indices": int(np.count_nonzero(ref["indices"] != blocked["indices"]))})
        map_path = output / "maps" / Path(relative).with_suffix(".npz")
        map_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(map_path, reference=predictions["reference"][1], blocked=predictions["blocked"][1])
        rows.append(row)
        del image, predictions, traces
        if (number + 1) % 5 == 0 or number + 1 == len(paths):
            print(f"  compared {number + 1}/{len(paths)}", flush=True)
    write_json(output / "prediction_comparison.json", rows)
    labels = [manifest["files"][p]["label"] for p in paths]
    metrics = {backend: {"image": {name: binary_metrics(labels, scores[backend], value) for name, value in thresholds.items()},
                         "pixel": summarize_counts(pixel_counts[backend])} for backend in BACKENDS}

    def forward_original(image, size):
        result = model(image)
        original = F.interpolate(result.anomaly_map, size=size, mode="bilinear", align_corners=False)
        return result.pred_score, original

    print("[2/3] WARMUP: both backends; no timing recorded", flush=True)
    with torch.inference_mode():
        for backend in BACKENDS:
            with use_retrieval(model, backend, args.query_chunk, args.bank_chunk):
                for index in range(args.warmup):
                    number = index % len(paths)
                    record = manifest["files"][paths[number]]
                    image = images[number].unsqueeze(0).to(device)
                    result = forward_original(image, (record["height"], record["width"]))
                    torch.cuda.synchronize(device)
                    del result, image
        print("[3/3] BENCHMARK: alternating backend order, CUDA synchronized, GPU-resident inputs", flush=True)
        timings = []
        for repeat in range(args.repeats):
            for number, (relative, cpu_image) in enumerate(zip(paths, images)):
                record = manifest["files"][relative]
                image = cpu_image.unsqueeze(0).to(device)
                order = BACKENDS if (repeat + number) % 2 == 0 else BACKENDS[::-1]
                for backend in order:
                    with use_retrieval(model, backend, args.query_chunk, args.bank_chunk):
                        torch.cuda.synchronize(device)
                        torch.cuda.reset_peak_memory_stats(device)
                        before_bytes = torch.cuda.memory_allocated(device)
                        start = time.perf_counter()
                        result = forward_original(image, (record["height"], record["width"]))
                        torch.cuda.synchronize(device)
                        milliseconds = (time.perf_counter() - start) * 1000
                        peak_bytes = torch.cuda.max_memory_allocated(device)
                        timings.append({"repeat": repeat, "image_path": relative, "backend": backend,
                                        "milliseconds": milliseconds, "allocated_before_bytes": before_bytes,
                                        "peak_allocated_bytes": peak_bytes, "incremental_peak_bytes": peak_bytes - before_bytes})
                        del result
                del image
            print(f"  timing repeat {repeat + 1}/{args.repeats} completed", flush=True)
    write_json(output / "timings.json", timings)
    performance = {}
    for backend in BACKENDS:
        samples = [r for r in timings if r["backend"] == backend]
        milliseconds = [r["milliseconds"] for r in samples]
        performance[backend] = {"samples": len(samples), "mean_ms": float(np.mean(milliseconds)),
                                "p95_ms": float(np.quantile(milliseconds, .95, method="linear")),
                                "max_peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in samples),
                                "max_incremental_peak_bytes": max(r["incremental_peak_bytes"] for r in samples),
                                "round_mean_ms": [float(np.mean([r["milliseconds"] for r in samples if r["repeat"] == repeat])) for repeat in range(args.repeats)]}
    correctness = {
        "reference_reproduces_saved_scores": all(r["reference_matches_saved_score"] for r in rows),
        "all_scores_close": all(r["score_close"] for r in rows), "all_maps_close": all(r["map_close"] for r in rows),
        "all_search_distances_close": all(s["distances_close"] for r in rows for s in r["search_calls"]),
        "image_decision_changes": {name: sum(r["image_decision_changes"][name] for r in rows) for name in thresholds},
        "pixel_decision_changes": sum(r["pixel_decision_changes"] for r in rows),
        "max_score_abs_difference": max(r["score_abs_difference"] for r in rows),
        "max_map_abs_difference": max(r["map_max_abs_difference"] for r in rows),
    }
    compatible = (all(correctness[k] for k in ("reference_reproduces_saved_scores", "all_scores_close", "all_maps_close", "all_search_distances_close"))
                  and not any(correctness["image_decision_changes"].values()) and correctness["pixel_decision_changes"] == 0)
    if (bank_before != state_sha256({"bank": model.memory_bank})
            or state_sha256(model.feature_extractor.state_dict()) != baseline_report["frozen_backbone_state_sha256"]
            or any(file_sha256(args.baseline_dir / name) != digest for name, digest in identities.items())
            or file_sha256(args.policy) != policy_sha):
        raise ValueError("Frozen model, baseline or policy changed")
    result = {"status": "completed_development_backend_comparison_not_final_test", "started_at": started,
              "completed_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
              "config": run_config, "correctness": correctness, "compatibility_gate_passed": compatible,
              "metrics": metrics, "performance": performance, "image_thresholds": thresholds,
              "pixel_threshold": pixel_threshold, "fixed_thresholds_for_implementation_comparison": True,
              "development_images": len(paths), "held_out_images_opened": 0, "new_model_fitted": False,
              "backend_automatically_adopted": False,
              "timing_boundary": "CUDA-resident input through frozen backbone, retrieval, weighted image score, anomaly map and original-size GPU interpolation; synchronized wall time; excludes disk IO, preprocessing, H2D, D2H, metrics and saving",
              "memory_boundary": "Measured PyTorch peak allocated bytes per call, includes resident model/input/output; not total device VRAM or process memory. Incremental peak subtracts pre-call allocation.",
              "limitations": ["One class, checkpoint and machine; only 45 seen development images.",
                              "Float32 GEMM shapes and equal-distance ties can change small distances, neighbor identities and downstream outputs.",
                              "Current bank is small; no memory bottleneck or speedup is assumed.",
                              "Performance differences cannot justify adoption when correctness checks fail.",
                              "Pixel confusion metrics and map differences are reported; pixel AUROC/AP are not recomputed."]}
    write_json(output / "report.json", result)
    summary = ["分块检索完整流程对照已完成（开发集实验）", f"严格兼容检查通过：{compatible}",
               f"两种图像阈值下的判断变化张数：{correctness['image_decision_changes']}",
               f"像素判断变化数量：{correctness['pixel_decision_changes']}"]
    for backend in BACKENDS:
        item = performance[backend]
        summary.append(f"{backend}: mean={item['mean_ms']:.3f} ms, P95={item['p95_ms']:.3f} ms, peak allocated={item['max_peak_allocated_bytes']/1024**2:.3f} MiB")
    summary += ["耗时只覆盖 GPU 输入已就绪后的完整模型前向与原尺寸还原。显存为 PyTorch 分配峰值。",
                "完整的正确性/性能结果见 report.json；尚未自动采用分块后端。",
                "兼容检查失败时先检查差异，不能把性能数字直接写成有效优化。"]
    (output / "结果摘要.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary), flush=True)
    print(f"RETRIEVAL_COMPARISON_COMPLETED | {output}", flush=True)


if __name__ == "__main__":
    main()
