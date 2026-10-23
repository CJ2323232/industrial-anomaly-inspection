"""Lesson 2: one fit image, blocked exact neighbors and dense reference check.

Use the saved backbone and bank. No fitting, classifier threshold, weighted
image score, full-resolution anomaly map or held-out evaluation is performed.
"""
import argparse
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
import json
from pathlib import Path

if __package__:
    from .baseline_utils import file_sha256, safe_file
    from .inspect_feature_flow import check_inputs
else:
    from baseline_utils import file_sha256, safe_file
    from inspect_feature_flow import check_inputs


def plot_grid(path, image_path, grid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    with Image.open(image_path) as image:
        axes[0].imshow(image)
    axes[0].set_title("Normal fit image (implementation check)")
    axes[0].axis("off")
    rendered = axes[1].imshow(grid, cmap="viridis", interpolation="nearest")
    axes[1].set_title("32 x 32 raw nearest-neighbor distances")
    axes[1].set_xlabel("Feature-grid column")
    axes[1].set_ylabel("Feature-grid row")
    fig.colorbar(rendered, ax=axes[1], label="Euclidean distance (not probability)")
    fig.suptitle("Relative colors on this image; no defect threshold or mask applied", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--query-chunk", type=int, default=128)
    parser.add_argument("--bank-chunk", type=int, default=256)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if min(args.query_chunk, args.bank_chunk) < 1:
        parser.error("Chunk sizes must be positive")
    baseline = args.baseline_dir.resolve()
    manifest, report, relative = check_inputs(baseline, args.data_root)
    if version("anomalib") != report["environment"]["anomalib"]:
        raise ValueError("Use the anomalib version recorded in the baseline")
    identities = {name: file_sha256(baseline / name) for name in ("model.pt", "calibration.json", "report.json")}
    if args.check_only:
        print(f"FIT_IMAGE: {relative}")
        print("NEIGHBOR_INPUT_CHECK_PASSED | no image forward, search or output written")
        return
    if args.output_root is None:
        parser.error("--output-root is required for an actual lesson")
    output_root = args.output_root.resolve()
    for protected in (baseline, args.data_root.resolve()):
        if output_root == protected or output_root.is_relative_to(protected):
            raise ValueError("Learning outputs must be outside baseline and dataset directories")

    import torch
    if __package__:
        from .feature_steps import organize_features
        from .nearest_neighbors import exact_nearest_neighbors
        from .run_trusted_baseline import load_checkpoint, load_tensor, make_transform, state_sha256
    else:
        from feature_steps import organize_features
        from nearest_neighbors import exact_nearest_neighbors
        from run_trusted_baseline import load_checkpoint, load_tensor, make_transform, state_sha256
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; this lesson also supports --device cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    # A hand-computable illustration, separate from the real 384D features.
    toy_q = torch.tensor([[0., 4.]], dtype=torch.float64)
    toy_bank = torch.tensor([[0., 0.], [3., 4.], [9., 9.]], dtype=torch.float64)
    toy_d, toy_i = exact_nearest_neighbors(toy_q, toy_bank, 1, 2)
    assert toy_d.item() == 3. and toy_i.item() == 1
    print("[1/3] TOY: query=(0,4), bank=[(0,0),(3,4),(9,9)]", flush=True)
    print("      nearest=(3,4), distance=3, bank index=1 (zero-based)", flush=True)

    model, saved = load_checkpoint(baseline / "model.pt", torch.device(args.device))
    model.requires_grad_(False)
    if saved["manifest_sha256"] != manifest["manifest_sha256"] or saved["model_config"] != report["config"]:
        raise ValueError("Checkpoint metadata mismatch")
    if state_sha256(model.feature_extractor.state_dict()) != report["frozen_backbone_state_sha256"]:
        raise ValueError("Backbone identity mismatch")
    if saved["model_config"]["layers"] != ["layer2", "layer3"] or saved["model_config"]["image_size"] != [256, 256]:
        raise ValueError("Lesson requires the recorded two-layer, 256-pixel configuration")
    if list(model.memory_bank.shape) != report["memory_bank_shape"]:
        raise ValueError("Memory-bank shape mismatch")
    image = load_tensor(args.data_root, relative, make_transform(saved["model_config"])).unsqueeze(0).to(args.device)
    print(f"[2/3] FEATURES: {relative}", flush=True)
    with torch.inference_mode():
        raw = model.feature_extractor(image)
        features = organize_features(raw["layer2"], raw["layer3"])
        reference_features = model.reshape_embedding(model.generate_embedding(
            {name: model.feature_pooler(value) for name, value in raw.items()}))
        torch.testing.assert_close(features["patch_vectors"], reference_features, atol=1e-6, rtol=1e-6)
        # Promote the SAME float32 vectors to float64 solely for a numerical
        # implementation comparison. This does not regenerate backbone features.
        queries, bank = features["patch_vectors"].double(), model.memory_bank.double()
        if list(queries.shape) != [1024, 384]:
            raise ValueError("Unexpected patch-vector shape")
        print(f"      queries={list(queries.shape)}, bank={list(bank.shape)}", flush=True)
        print("[3/3] SEARCH: blocked direct Euclidean vs dense anomalib distance formula, both float64", flush=True)
        distances, indices = exact_nearest_neighbors(queries, bank, args.query_chunk, args.bank_chunk)
        reference_matrix = model.euclidean_dist(queries, bank)
        reference_distances, reference_indices = reference_matrix.min(dim=1)
        # Norm expansion in the reference can yield a tiny positive distance
        # for an identical vector; compare distances, not mandatory index identity.
        atol, rtol = 1e-6, 1e-7
        torch.testing.assert_close(distances, reference_distances, atol=atol, rtol=rtol)
        chosen_reference = reference_matrix.gather(1, indices[:, None]).squeeze(1)
        torch.testing.assert_close(chosen_reference, reference_distances, atol=atol, rtol=rtol)
        reference_choice_direct = torch.linalg.vector_norm(queries - bank[reference_indices], dim=1)
        torch.testing.assert_close(distances, reference_choice_direct, atol=atol, rtol=rtol)
        max_error = float((distances - reference_distances).abs().max().item())
        index_matches = int((indices == reference_indices).sum().item())
        height, width = features["fused_map"].shape[-2:]
        values, locations = distances.cpu().tolist(), indices.cpu().tolist()
    if state_sha256(model.feature_extractor.state_dict()) != report["frozen_backbone_state_sha256"]:
        raise ValueError("Frozen backbone changed")
    if any(file_sha256(baseline / name) != digest for name, digest in identities.items()):
        raise ValueError("Original baseline files changed")
    count_q, count_b = queries.shape[0], bank.shape[0]
    dense_bytes = count_q * count_b * queries.element_size()
    block_bytes = min(count_q, args.query_chunk) * min(count_b, args.bank_chunk) * queries.element_size()
    rows = [{"patch_index": i, "grid_row": i // width, "grid_column": i % width,
             "nearest_bank_index": locations[i], "distance": value} for i, value in enumerate(values)]
    farthest = sorted(rows, key=lambda row: row["distance"], reverse=True)[:5]
    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output = output_root / f"nearest_neighbors_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    result = {"purpose": "single_fit_image_exact_neighbor_implementation_check_not_detection_evaluation",
              "image_path": relative, "image_sha256": manifest["files"][relative]["image_sha256"],
              "manifest_sha256": manifest["manifest_sha256"], "checkpoint_sha256": identities["model.pt"],
              "device": args.device, "backbone_feature_dtype": "float32", "comparison_dtype": "float64",
              "environment": {"torch": str(torch.__version__), "anomalib": version("anomalib"),
                              "cuda": torch.version.cuda},
              "query_shape": list(queries.shape), "bank_shape": list(bank.shape),
              "query_chunk": args.query_chunk, "bank_chunk": args.bank_chunk,
              "reference": "anomalib PatchcoreModel.euclidean_dist on identical vectors promoted to float64",
              "reference_max_abs_difference": max_error, "comparison_atol": atol, "comparison_rtol": rtol,
              "reference_index_matches": index_matches, "queries_compared": count_q,
              "distance_and_selected_neighbor_checks_passed": True,
              "dense_distance_matrix_bytes": dense_bytes, "maximum_block_matrix_bytes": block_bytes,
              "memory_values_are_measured_peak_vram": False,
              "patch_neighbors": rows, "five_largest_distances": farthest,
              "held_out_images_opened": 0, "new_model_fitted": False, "threshold_applied": False,
              "source_sha256": {name: file_sha256(Path(__file__).with_name(name)) for name in
                                ("inspect_nearest_neighbors.py", "nearest_neighbors.py", "feature_steps.py",
                                 "inspect_feature_flow.py", "run_trusted_baseline.py", "baseline_utils.py")},
              "limits": ["One fit image, unsuitable for measuring generalization or detection accuracy.",
                         "Dense reference is allocated during this comparison; byte counts describe individual matrices, not total VRAM or speed.",
                         "No weighted image score or original-resolution anomaly map; do not compare patch distances to the image threshold 13.57558.",
                         "Bank row indices do not identify source-image locations; bank provenance was not stored.",
                         "Exact exhaustive search in floating point; equivalent near ties may produce different reference indices."]}
    (output / "nearest_neighbors.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [f"FIT_IMAGE: {relative}", f"query vectors: {list(queries.shape)}", f"normal bank: {list(bank.shape)}",
             "Distance comparison dtype: float64 (same stored feature values)",
             f"Distance checks passed; maximum absolute reference difference: {max_error:.10g}",
             f"Identical bank indices: {index_matches}/{count_q}; equivalent near ties are allowed",
             f"Nearest distances min/mean/max: {min(values):.6f} / {sum(values)/len(values):.6f} / {max(values):.6f}",
             f"Dense distance matrix: {dense_bytes} bytes; largest search block: {block_bytes} bytes",
             "These are matrix storage estimates, not measured peak GPU memory or speed.",
             "Largest five nearest distances (feature-grid coordinates, zero-based):"]
    lines += [f"  row={r['grid_row']}, column={r['grid_column']}, nearest bank row={r['nearest_bank_index']}, distance={r['distance']:.6f}" for r in farthest]
    lines += ["This is a normal fit image. Brighter grid cells are not confirmed defects.",
              "No threshold or weighted image score is applied; 13.576 is not a patch-distance cutoff."]
    (output / "nearest_neighbors.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot_grid(output / "distance_grid.png", safe_file(args.data_root, relative), distances.reshape(height, width).cpu().numpy())
    print("\n".join(lines), flush=True)
    print(f"NEAREST_NEIGHBOR_LESSON_COMPLETED | {output}", flush=True)


if __name__ == "__main__":
    main()
