"""Inspect one fit image's features and compare learning fusion with reference.

The reference checkpoint supplies the frozen backbone. No nearest-neighbor
scoring, fitting, threshold selection or development/final evaluation is run.
"""

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

if __package__:
    from .baseline_utils import file_sha256, load_manifest, safe_file
else:
    from baseline_utils import file_sha256, load_manifest, safe_file


def check_inputs(baseline, data_root):
    manifest = load_manifest(baseline / "split_manifest.json")
    report = json.loads((baseline / "report.json").read_text(encoding="utf-8"))
    if (report["status"] != "completed_development_reference_baseline_not_final_test"
            or manifest["manifest_sha256"] != report["manifest_sha256"]
            or report["held_out_images_opened"] != 0):
        raise ValueError("Expected the verified development reference baseline")
    if file_sha256(baseline / "model.pt") != report["checkpoint"]["sha256"]:
        raise ValueError("Checkpoint identity mismatch")
    relative = manifest["splits"]["fit"][0]
    path = safe_file(data_root, relative)
    if file_sha256(path) != manifest["files"][relative]["image_sha256"]:
        raise ValueError("Fit image identity mismatch")
    return manifest, report, relative


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    baseline = args.baseline_dir.resolve()
    manifest, report, relative = check_inputs(baseline, args.data_root)
    print(f"FIT_IMAGE: {relative}", flush=True)
    if args.check_only:
        print("FEATURE_INPUT_CHECK_PASSED | no model forward or output written")
        return
    if args.output_root is None:
        parser.error("--output-root is required for a lesson run")
    output_root = args.output_root.resolve()
    data_root = args.data_root.resolve()
    if (output_root == baseline or output_root.is_relative_to(baseline)
            or output_root == data_root or output_root.is_relative_to(data_root)):
        raise ValueError("Save learning outputs outside the baseline and dataset")

    import torch
    if __package__:
        from .feature_steps import organize_features
        from .run_trusted_baseline import load_checkpoint, load_tensor, make_transform, state_sha256
    else:
        from feature_steps import organize_features
        from run_trusted_baseline import load_checkpoint, load_tensor, make_transform, state_sha256

    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; use --device cpu for this one-image lesson")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, saved = load_checkpoint(baseline / "model.pt", torch.device(args.device))
    model.requires_grad_(False)
    if saved["manifest_sha256"] != manifest["manifest_sha256"] or saved["model_config"] != report["config"]:
        raise ValueError("Checkpoint metadata differs from report")
    config = saved["model_config"]
    if config["layers"] != ["layer2", "layer3"] or config["image_size"] != [256, 256]:
        raise ValueError("This lesson requires the recorded layer2/layer3, 256-pixel baseline")
    before = state_sha256(model.feature_extractor.state_dict())
    if before != report["frozen_backbone_state_sha256"]:
        raise ValueError("Backbone differs from baseline")
    image = load_tensor(args.data_root, relative, make_transform(config)).unsqueeze(0).to(args.device)
    with torch.inference_mode():
        raw = model.feature_extractor(image)
        learning = organize_features(raw["layer2"], raw["layer3"])
        reference_map = model.generate_embedding({name: model.feature_pooler(value) for name, value in raw.items()})
        reference_rows = model.reshape_embedding(reference_map)
        torch.testing.assert_close(learning["patch_vectors"], reference_rows, rtol=1e-6, atol=1e-6)
        max_error = float((learning["patch_vectors"] - reference_rows).abs().max().item())
    shapes = {"input": list(image.shape), **{name: list(value.shape) for name, value in raw.items()},
              **{name: list(value.shape) for name, value in learning.items()},
              "existing_memory_bank": list(model.memory_bank.shape)}
    expected = {"input": [1, 3, 256, 256], "layer2": [1, 128, 32, 32], "layer3": [1, 256, 16, 16],
                "fused_map": [1, 384, 32, 32], "patch_vectors": [1024, 384]}
    if any(shapes[key] != value for key, value in expected.items()):
        raise ValueError(f"Unexpected baseline feature dimensions: {shapes}")
    if before != state_sha256(model.feature_extractor.state_dict()):
        raise ValueError("Frozen backbone changed")
    if file_sha256(baseline / "model.pt") != report["checkpoint"]["sha256"]:
        raise ValueError("Source checkpoint changed")
    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output = output_root / f"feature_flow_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    result = {"purpose": "single_fit_image_feature_organization_learning_check",
              "image_path": relative, "image_sha256": manifest["files"][relative]["image_sha256"],
              "manifest_sha256": manifest["manifest_sha256"], "checkpoint_sha256": report["checkpoint"]["sha256"],
              "device": args.device, "shapes": shapes, "reference_max_abs_difference": max_error,
              "comparison_rtol": 1e-6, "comparison_atol": 1e-6,
              "feature_organization_matches_reference": True,
              "held_out_images_opened": 0, "new_model_fitted": False, "threshold_applied": False,
              "source_sha256": {name: file_sha256(Path(__file__).with_name(name))
                                for name in ("inspect_feature_flow.py", "feature_steps.py", "run_trusted_baseline.py", "baseline_utils.py")},
              "limits": "One input only. Backbone uses the anomalib reference loader; this is not an independent full detector."}
    (output / "feature_flow.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [f"{name}: {shape}" for name, shape in shapes.items()]
    lines += [f"Reference maximum absolute difference: {max_error}",
              "1024 = 32 x 32 spatial locations; 384 = 128 + 256 feature channels.",
              "A feature location has an overlapping receptive field, not an independent 8 x 8 crop.",
              "Memory-bank rows are retained feature vectors, not image counts."]
    (output / "feature_flow.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"FEATURE_FLOW_COMPLETED | {output}", flush=True)


if __name__ == "__main__":
    main()
