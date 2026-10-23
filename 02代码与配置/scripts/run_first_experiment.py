"""Run a small PatchCore reference experiment on the development split.

This is a pipeline check using anomalib, not an independently implemented model.
The held-out test subset is not evaluated. See docs/FIRST_RUN.md for limitations.
"""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[2] / "03原始数据/解压数据/MVTecAD")
    parser.add_argument("--category", default="metal_nut")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2] / "04实验结果/03早期运行与原理核验/first_run")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    category_dir = args.data_root / args.category
    for relative in ("train/good", "test/good", "ground_truth"):
        if not (category_dir / relative).is_dir():
            parser.error(f"Missing {category_dir / relative}. Extract the dataset first; see docs/FIRST_RUN.md")

    import torch
    from lightning.pytorch import seed_everything
    from anomalib.data import MVTecAD
    from anomalib.engine import Engine
    from anomalib.models import Patchcore

    engine_class = Engine
    if os.name == "nt":
        class WindowsEngine(Engine):
            """Use a real output directory without Windows symlink privileges."""

            def _setup_workspace(self, *workspace_args, **workspace_kwargs):
                # anomalib 2.2.0 creates a `latest` symlink for versioned runs.
                # Non-versioned workspaces use a regular `latest` directory.
                workspace_kwargs["versioned_dir"] = False
                return super()._setup_workspace(*workspace_args, **workspace_kwargs)

        engine_class = WindowsEngine

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Check the PyTorch install instructions in docs/FIRST_RUN.md")
    seed_everything(0, workers=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    data = MVTecAD(
        root=args.data_root,
        category=args.category,
        train_batch_size=args.batch_size,
        eval_batch_size=1,
        num_workers=0,
        val_split_mode="from_test",
        val_split_ratio=0.4,
        seed=0,
    )
    model = Patchcore(
        backbone="resnet18",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.01,
        pre_processor=Patchcore.configure_pre_processor(image_size=(256, 256)),
    )
    engine = engine_class(
        accelerator="gpu",
        devices=1,
        max_epochs=1,
        logger=False,
        default_root_dir=args.output,
    )
    engine.fit(model=model, datamodule=data)
    # Reuse DEVELOPMENT data for the pipeline check. Do not evaluate held-out data.
    results = engine.test(model=model, dataloaders=data.val_dataloader())

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, dataset in (("train", data.train_data), ("development", data.val_data), ("held_out", data.test_data)):
        manifest[name] = [Path(p).resolve().relative_to(args.data_root.resolve()).as_posix()
                          for p in dataset.samples.image_path.tolist()]
    (args.output / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = {
        "purpose": "pipeline_check_on_development_data_not_final_evaluation",
        "backbone": "resnet18",
        "image_size": [256, 256],
        "coreset_sampling_ratio": 0.01,
        "seed": 0,
        "metrics": results,
        "checkpoint": engine.best_model_path,
    }
    (args.output / "development_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Done. Results: {args.output.resolve()}", flush=True)
    print(f"Checkpoint: {engine.best_model_path}", flush=True)


if __name__ == "__main__":
    main()
