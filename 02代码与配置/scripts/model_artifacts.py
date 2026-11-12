"""Inspect/export/import pinned model bundles using only the Python standard library.

Bundles are directories with exactly bundle.json, model.pt, config.json.
Checksums establish identity, NOT redistribution permission. No pickle is loaded.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "02代码与配置/configs/delivery/model_artifacts.json"
PROFILES = ROOT / "02代码与配置/configs/delivery/final_demo_profiles.json"
DEFAULT_STORE = ROOT / ".cache/model-artifacts"
BUNDLE_FILES = {"bundle.json", "model.pt", "config.json"}


class ArtifactError(ValueError):
    """Actionable artifact diagnostics, safe to show in a CLI or GUI."""


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"无法读取JSON {path}: {exc}") from exc


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def entry_for(profile, catalog=None):
    table = catalog if catalog is not None else read_json(CATALOG)
    matches = [e for e in table["artifacts"].values() if profile in e["profiles"]]
    if len(matches) != 1:
        raise ArtifactError(f"没有唯一的已登记模型：{profile}")
    return matches[0]


def inspect_folder(folder, entry, bundle=False):
    folder = Path(folder)
    errors = []
    required = BUNDLE_FILES if bundle else {"model.pt", "config.json"}
    if folder.is_symlink():
        return ["模型目录不能是符号链接"]
    if not folder.is_dir():
        return [f"缺少模型目录：{folder}", "需要 model.pt 与 config.json；不能用ImageNet骨干权重代替正常特征库。"]
    if bundle:
        extra = {f.name for f in folder.iterdir()} - required
        if extra:
            errors.append("模型包存在未登记文件：" + ", ".join(sorted(extra)))
    for name in sorted(required):
        f = folder / name
        if f.is_symlink() or not f.is_file():
            errors.append(f"缺少普通文件或存在符号链接：{f}")
            continue
        if name == "bundle.json":
            if f.stat().st_size > 65536:
                errors.append("bundle.json 超过64KiB")
                continue
            try:
                manifest = read_json(f)
                expected = {"schema_version": 1, "artifact_id": entry["artifact_id"],
                            "files": entry["files"], "redistribution_granted": False}
                if manifest != expected:
                    errors.append("bundle.json 与代码内登记清单不一致；包内声明不能替换可信SHA")
            except ArtifactError as exc:
                errors.append(str(exc))
        else:
            expected = entry["files"][name]
            if f.stat().st_size != expected["bytes"]:
                errors.append(f"{name} 大小不匹配：需要 {expected['bytes']} bytes")
            elif sha256(f) != expected["sha256"]:
                errors.append(f"{name} SHA256不匹配，拒绝加载")
    return errors


def resolve_artifact(profile, policy=None, root=ROOT, store=None, catalog=None):
    root = Path(root).resolve()
    entry = entry_for(profile, catalog)
    if policy and policy["checkpoint_sha256"] != entry["files"]["model.pt"]["sha256"]:
        raise ArtifactError("演示配置与模型登记SHA不一致")
    managed = Path(store).resolve() if store is not None else root / ".cache/model-artifacts"
    imported = managed / entry["artifact_id"]
    # A present but corrupt imported artifact must not silently fall back.
    if imported.exists() or imported.is_symlink():
        errors = inspect_folder(imported, entry, bundle=True)
        if errors:
            raise ArtifactError("\n".join(errors))
        return imported / "model.pt", read_json(imported / "config.json"), "imported"
    original = (root / entry["legacy_checkpoint"]).parent
    errors = inspect_folder(original, entry)
    if errors:
        raise ArtifactError(
            f"模型未就绪 [{profile}]。\n"
            f"导入位置：{imported}\n历史位置：{original}\n" + "\n".join(errors) +
            "\n获取方式：向有权提供此模型包的持有人获取；当前没有公开权重下载链接。"
            "\n导入：python 02代码与配置/scripts/model_artifacts.py import --bundle <模型包目录>"
            "\n无需模型的替代入口：python 02代码与配置/scripts/run_public_score_summary.py"
        )
    return original / "model.pt", read_json(original / "config.json"), "legacy"


def export_bundle(profile, destination, root=ROOT, store=None, catalog=None):
    entry = entry_for(profile, catalog)
    model, _, _ = resolve_artifact(profile, root=root, store=store, catalog=catalog)
    destination = Path(destination).resolve()
    if destination.exists():
        raise ArtifactError("导出目录已经存在；请选择新目录，不覆盖现有模型包。")
    destination.mkdir(parents=True)
    # Failed export remains an invalid, diagnosable bundle; originals untouched.
    shutil.copyfile(model, destination / "model.pt")
    shutil.copyfile(model.parent / "config.json", destination / "config.json")
    manifest = {"schema_version": 1, "artifact_id": entry["artifact_id"],
                "files": entry["files"], "redistribution_granted": False}
    (destination / "bundle.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    errors = inspect_folder(destination, entry, bundle=True)
    if errors:
        raise ArtifactError("\n".join(errors))
    return destination


def import_bundle(bundle, store=DEFAULT_STORE, catalog=None):
    bundle = Path(bundle).resolve()
    manifest_path = bundle / "bundle.json"
    if not manifest_path.is_file() or manifest_path.is_symlink() or manifest_path.stat().st_size > 65536:
        raise ArtifactError("缺少合法的 bundle.json（普通文件，最多64KiB）")
    manifest = read_json(manifest_path)
    table = catalog if catalog is not None else read_json(CATALOG)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("artifact_id"), str):
        raise ArtifactError("bundle.json 必须是包含字符串artifact_id的JSON对象")
    artifact_id = manifest.get("artifact_id")
    if artifact_id not in table["artifacts"]:
        raise ArtifactError("未知artifact_id；不能通过编辑包内清单绕过登记SHA。")
    entry = table["artifacts"][artifact_id]
    if artifact_id != entry["artifact_id"] or not artifact_id.isascii() or not all(
            c.isalnum() or c in "_-" for c in artifact_id):
        raise ArtifactError("不安全的artifact_id")
    errors = inspect_folder(bundle, entry, bundle=True)
    if errors:
        raise ArtifactError("\n".join(errors))
    store = Path(store).resolve()
    destination = store / artifact_id
    if destination.exists() or destination.is_symlink():
        errors = inspect_folder(destination, entry, bundle=True)
        if errors:
            raise ArtifactError("已有导入目录不完整或被修改，拒绝覆盖：\n" + "\n".join(errors))
        return destination, False
    store.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".import-", dir=store) as staging_name:
        staging = Path(staging_name).resolve()
        assert staging.is_relative_to(store) and staging != store
        payload = staging / "payload"
        payload.mkdir()
        for name in sorted(BUNDLE_FILES):
            shutil.copyfile(bundle / name, payload / name)
        # Verify again after copying, guarding source changes during transfer.
        errors = inspect_folder(payload, entry, bundle=True)
        if errors:
            raise ArtifactError("\n".join(errors))
        if destination.exists():
            raise ArtifactError("另一个进程已创建导入目录，请重新执行检查")
        os.rename(payload, destination)
    return destination, True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list", help="列出登记模型和获取范围；不需要模型")
    doctor = sub.add_parser("check", help="核查缺失文件/大小/SHA；不加载模型")
    doctor.add_argument("--profile")
    doctor.add_argument("--store", type=Path)
    export = sub.add_parser("export", help="仅打包你有权使用的现有模型，不授予分发权")
    export.add_argument("--profile", required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--store", type=Path)
    incoming = sub.add_parser("import", help="校验后导入，不覆盖已有模型")
    incoming.add_argument("--bundle", type=Path, required=True)
    incoming.add_argument("--store", type=Path, default=DEFAULT_STORE)
    args = parser.parse_args()
    try:
        if args.command == "list":
            print(json.dumps(read_json(CATALOG), ensure_ascii=False, indent=2))
        elif args.command == "check":
            profiles = [args.profile] if args.profile else list(read_json(PROFILES)["profiles"])
            rows = []
            for profile in profiles:
                try:
                    path, _, origin = resolve_artifact(profile, store=args.store)
                    rows.append(dict(profile=profile,ready=True,checkpoint=str(path),origin=origin))
                except ArtifactError as exc:
                    rows.append(dict(profile=profile,ready=False,diagnostic=str(exc)))
            print(json.dumps(dict(model_loaded=False,checks=rows),ensure_ascii=False,indent=2))
            return 0 if all(r["ready"] for r in rows) else 2
        elif args.command == "export":
            print(export_bundle(args.profile,args.output,store=args.store))
        elif args.command == "import":
            path, created = import_bundle(args.bundle,args.store)
            print(json.dumps(dict(path=str(path),created=created,model_loaded=False),ensure_ascii=False))
        return 0
    except (ArtifactError,OSError) as exc:
        print(json.dumps(dict(error=str(exc),model_loaded=False),ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

