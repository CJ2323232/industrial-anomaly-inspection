"""Install a pinned research model archive; no training or deserialization."""
from pathlib import Path
import argparse, hashlib, json, os, shutil, stat, tempfile, urllib.request, zipfile
try:
    from .model_artifacts import ROOT, CATALOG, DEFAULT_STORE, ArtifactError, read_json, sha256, entry_for, inspect_folder, import_bundle
except ImportError:
    from model_artifacts import ROOT, CATALOG, DEFAULT_STORE, ArtifactError, read_json, sha256, entry_for, inspect_folder, import_bundle
RELEASES = ROOT / "02代码与配置/configs/delivery/model_releases.json"
BUNDLE = {"bundle/bundle.json", "bundle/model.pt", "bundle/config.json"}

def release_for(profile, releases=None, catalog=None):
    entry = entry_for(profile, catalog)
    table = releases if releases is not None else read_json(RELEASES)
    release = table["artifacts"].get(entry["artifact_id"])
    if not release:
        raise ArtifactError("没有登记此模型的发布包")
    return entry, release

def verify_archive(path, release):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ArtifactError("请选择普通ZIP文件")
    if path.stat().st_size != release["bytes"] or sha256(path) != release["sha256"]:
        raise ArtifactError("ZIP大小或SHA256不匹配，拒绝安装")
    expected = release["files"]
    if not BUNDLE.issubset(expected):
        raise ArtifactError("登记清单缺少模型包文件")
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        names = [info.filename for info in entries]
        if len(set(names)) != len(names) or set(names) != set(expected):
            raise ArtifactError("ZIP包含重复、缺失或未登记路径")
        for info in entries:
            name = info.filename
            if name.startswith("/") or "\\" in name or ":" in name or any(x in ("", ".", "..") for x in name.split("/")):
                raise ArtifactError("ZIP路径不安全")
            if info.is_dir() or info.flag_bits & 1 or stat.S_ISLNK(info.external_attr >> 16):
                raise ArtifactError("不接受ZIP目录条目、加密文件或符号链接")
            if info.file_size != expected[name]["bytes"]:
                raise ArtifactError("ZIP解压大小不匹配："+name)
            digest = hashlib.sha256()
            with archive.open(info) as stream:
                for chunk in iter(lambda: stream.read(1024*1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected[name]["sha256"]:
                raise ArtifactError("ZIP内容SHA256不匹配："+name)
    return True

class HTTPSRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not newurl.lower().startswith("https://"):
            raise ArtifactError("拒绝非HTTPS重定向")
        return super().redirect_request(req, fp, code, msg, headers, newurl)

def download_archive(release, destination, opener=None):
    url = release.get("url")
    if release.get("status") != "published" or not isinstance(url, str) or not url.startswith("https://"):
        raise ArtifactError("模型包尚未公开发布，没有可用下载地址。可用 --archive 导入合法取得的研究包；不会编造链接。")
    opener = opener or urllib.request.build_opener(HTTPSRedirect()).open
    destination = Path(destination)
    if destination.exists():
        raise ArtifactError("下载目标已存在，拒绝覆盖")
    try:
        with opener(urllib.request.Request(url, headers={"User-Agent":"industrial-anomaly-inspection-model-installer"}), timeout=30) as response:
            if not response.geturl().lower().startswith("https://"):
                raise ArtifactError("下载最终地址不是HTTPS")
            with destination.open("xb") as stream:
                size=0
                while True:
                    chunk=response.read(1024*1024)
                    if not chunk:break
                    size+=len(chunk)
                    if size>release["bytes"]:
                        raise ArtifactError("下载大小超过登记值")
                    stream.write(chunk)
        verify_archive(destination,release)
        return destination
    except Exception:
        if destination.is_file():destination.unlink()
        raise

def install_archive(path, profile, store=DEFAULT_STORE, releases=None, catalog=None):
    entry, release = release_for(profile, releases, catalog)
    verify_archive(path, release)
    store = Path(store).resolve()
    store.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".release-", dir=store) as temp:
        stage=Path(temp).resolve()
        assert stage.is_relative_to(store) and stage!=store
        # Only individually verified registered paths are written; never extractall.
        with zipfile.ZipFile(path) as archive:
            for name in release["files"]:
                dest=stage/name;dest.parent.mkdir(parents=True,exist_ok=True)
                with archive.open(name) as source, dest.open("xb") as target:
                    shutil.copyfileobj(source,target)
        for name, expected in release["files"].items():
            if sha256(stage/name)!=expected["sha256"]:
                raise ArtifactError("解压内容发生变化，拒绝安装："+name)
        errors=inspect_folder(stage/"bundle",entry,True)
        if errors:raise ArtifactError("\n".join(errors))
        notices=stage/"notices";notices.mkdir()
        for name in release["files"]:
            if name not in BUNDLE:
                target=notices/name;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(stage/name,target)
        target_notices=store/"_notices"/entry["artifact_id"]
        if target_notices.is_symlink():raise ArtifactError("许可目录不能是符号链接")
        if target_notices.exists():
            actual={f.relative_to(target_notices).as_posix() for f in target_notices.rglob("*") if f.is_file()}
            expected=set(release["files"])-BUNDLE
            if actual!=expected or any((target_notices/n).is_symlink() or sha256(target_notices/n)!=release["files"][n]["sha256"] for n in expected):
                raise ArtifactError("已有许可文件被修改，拒绝覆盖")
        dest,created=import_bundle(stage/"bundle",store,catalog)
        target_notices.parent.mkdir(parents=True,exist_ok=True)
        if not target_notices.exists():os.rename(notices,target_notices)
    return dict(artifact=str(dest),notices=str(target_notices),created=created,model_loaded=False)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile",default="r18_reference")
    choice=parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--archive",type=Path)
    choice.add_argument("--download",action="store_true")
    parser.add_argument("--store",type=Path,default=DEFAULT_STORE)
    args=parser.parse_args()
    try:
        _,release=release_for(args.profile)
        if args.download:
            # Refuse unavailable releases before even creating download scratch.
            if release.get("status")!="published" or not release.get("url"):
                raise ArtifactError("研究模型包尚未公开发布；当前没有下载地址。请使用合法取得的 --archive。")
            scratch=ROOT/".cache/model-downloads";scratch.mkdir(parents=True,exist_ok=True)
            with tempfile.TemporaryDirectory(dir=scratch) as temp:
                stage=Path(temp).resolve()
                assert stage.is_relative_to(scratch.resolve())
                archive=download_archive(release,stage/"model.zip")
                result=install_archive(archive,args.profile,args.store)
        else:result=install_archive(args.archive,args.profile,args.store)
        print(json.dumps(result,ensure_ascii=False))
        return 0
    except (ArtifactError,OSError,ValueError,zipfile.BadZipFile) as exc:
        parser.exit(2,str(exc)+"\n")
if __name__=="__main__":raise SystemExit(main())

