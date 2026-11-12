"""Verify the pinned research assets on the runner before publishing one release."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

from install_model_release import verify_archive

ROOT = Path(__file__).resolve().parents[2]
REPO = "CJ2323232/industrial-anomaly-inspection"
RELEASE_ID = 428745479
TAG = "research-models-v1"
TARGET = "956e9f8839785917d53392ceda7226e48d53c2b7"
PINS = {
    "r18-reference-v1.zip": (13974512, "aad3ad16dd25f28a1af52da46cdba54b38b98087145d1c4e619f6b31fb97bd41"),
    "wrn50-reference-v1.zip": (110794848, "2be5dc849a43b8a92c10c5b6811fb793f37d61c64cf67df03b7598cb4740ddfb"),
    "SHA256SUMS.txt": (178, "f809ef38a0ad3adc5fc2a9b5b6e610ef2ae43b3f1c103d546571c6edd065622e"),
}


def api(*parts):
    result = subprocess.run(["gh", "api", *parts], capture_output=True, text=True,
                            encoding="utf-8", check=True, timeout=60)
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT/"02代码与配置/configs/delivery/model_releases.json").read_text(encoding="utf-8"))
    for artifact in manifest["artifacts"].values():
        if (artifact["bytes"], artifact["sha256"]) != PINS[artifact["filename"]]:
            raise ValueError("Source manifest differs from approved asset pins")
    if not args.publish:
        print("Release pins verified. No network operation requested.")
        return
    if os.environ.get("GITHUB_REPOSITORY") != REPO:
        raise ValueError("This publisher is restricted to the original repository runner")
    endpoint = f"repos/{REPO}/releases/{RELEASE_ID}"
    release = api(endpoint)
    if release["tag_name"] != TAG or release["target_commitish"] != TARGET:
        raise ValueError("Release identity differs")
    assets = {asset["name"]: asset for asset in release["assets"]}
    if set(assets) != set(PINS):
        raise ValueError("Release asset allowlist differs")
    output = ROOT/"04实验结果/cloud-release-verification"
    output.mkdir(parents=True, exist_ok=True)
    verified = []
    for name, (size, expected_sha) in PINS.items():
        asset = assets[name]
        if (asset["size"], asset["digest"], asset["state"]) != (size, "sha256:"+expected_sha, "uploaded"):
            raise ValueError("Remote metadata differs: "+name)
        for attempt in range(3):
            fd, temporary = tempfile.mkstemp(dir=output)
            os.close(fd)
            destination = Path(temporary)
            try:
                with destination.open("wb") as stream:
                    subprocess.run(["gh", "api", f"repos/{REPO}/releases/assets/{asset['id']}",
                                    "-H", "Accept: application/octet-stream"],
                                   stdout=stream, check=True, timeout=180)
                break
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                if attempt == 2:
                    raise
                time.sleep(2)
        actual_sha = hashlib.sha256(destination.read_bytes()).hexdigest()
        if destination.stat().st_size != size or actual_sha != expected_sha:
            raise ValueError("Downloaded bytes differ: "+name)
        if name.endswith(".zip"):
            entry = next(item for item in manifest["artifacts"].values() if item["filename"] == name)
            verify_archive(destination, entry)
        verified.append({"name": name, "bytes": size, "sha256": actual_sha})
        print("Full download and SHA256 verified: "+name, flush=True)
    # Publication is gated on all three actual downloads and both inner ZIP checks.
    if release["draft"]:
        api("--method", "PATCH", endpoint, "-F", "draft=false", "-f", "make_latest=false")
    published = api(endpoint)
    if published["draft"]:
        raise ValueError("Release remains a draft")
    receipt = {"release_id": RELEASE_ID, "tag": TAG, "url": published["html_url"],
               "runner_downloads_verified": verified, "cpu_inference_after_public_download": "pending"}
    (output/"receipt.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as stream:
            stream.write("Research release published after full byte and inner ZIP verification.\n\n")
            stream.write(published["html_url"]+"\n")
    print("PUBLISH_SUCCESS "+published["html_url"], flush=True)


if __name__ == "__main__":
    main()
