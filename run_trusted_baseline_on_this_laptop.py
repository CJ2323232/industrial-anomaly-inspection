"""Manual launcher for this chat's existing laptop environment and weight cache.

This local helper contains workspace assumptions; use the portable script in
scripts/run_trusted_baseline.py on another machine.
"""

import argparse
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--data-root", type=Path, default=root / "datasets/MVTecAD")
    parser.add_argument("--output-root", type=Path, default=root / "experiments")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    workspace = root
    python = workspace / ".venv/Scripts/python.exe"
    if not python.is_file():
        raise SystemExit("The verified Python environment was not found")
    environment = os.environ.copy()
    for name, relative in {"TORCH_HOME": ".cache/models/torch", "HF_HOME": ".cache/models/huggingface",
                           "XDG_CACHE_HOME": ".cache/models", "MPLCONFIGDIR": ".cache/matplotlib",
                           "TEMP": ".cache/tmp", "TMP": ".cache/tmp"}.items():
        environment[name] = str(workspace / relative)
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["HF_HUB_OFFLINE"] = "1"
    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    destination = args.output_root / f"baseline_{timestamp}"
    command = [str(python), "-X", "utf8", "-u", "scripts/run_trusted_baseline.py",
               "--data-root", str(args.data_root.resolve()), "--manifest", "configs/splits/metal_nut-v1.json",
               "--output", str(destination.resolve()), "--seed", str(args.seed)]
    if args.check_only:
        command.append("--check-only")
        return subprocess.run(command, cwd=root, env=environment, check=False).returncode
    args.output_root.mkdir(parents=True, exist_ok=True)
    log_path = destination.with_suffix(".log")
    print(f"Experiment output: {destination.resolve()}", flush=True)
    print(f"Console log: {log_path.resolve()}", flush=True)
    with log_path.open("x", encoding="utf-8") as stream:
        with subprocess.Popen(command, cwd=root, env=environment, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace") as process:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end="", flush=True)
            code = process.wait()
    print(f"Process exit code: {code}", flush=True)
    return code


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
