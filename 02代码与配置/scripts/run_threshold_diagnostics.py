"""Reproduce the saved-map scan and constrained selection in a fresh folder."""
from pathlib import Path
import subprocess
import sys


def main():
    scripts = Path(__file__).resolve().parent
    root = scripts.parents[1]
    parent = root / '04实验结果/11阈值扫描与方案选择'
    before = set(parent.iterdir()) if parent.exists() else set()
    subprocess.run([sys.executable, '-X', 'utf8', '-u', str(scripts / 'run_threshold_scan.py'), *sys.argv[1:]], check=True)
    created = set(parent.iterdir()) - before
    if len(created) != 1:
        raise RuntimeError('Could not uniquely identify this execution output')
    output = created.pop()
    subprocess.run([sys.executable, '-X', 'utf8', '-u', str(scripts / 'select_guarded_threshold.py'), '--analysis', str(output)], check=True)
    print('FULL_THRESHOLD_DIAGNOSTICS_COMPLETED | ' + str(output), flush=True)


if __name__ == '__main__':
    main()
