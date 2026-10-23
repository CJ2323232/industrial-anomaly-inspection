"""Run the local experiment with the verified environment on this laptop."""

import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    project_root = root.parent
    workspace = project_root
    python = workspace / '.venv/Scripts/python.exe'
    if not python.is_file():
        raise SystemExit('The verified local Python environment was not found; see LOCAL_RUN.md.')
    environment = os.environ.copy()
    for name, relative in {'TORCH_HOME': '.cache/models/torch', 'HF_HOME': '.cache/models/huggingface',
                           'XDG_CACHE_HOME': '.cache/models', 'MPLCONFIGDIR': '.cache/matplotlib',
                           'TEMP': '.cache/tmp', 'TMP': '.cache/tmp'}.items():
        cache = workspace / relative
        cache.mkdir(parents=True, exist_ok=True)
        environment[name] = str(cache)
    environment['PYTHONIOENCODING'] = 'utf-8'
    arguments = sys.argv[1:]
    if '--output' not in arguments and not any(value.startswith('--output=') for value in arguments):
        timestamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
        arguments += ['--output', str(project_root / '04实验结果/03早期运行与原理核验' / f'run_{timestamp}')]
    return subprocess.run([str(python), '-X', 'utf8', '-u', 'scripts/run_first_experiment.py', *arguments],
                          cwd=root, env=environment, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
