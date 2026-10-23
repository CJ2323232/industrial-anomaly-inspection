# Project working rules

- This directory is the user's canonical project folder. Keep all project source, notes, launchers, datasets, downloads, generated results, project-specific environment and caches inside it. Resolve paths from the project or script location, never from the chat workspace.
- Drive progress through concrete project deliverables, hypotheses and measured experiments. Explain relevant principles within that work; do not make a sequence of lessons a prerequisite for project progress.
- The user manually launches fitting and learning experiments. Prepare and validate launchers; do not start new fitting or image evaluation without authorization. Synthetic checks and input-only preflight are allowed.
- Preserve existing baselines and experimental evidence. Record new decisions and results separately; keep the 70 held-out images sealed until final configurations are frozen and evaluation is authorized.
- Historical source and launcher backups are under `.migration/`. Treat their embedded paths as historical, not as active execution locations.
- Do not publish raw dataset images, model weights, the virtual environment, caches, private absolute paths or local migration reports. Use explicit file allowlists for GitHub synchronization.
- Be precise about contribution: the backbone and reference loader still use anomalib; the learning feature organization module is AI-assisted PyTorch code, not a complete original detector.

- Folder layout (2026-10-23): launchers in `01运行入口/`; source/tests/configs in `02代码与配置/`; raw data in `03原始数据/`; generated results in `04实验结果/`; documentation in `05项目文档/`. Keep `.venv`, `.cache`, `.migration` at the project root. Run Python tests with the code folder as cwd.
- Old paths embedded in frozen reports are historical provenance. Consult `.migration/organization_20261023/move_plan.json`; do not rewrite frozen evidence to update paths.
