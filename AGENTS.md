# Project working rules

- This directory is the user's canonical project folder. Keep all project source, notes, launchers, datasets, downloads, generated results, project-specific environment and caches inside it. Resolve paths from the project or script location, never from the chat workspace.
- The user manually launches fitting and learning experiments. Prepare and validate launchers; do not start new fitting or image evaluation without authorization. Synthetic checks and input-only preflight are allowed.
- Preserve existing baselines and experimental evidence. Record new decisions and results separately; keep the 70 held-out images sealed until final configurations are frozen and evaluation is authorized.
- Historical source and launcher backups are under `.migration/`. Treat their embedded paths as historical, not as active execution locations.
- Do not publish raw dataset images, model weights, the virtual environment, caches, private absolute paths or local migration reports. Use explicit file allowlists for GitHub synchronization.
- Be precise about contribution: the backbone and reference loader still use anomalib; the learning feature organization module is AI-assisted PyTorch code, not a complete original detector.
