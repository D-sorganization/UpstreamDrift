# Cursor Safety Starter

Guardrails for MATLAB + Python projects (golf swing modeling and chemical simulations). Created on 2025-08-09.

## Quick Start

```bash
# 1) Initialize repo + Git LFS
git init
git lfs install

# 2) Install pre-commit hooks
bash scripts/setup_precommit.sh

# 3) Protect main (on GitHub): require PR + status checks
# 4) Start safe WIP branch and code
git checkout -b chore/wip-$(date +%F)
```

## Daily Safety

- Commit every ~30 minutes (`wip:` if tests fail).
- End-of-day snapshot: `bash scripts/snapshot.sh`.
- Big AI refactor? Create `backup/before-ai-<desc>` branch first.

## Reproducibility

- `matlab/run_all.m` should regenerate results.
- Python env pinned via `python/requirements.txt` or `python/environment.yml`.

## Cursor Settings Synchronization

This repository includes optimized Cursor settings for cross-computer development:

- **`cursor-settings.json`** - Complete settings with stall prevention and performance optimization
- **`CURSOR_SETTINGS_README.md`** - Detailed usage guide for syncing settings across computers

Copy these settings to your Cursor `settings.json` for consistent development experience.

