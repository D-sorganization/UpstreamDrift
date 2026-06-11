# Scripts Directory

This directory contains utility and maintenance scripts for the UpstreamDrift project.

## Organization

- **`chore/`** — Code quality and refactoring utilities

  - `replace_prints.py` — Replace `print()` calls with logger.info() in source files. Use before committing to ensure compliance with no-print-in-src rule.
  - `patch_analyzers.py` — Runtime monkey-patching utility for analyzer classes (development only, not part of CI)

- **`ci/`** — CI/CD and testing infrastructure

  - `start_api_server.py` — Start the local API server for integration testing
  - `run_local_heavy_tests.sh` — Run the full test suite with heavy markers on local machine
  - `check_file_size_budget.py` — Verify files don't exceed size budget (runs in CI)
  - `check_tutorial_imports.py` — Validate Python imports in tutorial documentation (runs in CI)
  - `check_pip_audit_waivers.py` — Validate JSON pip-audit waivers and enforce expiry dates (runs in CI)
  - `verify_installation.py` — Verify the UpstreamDrift installation and dependencies

- **`config/`** — Script-owned CI configuration

  - `pip_audit_waivers.json` — Time-bounded pip-audit CVE waivers consumed by `scripts/ci/check_pip_audit_waivers.py`

- `check_workflows_no_silent_failures.py` — Reject silent security scanner failure patterns in core security workflows

- **`maintenance/`** — System and deployment utilities

  - `start-gaai-daemon.sh` — Bootstrap the GAAI framework daemon
  - `run_wsl.sh` — Run tests in WSL environment (Windows)

- **`analysis/`** — Data analysis and reporting (reserved for future use)

## Root-level Entry Points

These scripts are intentionally at the repository root as they are primary entry points:

- `launch_golf_suite.py` — Deprecated launcher (being renamed per A2). Use `upstream-drift` CLI entry point instead (registered in `pyproject.toml`).
- `setup_golf_suite.py` — Setup and initialization helper (candidate for consolidation per C3)
- `install.sh` — Installation helper script
- `build_hooks.py` — Hatchling build hook (required by pyproject.toml, must stay at root)
- `launch_urdf_generator.bat` — Windows batch wrapper (candidate for consolidation per C3)

## Notes

- Never add new `.py` scripts to the repository root without explicit approval
- When possible, consolidate functionality into existing subdirectories
- Update this README when scripts are added or moved
- Each script should have clear docstrings explaining its purpose and intended usage
