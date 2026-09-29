# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added MIT copyright headers to all `movement_optimizer` source files
  under `src/` with SPDX-License-Identifier (closes #338).
- Added CI and DCO status badges to `README.md` (closes #334).
- Added PR check timing report step to `ci-standard.yml` so each run
  emits a markdown timing summary to `$GITHUB_STEP_SUMMARY` (closes #335).

### Changed

- **DbC**: Hardened `config.load_app_paths()` to validate that
  `MOVEMENT_OPTIMIZER_STATE_DIR` is non-empty before creating the
  `Path`, raising `ValueError` with a clear message instead of silently
  accepting whitespace (closes #329).

## [1.2.0] - 2026-04-26

### Changed

- **DRY**: Refactored `MainWindow._opt_worker` (95 lines) into three focused
  private helpers: `_resolve_exercise_params`, `_seg_mults`, and
  `_run_optimizer`; each helper is under 40 lines.
- **DbC**: Added precondition guards in `export.py` (`n_frames > 0`,
  `fps > 0`) and `cli.py` (`--body-mass`, `--height`, `--bar-mass`,
  `--duration`) with explicit `ValueError`/`parser.error()` messages.
- **LoD**: Fixed chained attribute access in `cli.py` (`_result_to_full_dict`)
  by extracting `.tolist()` chains into named locals; fixed
  `scripts/setup_dev.py` by naming `_SCRIPT_DIR` and `PROJECT_ROOT`.
- **Docs**: Added Google-style docstrings to `make_squat_config`,
  `make_full_squat_config`, `make_deadlift_config`, `BodyModel.__init__`,
  `LagrangianDynamics` properties and helpers, `ComparisonStore.__init__`,
  and `_build_parser` in the CLI module.

## [1.1.0] - 2026-03-23

### Added

- README.md with full project documentation
- LICENSE (MIT), CONTRIBUTING.md, and CHANGELOG.md
- CLI mode for headless batch optimization (`python3 -m movement_optimizer.cli`)
- Linux launch script (`launch-movement-optimizer.sh`)
- Dockerfile for reproducible builds
- Property-based tests with Hypothesis
- GUI widget smoke tests
- Benchmarking suite for performance regression testing
- `py.typed` marker for PEP 561 compliance
- N-DOF inverse dynamics in Rust accelerator
- 3D view controls with azimuth/elevation sliders and preset views
- Coverage threshold enforcement in CI (80%)
- Thread-safe ComparisonStore with `threading.Lock`
- Proper version management (`__version__` read from `pyproject.toml`)
- Batch vectorized COM and bar trajectory computation in `_package_results`

### Changed

- Split `gui.py` into `gui/` package (MainWindow, ExerciseTab, ComparisonDialog, widgets)
- Replaced all `assert` statements in `src/` with `ValueError`/`TypeError`
- Replaced bare `except Exception` catches in GUI with specific exceptions
- Replaced `print()` in GUI with `logger.error()`
- Added `encoding='utf-8'` to all `open()` calls in persistence.py
- Made `_balance_pose` a public API (`balance_pose`)
- Removed Black from dev dependencies
- Fixed mypy CI step (removed `|| true`)
- Fixed CI `paths-ignore` to not exclude `.github` changes
- Updated CLAUDE.md command examples to use `python3` instead of `py`
- Removed stale mypy `additional_dependencies` from pre-commit config
- Deduplicated test helpers (`_make_result` -> shared fixture in conftest.py)

### Fixed

- ComparisonStore thread-safety (was documented as thread-safe but lacked locks)

## [1.0.0] - 2025-01-01

### Added

- Initial release with GUI, 2D/3D models, and Rust accelerator
