# Pull Request: Phase 1 Stabilization Fixes

## Summary
This PR addresses critical stability and configuration issues identified in the "Comprehensive Code Review Report (2025-12-27)". It focuses on enabling proper test discovery, standardizing dependency versions, and fixing potential runtime hangs.

## Changes

### 1. Configuration & Dependency Fixes (`pyproject.toml`)
- **Ruff Version**: Updated to `0.14.10` to match CI/CD and pre-commit config, resolving potential inconsistent linting results.
- **Repository URLs**: Replaced placeholder URLs with actual organization links.

### 2. Code Quality & Reliability
- **Model Registry (`shared/python/model_registry.py`)**:
    - Replaced raw `print()` statements with `logger.error()` and `logger.warning()`.
    - Added specific exception handling for `yaml.YAMLError` and `OSError`.
    - Added explicit UTF-8 encoding for file operations.
- **Launcher (`launchers/golf_launcher.py`)**:
    - Added `timeout=5.0` to Docker version check to prevent UI hanging if Docker is unresponsive.
    - Added `subprocess.TimeoutExpired` handling.

### 3. Test infrastructure
- **Test Discovery**:
    - Added missing `__init__.py` files to `tests/unit/` and `tests/physics_validation/`.
- **Runtime Safety**:
    - Refactored `tests/physics_validation/test_complex_models.py`, `test_energy_conservation.py`, and `test_pendulum_accuracy.py`.
    - Moved engine imports (e.g., `import mujoco`) *inside* test functions guarded by `is_engine_available()`. This prevents the test collector from crashing with access violations when optional physics engines are missing.

## Verification
- `pytest --co` now successfully collects all tests without crashing (previously output 0 tests or crashed).
- Linting checks should pass with the synchronized ruff version.

## Next Steps
This PR sets the foundation for Phase 2 (Quality & Hygiene), which will address TODOs and duplicated code.
