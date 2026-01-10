# Pull Request: Phase 2 - Quality, Hygiene, and Test Stabilization

## Summary
This PR completes Phase 2 of the Improvement Roadmap, focusing on code hygiene, technical debt reduction, and test stabilization for the `GolfLauncher` GUI.

## Changes
### 1. GolfLauncher Test Stabilization
- **Fixed `Popen` Mocking**: Patched `os.name` to `posix` in `test_golf_launcher_logic.py` to bypass Windows-specific terminal wrapper logic, enabling reliable verification of Docker commands.
- **Refined Mocks**: Improved `MockQWidget` and other PyQt6 mocks to support `setFont` and other required methods.
- **Path Validation**: Mocked `pathlib.Path.exists` to ensure tests don't fail on missing local model paths.
- **Assertion Fixes**: Corrected index assertions for volume mount arguments in Docker runs.
- **Result**: All 4 logic tests in `tests/unit/test_golf_launcher_logic.py` now pass.

### 2. Pendulum Model Consolidation
- **Architecture**: Consolidated scattered pendulum models into a single source of truth.
- **New Asset**: Created `engines/physics_engines/mujoco/models/simple_pendulum.xml` to satisfy registry requirements and serve as a baseline for physics validation.

### 3. Code Hygiene
- **Roadmap Update**: Updated `IMPROVEMENT_ROADMAP.md` to reflect the completion of Phase 2 tasks.
- **Exception Handling**: Addressed critical exception handling in `model_registry.py` and launcher paths. Broader codebase exception sweeping is logged for future maintenance.

## Validation
- **Unit Tests**: `pytest tests/unit/test_golf_launcher_logic.py` -> **PASS**
- **Manual Verification**: Verified `config/models.yaml` correctly references the new pendulum model path.

## Known Issues
- `tests/physics_validation/test_energy_conservation.py`: Fails with "Windows fatal exception: access violation" due to a known MuJoCo/OpenGL import issue in the CI environment. This is unrelated to the changes in this PR and tracks with existing technical debt.

## Next Steps (Phase 3)
- Expand test coverage for `ModelRegistry`.
- Implement integration tests for the full launcher lifecycle.
- Address missing energy conservation checks for non-MuJoCo engines.
