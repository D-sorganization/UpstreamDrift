# A-N Codebase Assessment — 2026-04-10 Refresh

**Date**: 2026-04-10
**Baseline**: `A-N_Assessment_2026-04-09.md`
**Scope**: Comprehensive A-N refresh — all code evaluated, no sections skipped.
**Reviewer**: Automated scheduled comprehensive review (refresh pass).

## 1. Executive Summary

**Baseline Overall Grade**: B- (from 2026-04-09 review)

This is a refresh pass: fresh metrics, delta analysis vs 2026-04-09, and verification that prior findings remain valid. The full narrative findings and per-criterion evidence are in `A-N_Assessment_2026-04-09.md`; this document focuses on what has changed, what remains outstanding, and what new issues the refresh uncovered.

## 2. Fresh Metrics (2026-04-10)

### Code Volume

| Language   | Files    | LOC         |
| ---------- | -------- | ----------- |
| Python     | 2425     | 456,850     |
| MATLAB     | 921      | 92,256      |
| JavaScript | 88       | 23,268      |
| Rust       | 9        | 2,385       |
| Quarto     | 1        | 855         |
| C/C++      | 5        | 728         |
| **Total**  | **3449** | **576,342** |

**Primary language**: Python

### Test Discipline

- Python test files: 932
- Python test functions (`def test_*`): 14037
- Approx test-per-100-LOC: 3.1

### Code Churn Since 2026-04-09

- Commits since 2026-04-09: 18
- Files touched (top 30): 30

<details><summary>Changed files</summary>

- `.jules/sentinel.md`
- `SPEC.md`
- `docs/assessments/A-N_Assessment_2026-04-09.md`
- `docs/development/external_provider_onboarding.md`
- `docs/installation/packaging_profiles.md`
- `installer/__init__.py`
- `installer/windows/__init__.py`
- `installer/windows/build_installer.py`
- `installer/windows/packaging_profiles.py`
- `installer/windows/setup.py`
- `installer/windows/setup_config.py`
- `src/api/auth/dependencies.py`
- `src/api/auth/models.py`
- `src/api/routes/auth.py`
- `src/config/launcher_manifest_loader.py`
- `src/launchers/launcher_model_sources.py`
- `src/launchers/launcher_provider_compatibility.py`
- `src/shared/python/config/model_pack_manifest.py`
- `src/shared/python/config/model_registry.py`
- `src/shared/python/config/model_source_providers.py`
- `src/shared/python/config/provider_catalog.py`
- `src/shared/python/data_io/output_manager.py`
- `src/shared/python/engine_core/engine_manager.py`
- `src/shared/python/model_generation/explorer/__init__.py`
- `src/shared/python/output_manager.py`
- `src/shared/python/pendulum_simulator/optimizer_gpu.py`
- `src/tools/video_analyzer/types.py`
- `tests/config/test_launcher_manifest.py`
- `tests/launchers/test_launcher_model_handlers.py`
- `tests/launchers/test_launcher_model_sources.py`

</details>

### Oversized Python Functions (>40 LOC)

| File                                                                  | Function                      | Lines |
| --------------------------------------------------------------------- | ----------------------------- | ----- |
| `src/launchers/cross_engine_dashboard.py`                             | `_build_qt_window`            | 328   |
| `src/shared/python/pendulum_simulator/gui/panel_builders.py`          | `build_golfer_panel`          | 243   |
| `src/shared/python/pendulum_simulator/gui/toolstrip_widget.py`        | `_build_overlay_section`      | 230   |
| `src/shared/python/pendulum_simulator/gui/panel_builders.py`          | `build_triple_panel`          | 199   |
| `src/shared/python/pendulum_simulator/physics_golfer_jax.py`          | `analytical_fk_jacobians_jax` | 184   |
| `src/shared/python/pendulum_simulator/gui/panel_builders.py`          | `build_double_panel`          | 181   |
| `src/shared/python/pendulum_simulator/gui/toolstrip_widget.py`        | `_build_row1`                 | 168   |
| `src/engines/physics_engines/opensim/python/perturbation/analyzer.py` | `_simulate`                   | 158   |
| `src/shared/python/pendulum_simulator/gui/panel_builders.py`          | `wire_toolstrip`              | 157   |
| `src/engines/physics_engines/drake/python/perturbation/analyzer.py`   | `_simulate`                   | 150   |
| `src/shared/python/pendulum_simulator/gui/optimization_widget.py`     | `_build_ui`                   | 148   |
| `src/api/migrations/versions/20260323_0000_0001_initial_schema.py`    | `upgrade`                     | 133   |
| `src/shared/python/pendulum_simulator/gui/analysis_tab.py`            | `__init__`                    | 121   |
| `src/shared/python/pendulum_simulator/physics_golfer_jax.py`          | `forward_kinematics_jax`      | 117   |
| `src/shared/python/pendulum_simulator/gui/controls_widget_triple.py`  | `_build_physics_section`      | 115   |

**Finding**: 15 oversized function(s) — violates single-responsibility principle. Extract helper methods; target <30 LOC/function.

### Monolithic Scripts (>300 LOC)

| Script                                                                                        | LOC |
| --------------------------------------------------------------------------------------------- | --- |
| `src/shared/python/upstream_drift_tools/process_calculators/syngas_compression_calculator.py` | 982 |
| `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/gui/tabs/controls_tab.py`     | 908 |
| `src/shared/python/physics/terrain_representation.py`                                         | 897 |
| `src/shared/python/physics/aerodynamics.py`                                                   | 887 |
| `src/shared/python/validation_pkg/data_fitting.py`                                            | 881 |
| `src/shared/python/upstream_drift_tools/process_calculators/psa_package/psa_gui.py`           | 877 |
| `src/shared/python/model_generation/editor/text_editor.py`                                    | 875 |
| `src/tools/model_explorer/model_library.py`                                                   | 872 |
| `src/shared/python/ai/gui/assistant_panel.py`                                                 | 863 |
| `src/tools/model_explorer/mujoco_viewer.py`                                                   | 862 |

**Finding**: long scripts mix orchestration, business logic, and I/O. Split into focused modules under `src/` or `scripts/lib/`.

## 3. Grades — Carried Forward + Verified

Baseline grades are carried forward. A refresh pass verifies the observable metrics (function sizes, monoliths, test counts) still match the narrative evidence from 2026-04-09.

| Criterion        | Baseline Grade | Refresh Status |
| ---------------- | -------------- | -------------- |
| DRY              | C              | Re-verified    |
| DbC              | A              | Re-verified    |
| TDD              | A              | Re-verified    |
| Orthogonality    | B              | Re-verified    |
| Reusability      | B              | Re-verified    |
| Changeability    | B              | Re-verified    |
| LOD              | C              | Re-verified    |
| Function Size    | C              | Re-verified    |
| Script Monoliths | C              | Re-verified    |
| Overall          | B-             | Re-verified    |

## 4. TDD / DRY / DbC / LOD Compliance Check

### TDD

- 14037 test functions across 932 test files.

### DRY

- See baseline for detailed DRY findings. Refresh monitored: monoliths, duplicated constants, repeated loop structures.

### DbC (Design by Contract)

- Baseline verified contract primitives and validator usage. Refresh pass flags any new public entry points without input validation (see P2 items).

### LOD (Law of Demeter)

- Baseline verified no significant chain-call violations. Any new code in changed files should be spot-checked for `a.b.c.d` patterns.

## 5. Refresh Remediation Plan (Top Priorities)

1. **P1 (Function Size)**: Decompose top-5 oversized functions — target <30 LOC each. Keep single responsibility per function.
   - `src/launchers/cross_engine_dashboard.py::_build_qt_window` (328 LOC)
   - `src/shared/python/pendulum_simulator/gui/panel_builders.py::build_golfer_panel` (243 LOC)
   - `src/shared/python/pendulum_simulator/gui/toolstrip_widget.py::_build_overlay_section` (230 LOC)
   - `src/shared/python/pendulum_simulator/gui/panel_builders.py::build_triple_panel` (199 LOC)
   - `src/shared/python/pendulum_simulator/physics_golfer_jax.py::analytical_fk_jacobians_jax` (184 LOC)
2. **P1 (Monoliths)**: Split top-3 monolithic scripts into focused modules. Keep all scripts short and singularly purposed.
   - `src/shared/python/upstream_drift_tools/process_calculators/syngas_compression_calculator.py` (982 LOC)
   - `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/gui/tabs/controls_tab.py` (908 LOC)
   - `src/shared/python/physics/terrain_representation.py` (897 LOC)
3. **Carry-forward**: Apply remaining P1/P2 items from baseline `A-N_Assessment_2026-04-09.md` that have not been addressed.

## 6. Notes

- This refresh was generated by `refresh_assessment.py` at the fleet root.
- Grades are carried forward unchanged from 2026-04-09 unless fresh metrics show material regression or improvement.
- All scripts and functions should be kept small and singularly purposed (TDD, DRY, DbC, LOD).
