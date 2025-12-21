# Physics Verification Report
**Date:** 2025-12-21 09:53:14

## 1. Engine Status
| Engine | Status | Version | Notes |
|---|---|---|---|
| mujoco | available | 3.3.4 | MuJoCo 3.3.4 ready |
| drake | missing_binary | unknown | Drake installed but 'pydrake.multibody' missing. Installation might be corrupted. |
| pinocchio | not_installed | N/A | Pinocchio Python package not installed. Install with: pip install pin |
| pendulum | missing_assets | local | Pendulum models missing: module: pendulum_solver.py |

## 2. Validation Test Results
| Test Case | Outcome | Duration (s) |
|---|---|---|
| test_pinocchio_golfer_stability | SKIPPED | 0.0000 |
| test_mujoco_myoarm_stability | PASSED | 0.0157 |
| test_mujoco_ballistic_energy_conservation | PASSED | 0.0224 |
| test_pinocchio_energy_check | SKIPPED | 0.0000 |
| test_mujoco_pendulum_accuracy | PASSED | 0.0928 |
| test_drake_pendulum_accuracy | SKIPPED | 0.0000 |

## 3. Analysis & Recommendations
### ⚠️ Skipped Tests
- **tests/physics_validation/test_complex_models.py::test_pinocchio_golfer_stability** was skipped. Check engine availability.
- **tests/physics_validation/test_energy_conservation.py::test_pinocchio_energy_check** was skipped. Check engine availability.
- **tests/physics_validation/test_pendulum_accuracy.py::test_drake_pendulum_accuracy** was skipped. Check engine availability.