# Phase 1 & 2: Cross-Engine Validation & URDF/C3D Features — Working Checklist

**Branch**: `feat/phase1-cross-engine-validation`  
**PR**: [#347](https://github.com/D-sorganization/Golf_Modeling_Suite/pull/347)  
**Session**: 2026-01-10

---

## ✅ Phase 1: Cross-Engine Validation Activation (COMPLETE)

### Task 1.1: Model Loading Infrastructure

- [x] Create `tests/fixtures/models/` directory
- [x] Create `simple_pendulum.urdf` — 1-DOF analytical reference
- [x] Create `double_pendulum.urdf` — 2-DOF for counterfactual tests
- [x] Add `fixtures_lib.py` with shared fixtures
- [x] Implement `mujoco_pendulum`, `drake_pendulum`, `pinocchio_pendulum` fixtures
- [x] Implement `available_engines()` fixture for dynamic detection

### Task 1.2: Three-Way Engine Triangulation

- [x] Implement pairwise validation logic (MuJoCo ↔ Drake)
- [x] Implement pairwise validation logic (MuJoCo ↔ Pinocchio)
- [x] Implement pairwise validation logic (Drake ↔ Pinocchio)
- [x] Add Pinocchio as tiebreaker logic
- [x] Remove all `pytest.skip` calls in test_cross_engine_consistency.py

### Task 1.3: Conservation Law Test Activation

- [x] Implement energy conservation test with inline pendulum (RK4, 0.5ms timestep)
- [x] Implement work-energy theorem test with actuated model
- [x] Implement drift-control superposition test
- [x] Implement ZTCF equals drift test
- [x] Implement ZVCF eliminates Coriolis test
- [x] Remove all `pytest.skip` placeholders

### Test Results (Phase 1)

- ✅ **21 passed** (Conservation + CrossEngine unit tests)
- ⏭️ **11 skipped** (Engines not installed - expected in CI)

---

## ✅ Phase 2: URDF Generator & C3D Integration (COMPLETE)

### Task 2.1: MuJoCo Visualization Embed ✅

- [x] Create `mujoco_viewer.py` with MuJoCoViewerWidget
- [x] Implement `URDFToMJCFConverter` for real-time URDF preview
- [x] Add `MuJoCoOffscreenRenderer` for Qt-embedded rendering
- [x] Mouse-based camera control (rotate, zoom)
- [x] Visualization toggles: Collision, Frames, Joint Limits
- [x] Physics sanity checks:
  - [x] Inertia positive-definiteness validation
  - [x] Joint axis normalization check
- [x] "Launch Full Viewer" button for standalone MuJoCo

### Task 2.2: Force-Plate Parsing Pipeline ✅

- [x] Add `get_force_plate_channels()` to C3DDataReader
  - [x] Standard naming detection (Fx1, Fy1, Fz1, Mx1, My1, Mz1)
  - [x] Prefixed naming detection (Force.Fx1, FP1Fx)
  - [x] Channel mapping by plate number
- [x] Add `force_plate_dataframe()` to C3DDataReader
  - [x] Extract GRF components from analog channels
  - [x] Compute Center of Pressure (COP_x = -My/Fz, COP_y = Mx/Fz)
  - [x] Handle missing contact (COP = NaN when Fz < 10N)
  - [x] Optional time column using analog sample rate
- [x] Add `get_force_plate_count()` convenience method
- [x] Add unit tests (12 tests passing)

### Task 2.3: Force-Plate Visualization ✅

- [x] Create `ForcePlotTab` for C3D viewer
- [x] GRF component time-series (Fx, Fy, Fz, Mx, My, Mz)
- [x] COP trajectory trace with time-colored scatter plot
- [x] Multi-plate support with dropdown selection
- [x] Start/end markers on COP trajectory

---

## Quality Gates ✅

### Pre-Commit Verification

- [x] `black .` passes
- [x] `ruff check .` passes
- [x] All tests pass (Phase 1: 21 passed, 11 skipped; Phase 2: 12 passed)

### Commits

1. `feat(phase1): Add cross-engine validation fixtures and test infrastructure`
2. `fix: Tune pendulum model physics for energy conservation tests`
3. `feat(phase2): Add force plate parsing pipeline (Guideline E5)`
4. `docs: Update checklist with Phase 1 and 2.2 completion`
5. `feat(phase2): Add MuJoCo viewer and force plate visualization (Tasks 2.1, 2.3)`

---

## Acceptance Criteria Summary

| Guideline                         | Task      | Status      |
| --------------------------------- | --------- | ----------- |
| M2: Cross-engine infrastructure   | Phase 1   | ✅ Complete |
| P3: Tolerance-based validation    | Phase 1   | ✅ Complete |
| G1/G2: ZTCF/ZVCF tests            | Phase 1   | ✅ Complete |
| F: Drift-control decomposition    | Phase 1   | ✅ Complete |
| O3: Energy conservation <1% drift | Phase 1   | ✅ Complete |
| B3: URDF visualization            | Phase 2.1 | ✅ Complete |
| E5: Ground reaction forces        | Phase 2.2 | ✅ Complete |
| A1: C3D force-plate parsing       | Phase 2.2 | ✅ Complete |
| A1: Force-plate visualization     | Phase 2.3 | ✅ Complete |

---

## Next Steps: Phase 3

### Task 3.1: Jacobian Coverage Completion

- [ ] Implement `compute_jacobian()` for OpenSim
- [ ] Verify MyoSuite Jacobian body ID resolution
- [ ] Add Jacobian shape tests (6×nv)

### Task 3.2: Mobility/Force Ellipsoid Visualization

- [ ] Add `compute_force_ellipsoid()`
- [ ] Implement 3D ellipsoid rendering
- [ ] Add export to OBJ/STL

### Task 3.3: Flexible Shaft Engine Integration

- [ ] Add `set_shaft_properties()` to PhysicsEngine
- [ ] Implement modal shaft in MuJoCo
- [ ] Add cross-engine validation for shaft deflection

---

## Files Changed (Total: 8 new, 4 modified)

### New Files

- `tests/fixtures/models/simple_pendulum.urdf`
- `tests/fixtures/models/double_pendulum.urdf`
- `tests/fixtures/fixtures_lib.py`
- `tests/integration/conftest.py`
- `tests/unit/test_c3d_force_plate.py`
- `tools/urdf_generator/mujoco_viewer.py`
- `engines/.../apps/ui/tabs/force_plot_tab.py`

### Modified Files

- `tests/integration/test_conservation_laws.py`
- `tests/integration/test_cross_engine_validation.py`
- `tests/integration/test_cross_engine_consistency.py`
- `engines/.../c3d_reader.py`
