# Phase 1 & 2: Cross-Engine Validation & Force Plate Pipeline — Working Checklist

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

## ✅ Phase 2: Force Plate Parsing (Task 2.2 COMPLETE)

### Task 2.2: Force-Plate Parsing Pipeline

- [x] Add `get_force_plate_channels()` to C3DDataReader
  - [x] Standard naming detection (Fx1, Fy1, Fz1, Mx1, My1, Mz1)
  - [x] Prefixed naming detection (Force.Fx1, FP1Fx)
  - [x] Channel mapping by plate number

- [x] Add `force_plate_dataframe()` to C3DDataReader
  - [x] Extract GRF components from analog channels
  - [x] Compute Center of Pressure (COP_x = -My/Fz, COP_y = Mx/Fz)
  - [x] Handle missing contact (COP = NaN when Fz < 10N threshold)
  - [x] Optional time column using analog sample rate
  - [x] Filter by specific plate number

- [x] Add `get_force_plate_count()` convenience method

- [x] Add unit tests for force plate parsing
  - [x] Test standard channel naming
  - [x] Test prefixed channel naming
  - [x] Test no force plates case
  - [x] Test dataframe extraction
  - [x] Test COP computation
  - [x] Test COP NaN when no contact
  - [x] Test plate selection
  - [x] Test invalid plate error
  - [x] Test force plate count

### Test Results (Phase 2)

- ✅ **12 passed** (Force plate tests)

---

## 📋 Remaining Phase 2 Tasks

### Task 2.1: MuJoCo Visualization Embed (Not Started)

- [ ] Integrate MuJoCo passive viewer into Qt widget
- [ ] Implement real-time URDF → MJCF conversion
- [ ] Add collision/frame/joint limit visualization toggles
- [ ] Add inertia validation checks

### Task 2.3: Force-Plate Visualization (Not Started)

- [ ] Add force vector overlay to 3D marker view
- [ ] Add COP trajectory trace on ground plane
- [ ] Add GRF component time-series plots

---

## Quality Gates

### Pre-Commit Verification

- [x] `black .` passes
- [x] `ruff check .` passes
- [x] `mypy .` passes
- [x] Unit tests pass
- [x] Integration tests pass

### Commits

1. `feat(phase1): Add cross-engine validation fixtures and test infrastructure`
2. `fix: Tune pendulum model physics for energy conservation tests`
3. `feat(phase2): Add force plate parsing pipeline (Guideline E5)`

---

## Acceptance Criteria Summary

| Guideline                         | Task      | Status         |
| --------------------------------- | --------- | -------------- |
| M2: Cross-engine infrastructure   | Phase 1   | ✅ Complete    |
| P3: Tolerance-based validation    | Phase 1   | ✅ Complete    |
| G1/G2: ZTCF/ZVCF tests            | Phase 1   | ✅ Complete    |
| F: Drift-control decomposition    | Phase 1   | ✅ Complete    |
| O3: Energy conservation <1% drift | Phase 1   | ✅ Complete    |
| E5: Ground reaction forces        | Phase 2.2 | ✅ Complete    |
| A1: C3D force-plate parsing       | Phase 2.2 | ✅ Complete    |
| B3: URDF visualization            | Phase 2.1 | ⏳ Not started |
