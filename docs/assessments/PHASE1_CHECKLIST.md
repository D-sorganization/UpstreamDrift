# Phase 1: Cross-Engine Validation Activation — Working Checklist

**Branch**: `feat/phase1-cross-engine-validation`  
**PR**: [#347](https://github.com/D-sorganization/Golf_Modeling_Suite/pull/347)  
**Session**: 2026-01-10

---

## Task 1.1: Model Loading Infrastructure

### Fixtures

- [x] Create `tests/fixtures/models/` directory
- [x] Create `simple_pendulum.urdf` — 1-DOF analytical reference
- [x] Create `double_pendulum.urdf` — 2-DOF for counterfactual tests
- [x] Add `fixtures_lib.py` with shared fixtures

### Model Loader

- [x] Implement `mujoco_pendulum` fixture
- [x] Implement `drake_pendulum` fixture
- [x] Implement `pinocchio_pendulum` fixture
- [x] Implement `available_engines()` fixture for dynamic detection

### Test Activation (test_cross_engine_validation.py)

- [x] Remove `pytest.skip` from `test_forward_dynamics_mujoco_drake_agreement`
- [x] Remove `pytest.skip` from `test_inverse_dynamics_mujoco_drake_agreement`
- [x] Remove `pytest.skip` from `test_jacobian_mujoco_drake_agreement`
- [x] Remove `pytest.skip` from `test_three_way_validation_mujoco_drake_pinocchio`
- [x] Add proper skipif decorators for engine availability

---

## Task 1.2: Three-Way Engine Triangulation

### test_cross_engine_consistency.py Updates

- [x] Implement actual pairwise validation logic (MuJoCo ↔ Drake)
- [x] Implement actual pairwise validation logic (MuJoCo ↔ Pinocchio)
- [x] Implement actual pairwise validation logic (Drake ↔ Pinocchio)
- [x] Add Pinocchio as tiebreaker logic
- [x] Remove remaining `pytest.skip` calls

---

## Task 1.3: Conservation Law Test Activation

### test_conservation_laws.py Updates

- [x] Implement `test_energy_conservation_unforced` with inline pendulum
- [ ] **NEEDS PHYSICS TUNING** - test currently fails with 49.9% energy drift
- [x] Implement `test_work_energy_theorem`
- [x] Implement `test_drift_control_superposition`
- [x] Implement `test_ztcf_equals_drift`
- [x] Implement `test_zvcf_eliminates_coriolis`
- [x] Remove all `pytest.skip` placeholders

---

## Quality Gates

### Pre-Commit Verification

- [x] `black .` passes
- [x] `ruff check .` passes
- [x] `mypy .` passes
- [x] Unit tests (CrossEngineValidator) pass
- [ ] Integration tests (require engine tuning)

### CI Verification

- [x] Push to remote
- [x] Create PR #347
- [ ] CI passes all checks

---

## Progress Log

| Time  | Task                                       | Status |
| ----- | ------------------------------------------ | ------ |
| 14:19 | Branch created                             | ✅     |
| 14:30 | URDF fixtures created                      | ✅     |
| 14:45 | fixtures_lib.py implemented                | ✅     |
| 15:00 | test_cross_engine_validation.py rewritten  | ✅     |
| 15:15 | test_cross_engine_consistency.py rewritten | ✅     |
| 15:30 | test_conservation_laws.py rewritten        | ✅     |
| 15:45 | Integration conftest.py added              | ✅     |
| 16:00 | Quality gates passed                       | ✅     |
| 16:10 | Commit and push                            | ✅     |
| 16:15 | PR #347 created                            | ✅     |

---

## Known Issues / Follow-up Required

1. **Inline XML pendulum model needs physics tuning**
   - Energy drift of 49.9% in conservation test
   - Likely due to mass distribution or timestep
   - Needs verification of inertia tensor definitions

2. **CI may need pytest markers**
   - Add `@pytest.mark.mujoco` etc. for CI configuration

3. **Phase 2 Prerequisites**
   - URDF generator MuJoCo embed
   - Force-plate C3D integration

---

## Acceptance Criteria Summary

1. **Forward Dynamics**: ✅ Infrastructure ready, skips when engines unavailable
2. **Inverse Dynamics**: ✅ RMS comparison implemented
3. **Jacobians**: ✅ Element-wise comparison with ±1e-8 tolerance
4. **ZTCF/ZVCF**: ✅ Counterfactual tests implemented
5. **Energy Conservation**: ⚠️ Test exists but model needs tuning
