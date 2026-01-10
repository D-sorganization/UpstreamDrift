# Phase 1: Cross-Engine Validation Activation — Working Checklist

**Branch**: `feat/phase1-cross-engine-validation`  
**PR**: TBD  
**Session**: 2026-01-10

---

## Task 1.1: Model Loading Infrastructure

### Fixtures

- [ ] Create `tests/fixtures/models/` directory
- [ ] Create `simple_pendulum.urdf` — 1-DOF analytical reference
- [ ] Create `double_pendulum.urdf` — 2-DOF for counterfactual tests
- [ ] Add `conftest.py` with shared fixtures

### Model Loader

- [ ] Implement `load_test_model_mujoco()` fixture
- [ ] Implement `load_test_model_drake()` fixture
- [ ] Implement `load_test_model_pinocchio()` fixture
- [ ] Implement `available_engines()` fixture for dynamic detection

### Test Activation (test_cross_engine_validation.py)

- [ ] Remove `pytest.skip` from `test_forward_dynamics_mujoco_drake_agreement`
- [ ] Remove `pytest.skip` from `test_inverse_dynamics_mujoco_drake_agreement`
- [ ] Remove `pytest.skip` from `test_jacobian_mujoco_drake_agreement`
- [ ] Remove `pytest.skip` from `test_three_way_validation_mujoco_drake_pinocchio`
- [ ] Add proper skipif decorators for engine availability

---

## Task 1.2: Three-Way Engine Triangulation

### test_cross_engine_consistency.py Updates

- [ ] Implement actual pairwise validation logic (MuJoCo ↔ Drake)
- [ ] Implement actual pairwise validation logic (MuJoCo ↔ Pinocchio)
- [ ] Implement actual pairwise validation logic (Drake ↔ Pinocchio)
- [ ] Add Pinocchio as tiebreaker logic
- [ ] Remove remaining `pytest.skip` calls

---

## Task 1.3: Conservation Law Test Activation

### test_conservation_laws.py Updates

- [ ] Implement `test_energy_conservation_unforced` with inline pendulum
- [ ] Implement `test_momentum_conservation_free_floating`
- [ ] Implement `test_angular_momentum_no_external_torque`
- [ ] Implement `test_work_energy_theorem`
- [ ] Implement `test_indexed_acceleration_closure`
- [ ] Implement `test_symmetry_preservation`
- [ ] Remove all 6 `pytest.skip` placeholders

---

## Quality Gates

### Pre-Commit Verification

- [ ] `black .` passes
- [ ] `ruff check .` passes
- [ ] `mypy .` passes
- [ ] `pytest tests/integration/test_cross_engine_validation.py -v` passes
- [ ] `pytest tests/integration/test_cross_engine_consistency.py -v` passes
- [ ] `pytest tests/integration/test_conservation_laws.py -v` passes

### CI Verification

- [ ] Push to remote
- [ ] Create PR
- [ ] CI passes all checks

---

## Progress Log

| Time | Task                     | Status |
| ---- | ------------------------ | ------ |
|      | Branch created           |        |
|      | Fixtures created         |        |
|      | Test activation complete |        |
|      | Quality gates passed     |        |
|      | PR created               |        |

---

## Acceptance Criteria Summary

1. **Forward Dynamics**: MuJoCo ↔ Drake position agreement within ±1e-6 m
2. **Inverse Dynamics**: Torque RMS difference < 10%
3. **Jacobians**: Element-wise agreement within ±1e-8
4. **Energy Conservation**: Drift < 1% for 10-second simulation
5. **Indexed Acceleration Closure**: Residual < 1e-6 rad/s²
