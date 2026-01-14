# Implementation Plan: Assessment B High-Priority Fixes

**Date:** 2026-01-05  
**Goal:** Implement all high-priority fixes from Assessment B, emphasizing alignment with project design guidelines.

## Priority Ranking (Based on Assessment B)

### Tier 1: BLOCKER (Immediate - 48 hours)
- [x] **B-002**: Analytical RNE (ALREADY IMPLEMENTED in kinematic_forces.py:303-337)
- [ ] **B-001**: Deprecate/fix `compute_centripetal_acceleration()` - upgrade warning to NotImplementedError
- [ ] **B-004**: Extract magic numbers to named constants with units and sources
- [ ] **B-006**: Add comprehensive unit documentation (module-level and function-level)

### Tier 2: Critical (Short-term - 2 weeks)
- [ ] **B-003**: Document frame conventions and add coordinate system documentation
- [ ] **B-005**: Add energy conservation tests
- [ ] **B-007**: Document epsilon selection with analysis
- [ ] **B-008**: Add singularity detection for effective mass

### Tier 3: Performance & Polish (Long-term - 6 weeks)
- [ ] Second-order central difference for Jacobian derivative
- [ ] Spatial algebra layer (SE(3)/se(3))
- [ ] C++ extensions for hot paths

## Alignment with Project Guidelines

### Guideline N (Code Quality & CI/CD Gates)
- N2 (Type Safety): All fixes maintain strict type annotations
- N4 (Security & Safety): Replace magic numbers, add physical plausibility checks

### Guideline O (Physics Engine Integration)
- O3 (Numerical Stability): Document tolerances, add singularity detection
- O2 (State Isolation): Already implemented via MjDataContext and _perturb_data

### Guideline R (Documentation & Knowledge Management)
- R1 (Docstring Standards): Add units to all function signatures
- R2 (Adversarial Review): This plan itself is part of the review cycle

## Implementation Steps

### Step 1: Centralize Numerical Constants (B-004)
Create `shared/python/numerical_constants.py` with documented constants:
- EPSILON_FINITE_DIFF = 1e-6  # Finite difference step size
- EPSILON_SINGULARITY = 1e-10  # Singularity detection threshold
- EPSILON_ZERO_VELOCITY = 1e-6  # Zero velocity detection
- etc.

### Step 2: Upgrade Centripetal Warning to Error (B-001)
Replace warning in `compute_centripetal_acceleration()` with `NotImplementedError` to prevent accidental use.

### Step 3: Add Module-Level Unit Documentation (B-006)
Expand module docstring in `kinematic_forces.py` with comprehensive unit conventions.

### Step 4: Add Function-Level Unit Annotations (B-006)
Update all public methods to include units in docstrings:
```python
Args:
    qpos: Joint positions [nv] (rad for revolute, m for prismatic)
    qvel: Joint velocities [nv] (rad/s for revolute, m/s for prismatic)
Returns:
    Forces [nv] (N·m for revolute, N for prismatic)
```

### Step 5: Frame Convention Documentation (B-003)
Add explicit documentation of:
- World frame convention (Z-up, right-handed)
- Body frame conventions
- Jacobian frame mappings

### Step 6: Energy Conservation Tests (B-005)
Create `tests/integration/test_energy_conservation.py` with tests for:
- Free fall energy conservation
- Work-energy theorem
- Power balance (dE/dt = P_in - P_out)

### Step 7: Epsilon Analysis (B-007)
Add comments documenting why each epsilon value was chosen, with references to condition number analysis or literature sources.

### Step 8: Singularity Detection (B-008)
Add condition number monitoring to `compute_effective_mass()` and meaningful error messages when approaching singularities.

## Success Criteria

1. All T1 fixes implemented and tested
2. CI/CD passes without new errors
3. Coverage maintained or improved
4. All functions have unit documentation
5. Energy conservation tests pass
6. No magic numbers in physics calculations

## Notes

- Assessment A fixes (state isolation) already completed
- RNE optimization (B-002) already implemented
- Focus on scientific rigor and documentation per project guidelines
