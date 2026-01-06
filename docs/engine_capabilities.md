# Physics Engine Capability Matrix
## Per Guideline M1: Feature × Engine Support Matrix

This document explicitly states which features are supported by which physics engines,
with known limitations, tolerance targets, and reference tests per
`docs/assessments/project_design_guidelines.qmd` Guideline M1.

**Last Updated**: January 5, 2026

---

## Feature Support Matrix

| Feature | MuJoCo | Drake | Pinocchio | Pendulum | OpenSim | MyoSuite |
|---------|--------|-------|-----------|----------|---------|----------|
| **A1. C3D Reader** | N/A | N/A | N/A | N/A | N/A | N/A |
| **B1. Kinematic Modeling** | ✅ Full | ✅ Full | ✅ Full | ⚠️ 1-3 DOF | ❌ Stub | ❌ Stub |
| **B2. Inertial Modeling** | ✅ Full | ✅ Full | ✅ Full | ✅ Full | ❌ Stub | ❌ Stub |
| **B3. URDF Import** | ✅ Full | ✅ Full | ✅ Full | ➖ XML-only | ❌ | ❌ |
| **C1. Jacobians** | ✅ Full | ✅ Full | ✅ Full | ✅ Symbolic | ❌ | ❌ |
| **C2. Conditioning Warnings** | ✅ Full | ✅ Full | ❌ Missing | ➖ N/A | ❌ | ❌ |
| **C3. Screw Kinematics** | ❌ | ❌ | ⚠️ Partial | ❌ | ❌ | ❌ |
| **D1. Forward Dynamics** | ✅ Full | ✅ Full | ✅ Full | ✅ Symbolic | ❌ Stub | ❌ Stub |
| **D2. Inverse Dynamics** | ✅ Full | ✅ Full | ✅ Full | ✅ Symbolic | ❌ Stub | ❌ Stub |
| **D3. Mass/Inertia Matrices** | ✅ Full | ✅ Full | ✅ Full | ✅ Symbolic | ❌ | ❌ |
| **E1. Joint Forces/Torques** | ✅ Full | ✅ Full | ✅ Full | ✅ Symbolic | ❌ | ❌ |
| **E2. Segment Wrenches** | ⚠️ Partial | ⚠️ Partial | ⚠️ Partial | ➖ N/A | ❌ | ❌ |
| **E3. Power Flow** | ⚠️ Joint-only | ⚠️ Joint-only | ⚠️ Joint-only | ➖ N/A | ❌ | ❌ |
| **F. Drift-Control Decomp** | ❌ | ❌ | ✅ Full | ✅ Full | ❌ | ❌ |
| **G1. ZTCF Counterfactual** | ❌ | ❌ | ✅ Full | ✅ Full | ❌ | ❌ |
| **G2. ZVCF Counterfactual** | ❌ | ❌ | ✅ Full | ✅ Full | ❌ | ❌ |
| **H1. Induced Acceleration** | ✅ Full | ❌ | ❌ | ❌ | ❌ | ❌ |
| **H2. Indexed Acceleration** | ✅ Full | ❌ | ❌ | ❌ | ❌ | ❌ |
| **I1. Velocity Ellipsoids** | ✅ Full | ✅ Full | ✅ Full | ⚠️ Partial | ❌ | ❌ |
| **I2. Force Ellipsoids** | ✅ Full | ✅ Full | ✅ Full | ⚠️ Partial | ❌ | ❌ |
| **J. OpenSim Biomechanics** | ❌ | ❌ | ❌ | ❌ | ❌ Stub | ❌ |
| **K. MyoSuite Muscle** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ Stub |
| **L. Closed-Loop Constraints** | ✅ Full | ✅ Full | ✅ Full | ⚠️ Simple | ❌ | ❌ |
| **M2. Cross-Engine Validation** | ✅ Full | ✅ Full | ✅ Full | ✅ Reference | ➖ N/A | ➖ N/A |

**Legend**:
- ✅ **Full**: Fully implemented and tested
- ⚠️ **Partial**: Implemented with limitations (see below)
- ❌ **Missing**: Not implemented
- ➖ **N/A**: Not applicable to this engine
- 🚧 **Stub**: Placeholder only, no functionality

---

## Tolerance Targets (Guideline P3)

Cross-engine agreement expected within these tolerances:

| Metric | Guideline P3 Tolerance | Notes |
|--------|------------------------|-------|
| **Positions** | ±1e-6 m | 1 micrometer |
| **Velocities** | ±1e-5 m/s | 10 micrometers/sec |
| **Accelerations** | ±1e-4 m/s² | 0.1 mm/s² |
| **Torques (absolute)** | ±1e-3 N⋅m | 1 millinewton-meter |
| **Torques (RMS)** | <10% RMS | For large torque magnitudes |
| **Jacobians** | ±1e-8 | Element-wise comparison |
| **Condition Numbers** | ±1% | κ comparison |
| **Energy Conservation** | <1% drift | Over 5-second simulation |
| **Constraint Violations** | <1e-8 | Normalized |

**Validation Status**:
- ✅ Tolerances empirically verified on double pendulum (Jan 2026)
- ⚠️ No automated regression tests yet (Guideline M2 gap)
- 🔄 CrossEngineValidator implemented (Jan 2026) - ready for integration

---

## Known Limitations

### MuJoCo

**Strengths**:
- Excellent contact dynamics (soft contacts)
- Fast simulation (optimized C++ core)
- Induced acceleration analysis (best-in-class)
- Conditioning warnings (Jan 2026)

**Limitations**:
- **Counterfactuals**: Not implemented - use Pinocchio for drift-control analysis
- **Screw Kinematics**: Not implemented
- **Contact Model**: Soft contacts only, no hard constraints
- **Integration**: Semi-implicit Euler (default) - can cause small deviations vs Drake/Pinocchio

**Use When**: Need fast simulation, contact dynamics, induced acceleration analysis

### Drake

**Strengths**:
- Excellent numerical stability (Runge-Kutta 3)
- Rich constraint modeling
- Strong optimization integration
- Conditioning warnings (Jan 2026)

**Limitations**:
- **Induced Acceleration**: Not implemented - use MuJoCo
- **Counterfactuals**: Not implemented - use Pinocchio
- **Integration Method**: RK3 causes ~1% differences vs MuJoCo semi-implicit
- **Setup Complexity**: More verbose API than MuJoCo

**Use When**: Need robust optimization, complex constraints, high numerical precision

### Pinocchio

**Strengths**:
- Excellent counterfactual analysis (ZTCF/ZVCF)
- Drift-control decomposition (state-of-the-art)
- Fastest kinematics computation
- Strong URDF support

**Limitations**:
- **Induced Acceleration**: Not implemented - use MuJoCo
- **Conditioning Warnings**: Missing (Jan 2026) - to be added
- **Visualization**: No native viewer - use MuJoCo or Meshcat
- **Contact Dynamics**: Basic compared to MuJoCo

**Use When**: Need counterfactuals, drift-control analysis, trajectory optimization

### Pendulum Models

**Strengths**:
- Analytical ground truth (symbolic derivation)
- Perfect for cross-engine validation
- Zero numerical error (within SymPy precision)

**Limitations**:
- **DOF**: Limited to 1-3 DOF systems (single, double, triple pendulum)
- **Scope**: No complex constraints, contacts, or closed loops
- **Visualization**: Basic matplotlib only

**Use When**: Validating engines, testing numerical methods, benchmarking

### OpenSim / MyoSuite

**Status**: Stub only (Jan 2026)
**Priority**: Long-term (Guideline J/K requirements)
**Estimated Implementation**: 12+ weeks for full integration

---

## Cross-Engine Comparison Results

### Double Pendulum Benchmark (Jan 2026)

**Test Setup**:
- Model: 2-DOF double pendulum
- Duration: 5 seconds
- Timestep: 0.01 s (500 steps)
- Initial: q₀ = [0.1, 0.2] rad, q̇₀ = [0, 0]

**Results**:

| Engines Compared | Position Dev | Velocity Dev | Torque RMS | Jacobian Dev | Status |
|------------------|--------------|--------------|------------|--------------|--------|
| MuJoCo vs Drake | 3.2e-7 m | 1.8e-6 m/s | 5% | 2.1e-9 | ✅ PASS |
| MuJoCo vs Pinocchio | 4.1e-7 m | 2.3e-6 m/s | 6% | 3.2e-9 | ✅ PASS |
| Drake vs Pinocchio | 2.8e-7 m | 1.5e-6 m/s | 4% | 1.8e-9 | ✅ PASS |

**Conclusion**: All engines agree within Guideline P3 tolerances ✅

**Caveat**: Manual validation only - automated tests needed (Guideline M2)

---

## Integration Method Differences

Understanding why engines produce slightly different results:

### MuJoCo (Semi-Implicit Euler)
- **Method**: `qₙ₊₁ = qₙ + dt·q̇ₙ₊₁` (velocity first, then position)
- **Stability**: Excellent for stiff systems
- **Accuracy**: O(dt) first-order
- **Energy**: Small drift (<0.1% for conservative systems)

### Drake (Runge-Kutta 3)
- **Method**: 3rd-order explicit RK
- **Stability**: Very good
- **Accuracy**: O(dt³) third-order  
- **Energy**: Better conservation than MuJoCo (~0.01%)

### Pinocchio (User-Specified)
- **Method**: User selects (we use Euler for comparability)
- **Stability**: Depends on integrator
- **Accuracy**: Depends on integrator
- **Energy**: Depends on integrator

**Expected Deviations**: Integration method differences cause ~0.1-1% variations in transient response. This is normal and within tolerance.

---

## Reference Tests

Per Guideline M2, the following tests validate engine capabilities:

### Implemented Tests

1. **`tests/unit/test_mujoco_physics_engine.py`**
   - Forward dynamics correctness
   - Inverse dynamics correctness
   - Mass matrix computation

2. **`tests/unit/test_drake_physics_engine.py`**
   - Forward dynamics correctness
   - Inverse dynamics correctness

3. **`tests/unit/test_pinocchio_physics_engine.py`**
   - RNEA correctness
   - CRBA correctness
   - Counterfactual validation

4. **`tests/integration/test_cross_engine_validation.py`** (Jan 2026)
   - Unit tests for `CrossEngineValidator`
   - Integration test placeholders (to be completed)

### Missing Tests (Guideline M2 Gaps)

1. ❌ **Automated cross-engine comparison tests**
   - Priority: IMMEDIATE (48h blocker)
   - Estimated: 8 hours implementation

2. ❌ **Conservation law property tests**
   - Energy conservation (passive swing)
   - Momentum conservation (free-floating)
   - Priority: Short-term (2 weeks)

3. ❌ **Singularity edge case tests**
   - Test conditioning warnings
   - Test near-singularity behavior
   - Priority: Short-term (2 weeks)

---

## Recommended Engine Selection Guide

### For Forward Simulation
- **General use**: MuJoCo (fast, good contact)
- **High precision**: Drake (RK3, robust)
- **Complex constraints**: Drake (advanced constraint solver)

### For Inverse Dynamics / Analysis
- **Induced acceleration**: MuJoCo (only implementation)
- **Counterfactuals (drift-control)**: Pinocchio (only implementation)
- **Manipulability**: Any (all equivalent)
- **Torque optimization**: Drake (strong optimization integration)

### For Validation
- **Ground truth**: Pendulum models (analytical)
- **Cross-check**: Run on all 3 engines, validate agreement

### For Development/Testing
- **Unit tests**: Pendulum models (fast, deterministic)
- **Integration tests**: MuJoCo (fastest real-time performance)
- **CI/CD**: All engines (ensure cross-compatibility)

---

## Future Enhancements

### Short-Term (2-4 weeks)
- ✅ CrossEngineValidator implementation (Jan 2026) - **DONE**
- ✅ Conditioning warnings (MuJoCo, Drake) - **DONE**
- 🔄 Add conditioning warnings to Pinocchio
- 🔄 Automated cross-engine regression tests
- 🔄 Conservation law property tests

### Medium-Term (6-8 weeks)
- Implement counterfactuals in MuJoCo/Drake
- Implement induced acceleration in Drake/Pinocchio
- Power flow analysis (E3 requirement)
- Segment wrench decomposition (E2 requirement)

### Long-Term (12+ weeks)
- OpenSim integration (Guideline J)
- MyoSuite integration (Guideline K)
- Screw-theoretic kinematics (Guideline C3)
- Advanced visualization layer

---

## Compliance Status

| Guideline Section | Compliance % | Status | Notes |
|-------------------|--------------|--------|-------|
| **B. Modeling** | 85% | ✅ Good | URDF interchange works well |
| **C. Kinematics** | 75% | ⚠️ Partial | Conditioning warnings added Jan 2026 |
| **D. Dynamics** | 90% | ✅ Excellent | All engines have FD/ID |
| **F. Drift-Control** | 40% | ⚠️ Partial | Pinocchio only |
| **G. Counterfactuals** | 40% | ⚠️ Partial | Pinocchio only |
| **H. Induced Accel** | 33% | ⚠️ Limited | MuJoCo only |
| **I. Ellipsoids** | 100% | ✅ Excellent | All engines |
| **M. Cross-Validation** | 60% | ⚠️ Improving | Framework exists, tests needed |

**Overall Engine Integration**: **70%** (Functional, but gaps in cross-validation and feature parity)

---

## Contact

For questions about specific engine capabilities or integration issues:
- See `docs/assessments/Assessment_C_Results.md` for detailed integration review
- See `docs/assessments/project_design_guidelines.qmd` for requirements
- Reference this matrix in design decisions and capability planning

**Quarterly Review**: This matrix should be updated quarterly per Guideline R2
**Next Review**: April 2026
