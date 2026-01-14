# Golf Modeling Suite — Implementation Plan (Jan 2026)

**Generated**: 2026-01-10  
**Status**: APPROVED FOR IMPLEMENTATION  
**Target**: Production-Ready Alignment with Project Design Guidelines

---

## Executive Summary

This implementation plan consolidates findings from Assessment Prompts A, B, and C into an actionable roadmap. It corrects several outdated assessment findings, identifies the genuine gaps, and prioritizes work into four phases over 12 weeks.

### Key Finding Corrections

The assessments contain several **stale findings** that do not reflect current repo state:

| Assessment Claim                                       | Actual Status                                                                                                                                       |
| ------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| OpenSim/MyoSuite ZTCF/ZVCF raise `NotImplementedError` | **CORRECTED**: Both engines implement `compute_ztcf()` and `compute_zvcf()` fully (Lines 413-503 in OpenSim, Lines 430-507 in MyoSuite)             |
| ZTCF/ZVCF missing from interface                       | **CORRECTED**: `PhysicsEngine` Protocol defines both methods (Lines 204-278 in interfaces.py)                                                       |
| Cross-engine validator missing                         | **CORRECTED**: `CrossEngineValidator` fully implemented in `shared/python/cross_engine_validator.py` with P3 tolerances and severity classification |

### Genuine Gaps Confirmed

1. **Cross-engine validation integration tests are skipped** (Lines 140, 151, 161, 181 in `test_cross_engine_validation.py`)
2. **URDF generator lacks embedded MuJoCo visualization** (placeholder only in `visualization_widget.py`)
3. **C3D force-plate parsing is not integrated** (analog channels exported, but no GRF/COP pipeline)
4. **Mobility/Force ellipsoid 3D visualization/export is incomplete**
5. **Flexible shaft and handedness are utilities, not engine-integrated**

---

## Current State Summary

### Repository Quality Baseline

- **CI Status**: 100% pass rate across 941+ tests
- **Readiness Rating**: 9.8/10 (AI-First Production Standard)
- **Code Hygiene**: Zero violations for Ruff, Black, and Mypy (fleet-wide)

### Feature × Engine Matrix (Verified)

| Feature            | MuJoCo | Drake | Pinocchio | OpenSim | MyoSuite |
| ------------------ | ------ | ----- | --------- | ------- | -------- |
| Forward Dynamics   | ✅     | ✅    | ✅        | ✅      | ✅       |
| Inverse Dynamics   | ✅     | ✅    | ✅        | ✅      | ✅       |
| Mass Matrix        | ✅     | ✅    | ✅        | ✅      | ✅       |
| Jacobians          | ✅     | ✅    | ✅        | ⚠️      | ⚠️       |
| ZTCF/ZVCF          | ✅     | ✅    | ✅        | ✅      | ✅       |
| Drift-Control      | ✅     | ✅    | ✅        | ✅      | ✅       |
| Muscle Integration | ⚠️     | -     | -         | ✅      | ✅       |

Legend: ✅ Full | ⚠️ Partial | - Not Applicable

---

## Phase 1: Cross-Engine Validation Activation (Week 1-2)

**Goal**: Satisfy Guideline M2/P3 — Automated cross-engine validation in CI

### Task 1.1: Implement Model Loading Infrastructure

**Priority**: CRITICAL  
**Effort**: 8 hours  
**Files**:

- `tests/integration/test_cross_engine_validation.py`
- `tests/fixtures/models/simple_pendulum.urdf` (new)
- `tests/fixtures/models/double_pendulum.urdf` (new)

**Actions**:

1. Create shared URDF fixtures for testing (simple pendulum, double pendulum)
2. Implement `load_test_model()` fixture function that:
   - Detects available engines
   - Loads model into each engine
   - Returns engine instances with identical initial states
3. Remove `pytest.skip()` from integration tests
4. Add deterministic seeding for all RNG-dependent tests

**Acceptance Criteria**:

- `test_forward_dynamics_mujoco_drake_agreement` passes with P3 tolerances
- `test_inverse_dynamics_mujoco_drake_agreement` passes with RMS < 10%
- `test_jacobian_mujoco_drake_agreement` passes with tolerance ±1e-8

### Task 1.2: Three-Way Engine Triangulation

**Priority**: HIGH  
**Effort**: 4 hours  
**Files**:

- `tests/integration/test_cross_engine_consistency.py`

**Actions**:

1. Implement pairwise validation for MuJoCo ↔ Drake ↔ Pinocchio
2. Use Pinocchio as tiebreaker when MuJoCo and Drake disagree
3. Log deviation sources with possible causes per P3

**Acceptance Criteria**:

- All engine pairs pass within P3 tolerances on simple pendulum
- CI workflow includes cross-engine validation step

### Task 1.3: Conservation Law Test Activation

**Priority**: HIGH  
**Effort**: 4 hours  
**Files**:

- `tests/integration/test_conservation_laws.py`

**Actions**:

1. Implement `test_energy_conservation_unforced` with inline pendulum model
2. Implement `test_momentum_conservation_free_floating`
3. Remove placeholder skips (Lines 33, 56, 76, 94, 114, 145)

**Acceptance Criteria**:

- Energy drift < 1% for 10-second conservative simulation
- Momentum conservation within ±1e-6 for isolated systems

---

## Phase 2: URDF Generator & C3D Integration (Week 3-6)

**Goal**: Satisfy Guidelines B3 (URDF visualization) and A1/E5 (force-plate parsing)

### Task 2.1: MuJoCo Visualization Embed

**Priority**: CRITICAL  
**Effort**: 16 hours  
**Files**:

- `tools/urdf_generator/visualization_widget.py`
- `tools/urdf_generator/mujoco_viewer.py` (new)

**Actions**:

1. Integrate MuJoCo passive viewer (`mujoco.viewer`) into Qt widget
2. Implement real-time URDF → MJCF conversion for preview
3. Add toggles for:
   - Collision geometry display
   - Frame/axis display
   - Joint limit visualization
4. Add physics sanity checks on edit (inertia positive-definiteness, joint axis normalization)

**Acceptance Criteria**:

- URDF changes reflect in 3D view within 100ms
- Collision and visual geometry toggleable independently
- Inertia validation errors shown inline

### Task 2.2: Force-Plate Parsing Pipeline

**Priority**: HIGH  
**Effort**: 12 hours  
**Files**:

- `engines/Simscape_Multibody_Models/3D_Golf_Model/python/src/c3d_reader.py`
- `shared/python/ground_reaction_forces.py` (exists, needs integration)

**Actions**:

1. Add `force_plate_dataframe()` method to `C3DDataReader`
2. Detect standard force-plate analog labels (Fx1, Fy1, Fz1, Mx1, My1, Mz1, etc.)
3. Compute COP trajectory from moment/force ratio
4. Integrate with existing `ground_reaction_forces.py` module
5. Add unit tests with synthetic force-plate C3D data

**Acceptance Criteria**:

- GRF components extracted from C3D analog channels
- COP trajectory computed and exportable
- Integration with kinetic analysis pipeline verified

### Task 2.3: Force-Plate Visualization

**Priority**: MEDIUM  
**Effort**: 8 hours  
**Files**:

- `apps/c3d_viewer.py`
- `shared/python/plotting.py`

**Actions**:

1. Add force vector overlay to 3D marker view
2. Add COP trajectory trace on ground plane
3. Add GRF component time-series plots

**Acceptance Criteria**:

- GRF vectors visible at force-plate locations
- COP trace shows temporal evolution

---

## Phase 3: Scientific Feature Completion (Week 7-10)

**Goal**: Satisfy Guidelines C1, C2, I, B5, B6

### Task 3.1: Jacobian Coverage Completion

**Priority**: HIGH  
**Effort**: 8 hours  
**Files**:

- `engines/physics_engines/opensim/python/opensim_physics_engine.py`
- `engines/physics_engines/myosuite/python/myosuite_physics_engine.py`

**Actions**:

1. Implement `compute_jacobian()` for OpenSim using SimTK MatterSubsystem
2. Verify MyoSuite Jacobian uses correct body ID resolution
3. Add tests for Jacobian shape (6×nv) and frame consistency

**Acceptance Criteria**:

- All 5 engines return spatial Jacobians for arbitrary bodies
- Jacobian format standardized: [Angular:3, Linear:3] × nv

### Task 3.2: Mobility/Force Ellipsoid Visualization

**Priority**: MEDIUM  
**Effort**: 12 hours  
**Files**:

- `shared/python/manipulability.py`
- `shared/python/plotting.py`
- `apps/analysis_gui.py`

**Actions**:

1. Add `compute_force_ellipsoid()` using Jacobian transpose
2. Implement 3D ellipsoid rendering in Matplotlib
3. Add export to OBJ/STL for external visualization
4. Integrate into Analysis GUI as toggleable overlay

**Acceptance Criteria**:

- Velocity and force ellipsoids rendered at task points
- Ellipsoids update with configuration changes
- Export generates valid mesh files

### Task 3.3: Flexible Shaft Engine Integration

**Priority**: MEDIUM  
**Effort**: 8 hours  
**Files**:

- `shared/python/flexible_shaft.py`
- `engines/physics_engines/mujoco/python/mujoco_humanoid_golf/physics_engine.py`
- `engines/physics_engines/drake/python/drake_physics_engine.py`

**Actions**:

1. Add `set_shaft_properties()` method to `PhysicsEngine` interface
2. Implement modal shaft model in MuJoCo (multi-body approximation)
3. Implement beam element shaft in Drake (compliant elements)
4. Add cross-engine validation for shaft deflection

**Acceptance Criteria**:

- Shaft flexibility affects clubhead trajectory
- Energy stored/released validates against beam theory
- Cross-engine deflection agreement within 10%

### Task 3.4: Handedness Integration

**Priority**: LOW  
**Effort**: 4 hours  
**Files**:

- `shared/python/handedness_support.py`
- `launchers/golf_launcher.py`

**Actions**:

1. Add handedness toggle to launcher UI
2. Apply mirroring transformation to trajectory data
3. Update configuration persistence

**Acceptance Criteria**:

- Left-handed mode mirrors all trajectories and visualizations
- Configuration persists across sessions

---

## Phase 4: Documentation & Acceptance Suite (Week 11-12)

**Goal**: Satisfy Guidelines M2 (acceptance tests), R1 (docs), R2 (assessment cycle)

### Task 4.1: Gold-Standard Acceptance Suite

**Priority**: CRITICAL  
**Effort**: 16 hours  
**Files**:

- `tests/acceptance/` (new directory)
- `tests/acceptance/test_pendulum_gold_standard.py`
- `tests/acceptance/test_closed_loop_gold_standard.py`

**Actions**:

1. Create minimal analytical models as ground truth
2. Implement acceptance tests per M2:
   - Simple pendulum (kinematics, dynamics, Jacobians)
   - Double pendulum (counterfactual deltas)
   - Closed-loop mechanism (constraint Jacobians)
3. Integrate into CI with required pass for merge

**Acceptance Criteria**:

- All engines match analytical solutions within 1e-8
- Cross-engine comparison automated in CI
- Indexed acceleration closure < 1e-6 rad/s²

### Task 4.2: API Documentation

**Priority**: MEDIUM  
**Effort**: 8 hours  
**Files**:

- `docs/api/` (new directory)
- `docs/tutorials/` (new directory)

**Actions**:

1. Generate Sphinx documentation from docstrings
2. Write 5 end-to-end tutorials:
   - Loading and simulating a model
   - Cross-engine validation workflow
   - Drift-control decomposition analysis
   - ZTCF/ZVCF counterfactual experiments
   - C3D ingestion and kinematic fitting

**Acceptance Criteria**:

- All public APIs documented with examples
- Tutorials executable as notebooks

### Task 4.3: Assessment Documentation Update

**Priority**: HIGH  
**Effort**: 4 hours  
**Files**:

- `docs/assessments/Assessment_A_Results.md`
- `docs/assessments/Assessment_B_Results.md`
- `docs/assessments/Assessment_C_Results.md`

**Actions**:

1. Update stale findings with current implementation status
2. Mark completed items with verification dates
3. Document remaining gaps with implementation timeline

**Acceptance Criteria**:

- All assessment claims verifiable against current code
- No outdated `NotImplementedError` claims

---

## Implementation Timeline

```
Week 1-2:   Phase 1 - Cross-Engine Validation Activation
Week 3-6:   Phase 2 - URDF Generator & C3D Integration
Week 7-10:  Phase 3 - Scientific Feature Completion
Week 11-12: Phase 4 - Documentation & Acceptance Suite
```

## Success Metrics

| Metric                       | Current          | Target            |
| ---------------------------- | ---------------- | ----------------- |
| Cross-engine tests passing   | 0 (skipped)      | 100%              |
| URDF generator visualization | Placeholder      | Full MuJoCo embed |
| Force-plate GRF pipeline     | None             | Complete          |
| Ellipsoid visualization      | Computation only | 3D + Export       |
| API documentation coverage   | ~40%             | 95%               |

## Risk Mitigation

1. **Engine availability in CI**: Use pytest markers and skipif decorators for optional engines
2. **MuJoCo viewer Qt integration**: Fallback to subprocess-based viewer if embed fails
3. **Force-plate C3D format variations**: Document supported formats, add validation warnings
4. **Timeline overrun**: Phase 4 can be deferred; Phase 1 is non-negotiable

---

## Appendix A: File Change Summary

### New Files (14)

- `tests/fixtures/models/simple_pendulum.urdf`
- `tests/fixtures/models/double_pendulum.urdf`
- `tools/urdf_generator/mujoco_viewer.py`
- `tests/acceptance/test_pendulum_gold_standard.py`
- `tests/acceptance/test_closed_loop_gold_standard.py`
- `docs/api/index.rst`
- `docs/tutorials/01_loading_models.md`
- `docs/tutorials/02_cross_engine_validation.md`
- `docs/tutorials/03_drift_control_decomposition.md`
- `docs/tutorials/04_counterfactual_experiments.md`
- `docs/tutorials/05_c3d_ingestion.md`
- `docs/assessments/Assessment_A_Results.md`
- `docs/assessments/Assessment_B_Results.md`
- `docs/assessments/Assessment_C_Results.md`

### Modified Files (15)

- `tests/integration/test_cross_engine_validation.py`
- `tests/integration/test_cross_engine_consistency.py`
- `tests/integration/test_conservation_laws.py`
- `tools/urdf_generator/visualization_widget.py`
- `engines/Simscape_Multibody_Models/3D_Golf_Model/python/src/c3d_reader.py`
- `engines/physics_engines/opensim/python/opensim_physics_engine.py`
- `engines/physics_engines/myosuite/python/myosuite_physics_engine.py`
- `shared/python/manipulability.py`
- `shared/python/plotting.py`
- `shared/python/flexible_shaft.py`
- `shared/python/handedness_support.py`
- `shared/python/interfaces.py`
- `apps/analysis_gui.py`
- `apps/c3d_viewer.py`
- `launchers/golf_launcher.py`

---

## Appendix B: Assessment Prompt Alignment

| Prompt | Focus Area                         | This Plan Addresses |
| ------ | ---------------------------------- | ------------------- |
| **A**  | Architecture, Product Requirements | Phases 1, 2, 4      |
| **B**  | Scientific Rigor (D-I)             | Phases 1, 3, 4      |
| **C**  | Cross-Engine Validation            | Phases 1, 4         |

---

_Generated by Antigravity Agentic Assistant_
