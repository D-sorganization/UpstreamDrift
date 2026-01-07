# Golf Modeling Suite — Consolidated Remediation Roadmap  
## Based on Assessments A, B, C (January 2026)

**Document Date:** 2026-01-06  
**Baseline Assessments:**
- Assessment A: Architecture & Software Patterns
- Assessment B: Scientific Rigor & Numerical Correctness  
- Assessment C: Cross-Engine Validation & Integration

**Current Status**: **Production-ready architecture with incomplete feature scope**

---

## Executive Summary

### Overall Project Health

**STRENGTHS** (What's Working):
- ✅ **Principal-level Python architecture** (clean separation, protocol-based design)
- ✅ **Exceptional type safety** (Mypy strict mode, comprehensive type hints)
- ✅ **Best-in-class cross-engine validation** (P3 tolerance compliance)
- ✅ **Rigorous numerical foundations** (indexed acceleration closure verification)
- ✅ **Production-grade CI/CD** (Black/Ruff/Mypy gates, 60% coverage target)

**CRITICAL GAPS** (Blocking Production Release):
| Gap Category | Issue | Impact | Priority |
|--------------|-------|--------|----------|
| **Product Completeness** | ZTCF/ZVCF counterfactuals missing (Guideline G) | Cannot answer "Definition of Done" #3 | BLOCKER |
| **Product Completeness** | Mobility/Force ellipsoids 20% implemented (Guideline I) | Cannot answer "Definition of Done" #4 | BLOCKER |
| **Scientific Validation** | Zero analytical benchmarks | Cannot prove equations are correct | BLOCKER |
| **Integration** | OpenSim/MyoSuite non-functional | Advertised biomechanics features unusable | BLOCKER |
| **Observability** | No runtime energy/singularity warnings | Silent failures possible | CRITICAL |

### Can We Ship Today?

**SHORT ANSWER: NO (for advertised scope), YES (for reduced scope)**

**Shipping Criteria**:
- ✅ **CAN SHIP FOR**: Pure multibody dynamics (MuJoCo/Drake/Pinocchio)
  - Forward/Inverse dynamics: Excellent
  - Cross-engine validation: Production-ready
  - Drift-control decomposition: Rigorous

- ❌ **CANNOT SHIP FOR**: Biomechanics research (advertised in guidelines)
  - Counterfactuals (G1/G2): Not implemented
  - Ellipsoid visualization (I): Incomplete
  - OpenSim muscles (J): Stubs only
  - MyoSuite neural control (K): Non-functional

**RECOMMENDATION**: Either (1) **Reduce scope** to multibody dynamics and ship after 48h fixes, OR (2) **Implement missing features** over 6 weeks and ship complete product.

---

## Prioritized Remediation Plan

### Phase 1: Critical Blockers (48 Hours)

#### Goal: Unblock counterfactual experiments + add critical safety monitors

**Hour 0-8: ZTCF/ZVCF Implementation (Assessment A Finding F-001)**
- Add `compute_ztcf()` and `compute_zvcf()` to `PhysicsEngine` protocol
- Implement in PendulumPhysicsEngine (2h, simplest for validation)
- Implement in MuJoCoPhysicsEngine (2h)
- Implement in DrakePhysicsEngine (2h)
- Implement in PinocchioPhysicsEngine (2h)
- **Deliverable**: `engine.compute_ztcf(q, v)` functional in all engines

**Hour 8-12: Jacobian Conditioning Warnings (Assessment A Finding F-004)**
- Create `shared/python/manipulability.py` with `check_jacobian_conditioning()`
- Add `get_jacobian_conditioning()` to PhysicsEngine protocol
- Integrate warnings into CrossEngineValidator (κ > 1e6 → warning, κ > 1e10 → error)
- **Deliverable**: Singularity warnings active

**Hour 12-18: Cross-Engine Validator Severity Thresholds (Assessment C Finding C-003)**
- Add WARNING (2× tol), ERROR (10× tol), BLOCKER (100× tol) classifications
- Update logging to use severity levels
- **Deliverable**: Users know when deviation is "acceptable" vs "investigate" vs "blocker"

**Hour 18-24: Critical Documentation (Assessment C Finding C-005)**
- Create `docs/troubleshooting/cross_engine_deviations.md`
- Document integration method differences (MuJoCo vs Drake vs Pinocchio)
- Add quick reference for common deviation causes
- **Deliverable**: Users can self-diagnose deviations

**Hour 24-36: Energy Conservation Monitoring (Assessment B Finding B-006)**
- Add `get_total_energy()` to PhysicsEngine protocol
- Create `ConservationMonitor` class with drift warnings (>1% triggers error)
- Integrate into launchers (optional real-time energy plot)
- **Deliverable**: Energy drift >1% logged with corrective advice

**Hour 36-42: Input Validation Layer (Assessment A Finding F-005)**
- Add `@validate_physical_bounds` decorator
- Validate: mass > 0, dt > 0, joint limits min < max, inertia PD
- **Deliverable**: `engine.set_mass(m=-1.0)` raises `ValueError` immediately

**Hour 42-48: Pendulum Analytical Tests (Assessment B Finding B-001 — Quick Win)**
- Create `tests/analytical/test_pendulum_lagrangian.py`
- Analytical τ = mgl sin(θ) + Iθ̈ comparison (closed-form)
- **Deliverable**: First analytical ground truth validation

**PHASE 1 SUCCESS CRITERIA**:
- ✅ ZTCF/ZVCF operational (addresses Definition of Done #3)
- ✅ Singularity warnings active (prevents silent failures)
- ✅ Energy drift monitored (ensures integration quality)
- ✅ First analytical test (proves equations correct)

---

### Phase 2: Structural Improvements (2 Weeks)

#### Goal: Complete scientific validation infrastructure + ellipsoid visualization

**Week 1: Scientific Verification**

**Days 1-2: Analytical Benchmark Suite (Assessment B Finding B-001)**
- Implement 5 analytical tests:
  1. Simple pendulum (Lagrangian comparison)
  2. Double pendulum (symbolic SymPy derivation)
  3. Free fall (y = y₀ + v₀t - 0.5gt²)
  4. Rotating rigid body (Euler's equations)
  5. Two-link arm (Jacobian closed-form)
- **Deliverable**: `tests/analytical/` with mathematical ground truth

**Days 3-4: Dimensional Analysis Integration (Assessment B Finding B-003)**
- Integrate `pint` library into `PhysicsEngine` protocol
- Refactor all public APIs to use `pint.Quantity` (force [N], torque [N·m], etc.)
- Add runtime unit validation (catches radians/degrees mix-ups)
- **Deliverable**: `compute_torque(force_newtons, lever_meters)` auto-validated

**Day 5: Unit/Magic Number Audit (Assessment B Findings B-005, B-007)**
- Replace all float equality checks (`if theta == 0.0`) with `np.isclose()`
- Extract hardcoded constants to `physical_constants.py` with sources
- Document: gravity = 9.80665 m/s² (NIST CODATA 2018)
- **Deliverable**: Zero magic numbers, all constants cited

**Week 2: Ellipsoid Visualization + Diagnostics**

**Days 6-8: Mobility/Force Ellipsoid Visualization (Assessment A Finding F-002)**
- Design `EllipsoidVisualizer` API (interface design session)
- Implement 3D rendering using `meshcat` (web-based, mature library)
- Integrate into MuJoCo viewer pipeline
- Export ellipsoid parameters as JSON for reproducibility
- **Deliverable**: `visualizer.render_ellipsoid(J, frame="clubhead")` displays in 3D

**Days 9-10: Simulation Diagnostics Infrastructure (Assessment C Finding M3)**
- Create `shared/python/diagnostics.py` with `SimulationDiagnostics` class
- Implement checks: Jacobian conditioning, constraint rank, force sanity, energy drift
- Integrate into launchers (real-time health dashboard)
- **Deliverable**: Users see warnings before catastrophic failure

**Days 11-12: Nightly Cross-Engine CI (Assessment C Finding C-002)**
- GitHub Actions workflow: run cross-engine tests on 10 reference motions
- Email team if deviation > 2× tolerance (early drift detection)
- **Deliverable**: Automated continuous validation (no manual testing)

**Days 13-14: Provenance & Versioned Exports (Assessment A Findings F-007, F-008)**
- Create `SimulationContext` dataclass (constraints, integrator, timestep)
- Add schema versioning to all JSON/NPZ exports (version, timestamp, git SHA)
- **Deliverable**: Perfect reproducibility ("result from commit abc123, MuJoCo 3.3.1, seed 42")

**PHASE 2 SUCCESS CRITERIA**:
- ✅ Analytical tests prove mathematical correctness (not just cross-engine agreement)
- ✅ Dimensional analysis prevents unit errors
- ✅ Ellipsoid visualization operational (addresses Definition of Done #4)
- ✅ Automated CI catches cross-engine drift
- ✅ All exports versioned for reproducibility

---

### Phase 3: Biomechanics Integration (6 Weeks — IF SCOPE REQUIRES)

#### Goal: Unlock biomechanical muscle analysis capabilities

**Weeks 1-3: OpenSim Integration (Assessment A Finding F-003, Assessment C Finding C-001)**
- Week 1: Vendor opensim-core Python bindings, write integration tests
- Week 2: Implement Hill muscle model adapter in `opensim_physics_engine.py`
  - Activation → force mapping
  - Tendon compliance
  - Muscle wrapping geometry (baseline)
- Week 3: Cross-validate muscle forces vs published OpenSim benchmarks (ISB standards)
- **Deliverable**: `opensim_engine.compute_muscle_forces(activation)` functional

**Weeks 4-5: MyoSuite Neural Control (Assessment C Finding C-008)**
- Week 4: Integrate myosuite environment, add RL policy hooks
- Week 5: Implement hybrid torque/muscle models, comparative analysis tooling
- **Deliverable**: `myosuite_engine.run_policy(neural_controller)` operational

**Week 6: Power Flow & Inter-Segment Wrenches (Assessment A, Guideline E3)**
- Implement `shared/python/power_flow.py` (inter-segment power transfer analysis)
- Add wrench visualization (arrows) to 3D viewer
- Export power flow as time-series CSV
- **Deliverable**: Users can visualize energy transfer between segments

**PHASE 3 SUCCESS CRITERIA**:
- ✅ OpenSim muscle forces validated against published benchmarks
- ✅ MyoSuite RL policies functional
- ✅ Full 6-engine validation operational (MuJoCo/Drake/Pinocchio/Pendulum/OpenSim/MyoSuite)
- ✅ All "Definition of Done" questions answerable

---

## Gap Analysis: Design Guidelines vs. Implementation

### Feature Compliance Matrix

| Guideline | Requirement | Status | Gap | Remediation | Effort |
|-----------|-------------|--------|-----|-------------|--------|
| **A1-A3** | C3D Ingestion | ✅ 90% | Force plate parsing marked optional | Make mandatory per S requirements | 4h |
| **B1-B4** | Modeling/Interoperability | ✅ 100% | - | - | - |
| **C1-C3** | Jacobians/Kinematics | ✅ 95% | Conditioning warnings missing | Phase 1 (manipulability.py) | 4h |
| **D1-D3** | Dynamics Core| ⚠️ 65% | Control-only toggle missing | Add counterfactuals (Phase 1) | 8h |
| **E1-E3** | Forces/Wrenches | ⚠️ 50% | Power flow missing | Phase 3 (power_flow.py) | 12h |
| **F** | Drift-Control Decomposition | ✅ 100% | - | - | - |
| **G1-G2** | ZTCF/ZVCF Counterfactuals | ❌ 0% | Complete feature missing | Phase 1 (BLOCKER) | 8h |
| **H1-H2** | Indexed Acceleration | ⚠️ 70% | Muscle-driven IAA missing | Phase 3 (requires OpenSim) | 12h |
| **I** | Mobility/Force Ellipsoids | ⚠️ 20% | Visualization/exports missing | Phase 2 (BLOCKER) | 16h |
| **J** | OpenSim Biomechanics | ❌ 0% | Complete feature missing | Phase 3 (if scope requires) | 80h |
| **K** | MyoSuite Neural Control | ❌ 0% | Complete feature missing | Phase 3 (if scope requires) | 60h |
| **L** | Visualization | ⚠️ 30% | Ellipsoids/wrenches missing | Phase 2 + 3 | 28h |
| **M1-M3** | Cross-Engine Validation | ✅ 95% | Diagnostic gaps | Phase 2 (diagnostics.py) | 12h |
| **N1-N4** | Code Quality Gates | ✅ 100% | - | - | - |
| **O1-O3** | Engine Integration | ✅ 100% | - | - | - |
| **P1-P3** | Data Handling | ✅ 100% | - | - | - |
| **Q1-Q3** | GUI Standards | ⚠️ 40% | Analysis bundles missing | Phase 2 (versioned exports) | 8h |
| **R1-R3** | Documentation | ✅ 100% | - | - | - |
| **S1-S3** | Motion Matching | ⚠️ 30% | Trajectory optimization missing | Future work (not critical) | 40h |

**SUMMARY**:
- **Fully Implemented**: 8/19 categories (42%)
- **Partially Implemented**: 8/19 categories (42%)
- **Not Implemented**: 3/19 categories (16%)

**If Biomechanics Scope Removed** (drop J, K, partial H, partial S):
- **Fully Implemented**: 8/15 categories (53%)
- **Phase 1+2 Completion**: 14/15 categories (93%)

---

## Definition of Done: Can We Answer These Questions?

Per project guidelines, a release is acceptable only if it can answer:

### 1. What moved? (Kinematics)
**STATUS**: ✅ **YES**
- C3D reader operational
- Marker mapping functional
- Jacobians computed correctly

### 2. What caused it? (Indexed + Induced Acceleration)
**STATUS**: ✅ **YES**
- Indexed acceleration with closure verification
- Drift-control decomposition rigorous
- Missing: Muscle-driven IAA (requires Phase 3 OpenSim)

### 3. What could have happened instead? (Null Space + Counterfactuals)
**STATUS**: ❌ **NO** (BLOCKER)
- Null-space analysis missing (add to Phase 2)
- ZTCF/ZVCF counterfactuals missing (Phase 1 CRITICAL)
- **IMPACT**: Cannot isolate torque-attributed effects from passive dynamics

### 4. What was controllable? (Mobility/Force Ellipsoids)
**STATUS**: ❌ **NO** (BLOCKER)
- Ellipsoid dataclasses exist but no visualization/exports (Phase 2 CRITICAL)
- **IMPACT**: Cannot inspect manipulability or force transmission capabilities

### 5. What assumptions mattered? (Constraints + Inertias + Actuation Model)
**STATUS**: ⚠️ **PARTIAL**
- Constraint visualization exists
- Inertia parameters documented
- Missing: Provenance tracking (Phase 2)
- **IMPACT**: Can answer manually but not automated

**VERDICT**: Currently answers **2 of 5 questions**. After Phase 1+2: **5 of 5 questions**.

---

## Project Scope Commitment

**NON-NEGOTIABLE**: Full guideline compliance (19/19 categories)

This project exists at the **interface between biomechanics and robotics**. We do not reduce scope to meet deadlines. Every capability in `docs/project_design_guidelines.qmd` is essential to the mission:

- ✅ **Biomechanics** (OpenSim Hill muscles, MyoSuite neural control)
- ✅ **Robotics** (MuJoCo/Drake/Pinocchio multi-engine validation)
- ✅ **Scientific rigor** (Counterfactuals, analytical validation, conservation laws)
- ✅ **Interpretability** (Ellipsoids, drift-control decomposition, indexed acceleration)

### Implementation Timeline (Full Scope)

**TOTAL EFFORT**: **8.5 weeks** to production-ready biomechanics + robotics platform

**IMPLEMENT**:
- **Phase 1** (48 hours): Critical blockers (ZTCF/ZVCF, singularity warnings, energy monitors)
- **Phase 2** (2 weeks): Scientific validation infrastructure (analytical tests, ellipsoids, nightly CI)
- **Phase 3** (6 weeks): Biomechanics integration (OpenSim + MyoSuite, the core differentiator)

**SHIPPING CRITERIA**:
- ✅ Full guideline compliance (19/19 categories)
- ✅ 6-engine validation (MuJoCo/Drake/Pinocchio/Pendulum/OpenSim/MyoSuite)
- ✅ Muscle-driven motion analysis operational
- ✅ Neural control policy experiments functional
- ✅ All "Definition of Done" questions answerable

**TARGET USERS**: Biomechanics researchers, sports science labs, robotics-biomechanics interface investigators

**COMMITMENT**: We ship when it's **right**, not when it's **fast**.

---

## Risk Mitigation

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Phase 1 ZTCF implementation reveals fundamental engine limitation** | Low (20%) | BLOCKER | Start with Pendulum (analytical ground truth), validate before complex models |
| **Ellipsoid visualization performance poor for 15-DOF model** | Moderate (40%) | MAJOR | Use meshcat (web-based, async rendering), add LOD (level of detail) for complex models |
| **OpenSim integration hits API incompatibility** | High (60%) | BLOCKER | Prototype in Week 1 of Phase 3, pivot to alternative if blocked |
| **Cross-engine CI introduces flaky tests** | Moderate (30%) | MAJOR | Use fixed random seeds, version-control reference data, add retry logic |

### Scientific Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Analytical tests reveal bug in core dynamics** | Low (10%) | CRITICAL | Good news: catch before publication! Fix and re-validate |
| **Dimensional analysis breaks existing code** | High (50%) | MAJOR | Implement incrementally, use feature flags for gradual rollout |
| **Energy conservation monitoring too strict (false positives)** | Moderate (30%) | MINOR | Make threshold configurable (default 1%, allow user override to 5%) |

---

## Success Metrics

### Phase 1 (48h) Success Metrics
- ✅ All 4 engines implement ZTCF/ZVCF (pytest green)
- ✅ Zero Jacobian singularities undetected in test suite (warning coverage)
- ✅ Documentation reads: "Cross-engine deviation troubleshooting guide available"

### Phase 2 (2w) Success Metrics
- ✅ 5 analytical tests pass (pendulum, double pendulum, free fall, rigid body, two-link)
- ✅ Ellipsoid rendering demo video: 3D manipulation of clubhead force ellipsoid
- ✅ Nightly CI: 10 reference motions validated across MuJoCo/Drake/Pinocchio
- ✅ 100% of exports include schema version + provenance metadata

### Phase 3 (6w) Success Metrics (IF APPLICABLE)
- ✅ OpenSim muscle model cross-validated against ISB benchmark >95% agreement
- ✅ MyoSuite neural control demo: RL agent learns simple golf putt
- ✅ Full 6-engine validation operational

---

## Recommended Actions (Priority Order)

### Immediate (Start Today)
1. **Resource Allocation**: Assign developer(s) to Phase 1 critical path (48h sprint)
2. **Stakeholder Alignment**: Communicate 8.5-week timeline for full biomechanics + robotics implementation
3. **Infrastructure Setup**: Prepare development environment for OpenSim/MyoSuite integration

### Week 1 (Phase 1 Execution)
1. Implement ZTCF/ZVCF counterfactuals (all 4 engines: MuJoCo, Drake, Pinocchio, Pendulum)
2. Add Jacobian conditioning warnings (prevent silent singularities)
3. Create cross-engine deviation troubleshooting documentation
4. Deploy energy conservation monitors

### Weeks 2-3 (Phase 2 Execution)
1. Implement analytical benchmark suite (5 tests: pendulum, double pendulum, free fall, rigid body, two-link)
2. Integrate dimensional analysis library (pint) for unit safety
3. Deploy ellipsoid visualization (3D rendering via meshcat)
4. Set up nightly cross-engine CI automation

### Weeks 4-9 (Phase 3 Execution — Core Differentiator)
1. **OpenSim Integration** (Weeks 4-6): Hill muscle models, wrapping geometry, activation→force mapping
2. **MyoSuite Integration** (Weeks 7-8): Neural control policies, RL experiments, hybrid torque/muscle models
3. **Power Flow Visualization** (Week 9): Inter-segment energy transfer, wrench arrows, exports
4. **Final Validation** (Week 9): Full 6-engine cross-validation, biomechanical benchmark suite

**COMMITMENT**: No shortcuts. Full implementation of all 19 guideline categories.

---

## Appendix: Quick Reference

### What's Working Well (Keep Doing)
- ✅ Protocol-based architecture
- ✅ Strict CI/CD gates (Black/Ruff/Mypy)
- ✅ Comprehensive docstrings with units
- ✅ Cross-engine validation infrastructure

### What Needs Immediate Attention (48h)
- ❌ Zero counterfactual experiments (BLOCKER for interpretability)
- ❌ No singularity warnings (silent failures)
- ❌ No analytical tests (cannot prove correctness)

### What Needs Structural Work (2w)
- ⚠️ Ellipsoid visualization incomplete
- ⚠️ No automated nightly cross-validation
- ⚠️ Dimensional analysis missing (unit safety)

### What's Long-Term Work (6w)
- ⚠️ OpenSim biomechanics integration
- ⚠️ MyoSuite neural control integration

---

## Contact & Next Steps

**Assessment Authors**:
- Assessment A: Principal/Staff Python Engineer
- Assessment B: Principal Computational Scientist
- Assessment C: Senior Scientific Software Architect

**Next Meeting**: Scope decision (Option 1 vs Option 2)

**Questions?** Consult:
- `docs/assessments/Assessment_A_Architecture_Review_Jan2026.md` (software patterns)
- `docs/assessments/Assessment_B_Scientific_Rigor_Jan2026.md` (numerical correctness)
- `docs/assessments/Assessment_C_Cross_Engine_Jan2026.md` (cross-validation)

**Document Version**: 1.0  
**Last Updated**: 2026-01-06
