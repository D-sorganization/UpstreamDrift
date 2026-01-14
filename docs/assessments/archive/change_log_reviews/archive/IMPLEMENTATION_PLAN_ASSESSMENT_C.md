# Assessment C Implementation Plan

**Date:** 2026-01-05  
**Focus:** Physics Engine Integration & Cross-Validation  
**Status:** Many items already completed via Assessment B!

---

## Item Status Matrix

| ID | Severity | Item | Assessment B Overlap | Status |
|----|----------|------|---------------------|--------|
| **C-001** | BLOCKER | Centripetal physics error | B-001 | ✅ COMPLETE |
| **C-002** | Critical | Shared state mutation | A-001, A-003 | ✅ COMPLETE |
| **C-003** | Critical | No conservation tests | B-005 | ✅ COMPLETE |
| **C-004** | Major | Finite diff noise | B-002, B-009 | ✅ COMPLETE |
| **C-005** | Major | Unit ambiguity | B-006, B-003 | ✅ COMPLETE |
| **C-006** | Major | No cross-engine tests | *New* | 🔷 TODO |
| **C-007** | Major | 25% coverage threshold | *New* | 🔷 TODO |
| **C-008** | Minor | No singularity monitoring | B-008 | ✅ COMPLETE |
| **C-009** | Minor | No provenance tracking | *New* | 🔷 TODO |
| **C-010** | Nit | Mixed logging | *New* | 🔷 Low Priority |

### Summary
- **Already Complete (via Assessments A & B):** 6/10 items (60%)
- **New Items to Implement:** 4/10 items (40%)
- **BLOCKER/Critical Items:** 100% complete!

---

## Already Completed Items (Verification)

### C-001: Centripetal Acceleration (BLOCKER) ✅
**Completed in:** Assessment B-001  
**Implementation:**
- Method disabled with `NotImplementedError`
- Comprehensive migration documentation
- Alternative method documented

**Evidence:** `kinematic_forces.py:881-949` (NotImplementedError with detailed message)

---

### C-002: Shared State Mutation (Critical) ✅
**Completed in:** Assessment A-001, A-003 (prior work)  
**Implementation:**
- All `compute_*` methods use `self._perturb_data`
- `MjDataContext` pattern for state isolation
- Thread-safe analysis methods

**Evidence:**
- `kinematic_forces.py:194-227` (`__init__` allocates `_perturb_data`)
- `kinematic_forces.py:77-148` (`MjDataContext` implementation)
- All compute methods verified use private data

---

### C-003: No Conservation Law Tests (Critical) ✅
**Completed in:** Assessment B-005  
**Implementation:**
- Created `tests/integration/test_energy_conservation.py`
- 4 test classes: energy conservation, work-energy, power balance, angular momentum
- Uses documented physical constants

**Evidence:** `tests/integration/test_energy_conservation.py` (360 lines)

---

### C-004: Finite Difference Noise (Major) ✅
**Completed in:** Assessment B-002 (verified), B-009 (improved)  
**Implementation:**
- Analytical RNE for Coriolis forces (B-002)
- Upgraded to second-order central difference for Jacobian derivatives (B-009)
- O(ε²) error vs O(ε)

**Evidence:**
- `kinematic_forces.py:303-337` (RNE-based Coriolis)
- `kinematic_forces.py:643-670` (second-order central difference)

---

### C-005: Unit Ambiguity (Major) ✅
**Completed in:** Assessment B-006, B-003  
**Implementation:**
- Comprehensive 143-line module docstring
- Unit conventions for 7 categories
- Coordinate frame documentation
- All function parameters annotated with units

**Evidence:** `kinematic_forces.py:1-143` (module docstring)

---

### C-008: Singularity Monitoring (Minor) ✅
**Completed in:** Assessment B-008  
**Implementation:**
- Condition number monitoring in `compute_effective_mass()`
- 5 numerical stability checks
- Meaningful error messages with recovery suggestions

**Evidence:** `kinematic_forces.py:830-978` (enhanced `compute_effective_mass`)

---

## New Items Requiring Implementation

### C-006: Cross-Engine Validation Tests (Major) 🔷
**Priority:** HIGH (Scientific Trust)  
**Scope:** Compare MuJoCo vs Pinocchio for basic operations

**Implementation Plan:**
1. Create `tests/cross_engine/test_mujoco_vs_pinocchio.py`
2. Test scenarios:
   - Simple pendulum inverse dynamics
   - Jacobian consistency
   - Mass matrix agreement
3. Use tolerance thresholds from project guidelines (P3)
4. Skip if Pinocchio not installed

**Estimated Effort:** 4-6 hours (Medium)

---

### C-007: Increase Test Coverage Threshold (Major) 🔷
**Priority:** MEDIUM (Engineering Quality)  
**Current:** 25% threshold  
**Target:** 60% (per Assessment recommendations and project guidelines)

**Implementation Plan:**
1. Update `pyproject.toml` coverage threshold
2. Add unit tests for undertested modules:
   - `kinematic_forces.py` edge cases
   - `inverse_dynamics.py` null-space handling
   - Engine adapters
3. Focus on physics computation paths

**Estimated Effort:** 8-12 hours (Large) - ongoing effort

---

### C-009: Provenance Tracking (Minor) 🔷
**Priority:** MEDIUM (Reproducibility)  
**Scope:** Add metadata to exported results

**Implementation Plan:**
1. Create `shared/python/provenance.py` with `@dataclass ProvenanceInfo`
2. Include: timestamp, code version (git SHA), model hash, parameters
3. Add to CSV/NPZ export functions
4. Header format: `# Exported by golf-modeling-suite v1.0.0 (sha: abc123) at 2026-01-05T21:00:00Z`

**Estimated Effort:** 2-3 hours (Small)

---

### C-010: Standardize Logging (Nit) 🔷
**Priority:** LOW (Code Quality)  
**Currently:** Mixed `logging` and `structlog`  
**Target:** Standardize on `structlog`

**Implementation Plan:**
1. Replace `logging.getLogger()` → `structlog.get_logger()`
2. Ensure consistent across all modules
3. Update documentation

**Estimated Effort:** 2-3 hours (Small) - can defer

---

## Recommended Implementation Sequence

### Phase 1: High-Value Quick Wins (Today)
1. **C-009: Provenance Tracking** ✅ (2-3 hours)
   - Immediate reproducibility benefit
   - Small, contained change
   - Aligns with Guideline L (Data Integrity)

2. **Update pyproject.toml coverage** ✅ (15 minutes)
   - Change threshold 25% → 60%
   - Document rationale in commit

### Phase 2: Scientific Validation (Next Session)
3. **C-006: Cross-Engine Tests** (4-6 hours)
   - High scientific value
   - Enables "trust score" increase
   - Core to Assessment C requirements

### Phase 3: Engineering Hygiene (Future)
4. **C-007: Add Tests to Meet Coverage** (ongoing)
   - Incremental improvement
   - Focus on critical paths first

5. **C-010: Logging Standardization** (defer)
   - Nice-to-have, low priority
   - Can be done as cleanup task

---

## Quick Assessment C Summary Report

Once Phase 1 complete, we can claim:

### Assessment C Scorecard (Projected)

| Category | Before | After Phase 1 | After Phase 2 |
|----------|--------|---------------|---------------|
| **A. Scientific Correctness** | 6/10 | 6/10 | 8/10 (cross-validation) |
| **B. Numerical Methods** | 7/10 | 7/10 | 7/10 |
| **C. Architecture** | 7/10 | 7/10 | 7/10 |
| **G. Testing (Validity)** | 5/10 | 5/10 | 8/10 (cross-engine) |
| **H. Validation** | 4/10 | 5/10 (provenance) | 8/10 (external ref) |
| **L. Data Integrity** | 5/10 | 8/10 (provenance) | 8/10 |
| **Weighted Total** | 6.5/10 | 6.8/10 | 7.8/10 |

**Phase 1 Improvement:** +0.3 points (5%)  
**Phase 2 Improvement:** +1.3 points (20%)

---

## Success Criteria

### Minimum Bar for Scientific Trust (from C-007.10)
- [x] `compute_centripetal_acceleration()` deleted/fixed ✅
- [x] Energy conservation verified ✅
- [ ] Cross-engine validation (MuJoCo vs Pinocchio) 🔷
- [x] Unit conventions documented ✅
- [x] State mutation audited ✅
- [ ] Input validation (plausibility checks) 🔷 (partial)
- [ ] Analytical benchmark test 🔷
- [x] Numerical stability verified (κ < 1e6) ✅

**Current: 5/8 complete (62.5%)**  
**After Phase 1: 5/8 (62.5%)**  
**After Phase 2: 7/8 (87.5%)** ← Scientifically trustworthy

---

## Alignment with Project Guidelines

### Already Achieved
- ✅ **Guideline M2**: Acceptance test suite (energy conservation)
- ✅ **Guideline M3**: Failure reporting (singularity detection, meaningful errors)
- ✅ **Guideline O3**: Numerical stability (condition number monitoring)
- ✅ **Guideline R1**: Docstring standards (units documented)

### To Be Achieved
- 🔷 **Guideline M1**: Feature × engine support matrix (via C-006)
- 🔷 **Guideline M2**: Cross-engine comparison tests (via C-006)
- 🔷 **Guideline N3**: Test coverage 60%+ (via C-007)
- 🔷 **Guideline Q3**: Versioned exports (via C-009)

---

## References

- `docs/assessments/Assessment_C_Report_Jan2026.md` (C-001 through C-010)
- `docs/assessments/Assessment_A_Report_Jan2026.md` (State isolation - complete)
- `docs/assessments/Assessment_B_Report_Jan2026.md` (Scientific rigor - complete)
- `docs/project_design_guidelines.qmd` (Guidelines M, N, O, Q, R)

---

**Recommendation:** Start with Phase 1 (provenance + coverage threshold) for quick wins, then Phase 2 (cross-engine validation) for major scientific credibility boost.
