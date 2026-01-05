# Priority Upgrades for Golf Modeling Suite

**Generated:** 2026-01-05
**Based on:** Assessments A, B, and C audit results
**Current Overall Score:** 6.5-6.8/10

---

## Summary

All three assessments identified a **critical architectural flaw**: shared mutable state (`mujoco.MjData`) causing side effects, race conditions, and scientific reproducibility issues. Additionally, Assessment B found a **blocker-level physics error** in centripetal acceleration calculations.

---

## CRITICAL (Fix Within 48 Hours)

### 1. **Fix Physics Hallucination in Centripetal Acceleration**
   - **Severity:** BLOCKER
   - **Issue ID:** B-001
   - **Location:** `kinematic_forces.py:718`
   - **Problem:** `compute_centripetal_acceleration` uses `a_c = v²/r` assuming circular motion about origin (0,0,0), which is fundamentally wrong for articulated kinematic chains
   - **Impact:** Users relying on this for stress analysis will have invalid data
   - **Fix:**
     - **Immediate:** Mark method as "Experimental/Broken" with warning
     - **Short-term:** Rewrite using proper spatial acceleration: `a_total = J·q̈ + J̇·q̇` where the "centripetal/coriolis" part is `J̇·q̇`
   - **Effort:** 4-8 hours (immediate), 2-3 days (proper fix)

### 2. **Eliminate Side Effects in Validation Methods**
   - **Severity:** CRITICAL
   - **Issue IDs:** A-003, F-001
   - **Location:** `inverse_dynamics.py:500`
   - **Problem:** `validate_solution()` calls `mj_step`, advancing simulation time. "Checking" a result changes future calculations (Observer Effect)
   - **Impact:** Breaks scientific reproducibility, causes Heisenbugs
   - **Fix:**
     - Replace `mj_step` with `mj_forward` (static calculation), OR
     - Clone `MjData` before stepping, then restore state after validation
   - **Code Change:**
     ```python
     def validate_solution(self, ...):
         # Save state to prevent side effects
         qpos_backup = self.data.qpos.copy()
         qvel_backup = self.data.qvel.copy()
         time_backup = self.data.time

         # Use forward dynamics instead of stepping
         mujoco.mj_forward(self.model, self.data)

         # Restore state
         self.data.qpos[:] = qpos_backup
         self.data.qvel[:] = qvel_backup
         self.data.time = time_backup
     ```
   - **Effort:** 4-6 hours

---

## HIGH PRIORITY (Fix Within 2 Weeks)

### 3. **Isolate State in Analysis Methods**
   - **Severity:** CRITICAL (for concurrency)
   - **Issue IDs:** A-001, F-002
   - **Location:** `kinematic_forces.py:151`, `inverse_dynamics.py:148`
   - **Problem:** In-place modification of `self.data.qpos/qvel` in compute methods. If async visualizer accesses `data` during calculation, robot appears to spasm/teleport
   - **Impact:** Prevents parallel analysis, causes race conditions, breaks multi-threaded applications
   - **Fix:** Use dedicated `_perturb_data` (already allocated!) instead of shared `self.data`
   - **Code Change:**
     ```python
     def compute_coriolis_forces(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
         # Use private/scratch data structure instead of shared state
         self._perturb_data.qpos[:] = qpos
         self._perturb_data.qvel[:] = qvel
         mujoco.mj_forward(self.model, self._perturb_data)
         # ... rest of computation
     ```
   - **Effort:** 1-2 weeks (affects multiple analysis classes)

### 4. **Create Immutable Physics Interface**
   - **Severity:** MAJOR (architectural)
   - **Issue IDs:** A-001, A-003, F-002
   - **Problem:** Current architecture treats `MjData` as global mutable state, violating functional purity
   - **Impact:** Makes code hard to test, debug, and parallelize
   - **Fix:**
     - Implement context manager that saves/restores `MjData` state
     - Create wrapper treating `MjData` as immutable input, returning new state/derived values
   - **Example:**
     ```python
     class MjDataContext:
         def __enter__(self):
             self.backup = copy_mjdata(self.data)
         def __exit__(self, ...):
             restore_mjdata(self.data, self.backup)

     with MjDataContext(self.data):
         # Safe to mutate here, will restore after
         result = compute_forces(...)
     ```
   - **Effort:** 1-2 weeks

### 5. **Add Dependency Version Enforcement**
   - **Severity:** MINOR (but creates tech debt)
   - **Issue ID:** F-003
   - **Location:** `inverse_dynamics.py:111`
   - **Problem:** Try-except block for Jacobian signature hides version mismatches
   - **Impact:** Can cause dimension alignment errors if versions are wrong
   - **Fix:** Enforce MuJoCo version explicitly in `requirements.txt`, add startup version check
   - **Effort:** 2-4 hours

---

## MEDIUM PRIORITY (Fix Within 6 Weeks)

### 6. **Replace Finite Differences with Analytical Derivatives**
   - **Severity:** MAJOR (performance + numerics)
   - **Issue IDs:** A-002, B-002
   - **Location:** `kinematic_forces.py:221`, `kinematic_forces.py:276`
   - **Problem:**
     - Loops `N` forward kinematics calls per frame (O(N²) bottleneck)
     - Finite differences (`epsilon=1e-6`) introduce discretization noise
   - **Impact:** Slow for high-DOF models, noise propagates to control instability
   - **Fix:** Use analytical RNE recursion (`mujoco.mj_rne`) or `mj_deriv` for Jacobian-based computation
   - **Effort:** 3-4 weeks

### 7. **Fix Hidden Memory Allocation**
   - **Severity:** MINOR (maintainability)
   - **Issue ID:** A-004
   - **Location:** `kinematic_forces.py:83`
   - **Problem:** `_perturb_data` created in `__init__` increases memory footprint silently
   - **Impact:** Unclear memory usage, especially for multiple instances
   - **Fix:** Document allocation clearly, consider lazy initialization or pool pattern
   - **Effort:** 1-2 days

### 8. **Optimize Linear Algebra Operations**
   - **Severity:** MINOR (performance)
   - **Issue ID:** B-003
   - **Location:** `inverse_dynamics.py:469`
   - **Problem:** `lstsq(jacp.T, torques)` is correct but slow
   - **Impact:** Unnecessary computational overhead
   - **Fix:** Precompute pseudo-inverse if dimensions are stable
   - **Effort:** 4-8 hours

---

## ENHANCEMENTS (Non-Blocking Quality Improvements)

### 9. **Add Energy Conservation Verification**
   - **Recommendation from:** Assessment B, C
   - **Add:** Assertion that `Total Power == d/dt(Total Energy)` in `compute_kinematic_power`
   - **Benefit:** Additional physics validation, catches integration errors
   - **Effort:** 1-2 days

### 10. **Implement Analytical Derivative Checks**
   - **Recommendation from:** Assessment C
   - **Add:** Finite-difference verification of `mj_jacBody` vs numerical differentiation
   - **Benefit:** Validates kinematic chain definition
   - **Effort:** 2-3 days

### 11. **Use Spatial Arithmetic (Screw Theory)**
   - **Recommendation from:** Assessment B
   - **Replace:** Explicit "Centrifugal" and "Coriolis" decomposition with Lie algebra approach
   - **Benefit:** More robust, frame-independent force composition
   - **Effort:** 4-6 weeks (research + implementation)

### 12. **Add Null-Space Posture Control**
   - **Recommendation from:** Assessment C
   - **Add:** Expose null-space projection in Inverse Dynamics solver for redundant systems
   - **Benefit:** Enables "secondary tasks" (e.g., keep head still while swinging)
   - **Effort:** 2-3 weeks

### 13. **Move Heavy Computation to C++**
   - **Recommendation from:** Assessment B
   - **Long-term:** Move kinematic analysis to C++ plugin or MuJoCo engine extension
   - **Benefit:** Avoid Python loop overhead
   - **Effort:** 8-12 weeks

---

## Immediate Action Plan (Next Sprint)

1. **Day 1-2:** Mark `compute_centripetal_acceleration` as broken (Priority #1)
2. **Day 2-3:** Fix `validate_solution` side effects (Priority #2)
3. **Week 1-2:** Isolate state in analysis methods (Priority #3)
4. **Week 2:** Implement state isolation context manager (Priority #4)

**Expected Impact:** Moving from 6.5/10 → 8.0/10 after completing Critical + High Priority items

---

## Risk Mitigation

- **What Breaks First?** Parallel analysis will segfault or corrupt data
- **Deployment Fragility:** Binary dependencies (MuJoCo, PyQt6) make portable distribution hard
  - **Mitigation:** Consider containerization (Docker) for consistent environments

---

## Testing Requirements

For all fixes:
1. Add unit tests that verify no state mutation occurs
2. Test concurrent execution (multi-threaded analysis)
3. Validate physics correctness with ground truth data
4. Add regression tests for version compatibility

---

## Summary Scorecard

| Category | Current | After Critical | After High | After Medium |
|----------|---------|----------------|------------|--------------|
| Architecture | 4-5 | 6 | 8 | 8 |
| Scientific Correctness | 4 | 8 | 8 | 9 |
| Reliability | 5 | 7 | 8 | 8 |
| Performance | 5 | 5 | 5 | 7 |
| Maintainability | 7 | 7 | 8 | 9 |
| **Overall** | **6.5** | **7.5** | **8.0** | **8.5** |
