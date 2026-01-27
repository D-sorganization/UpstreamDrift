# Physics Audit Report - 2026-01-27

**Auditor:** PHYSICS AUDITOR
**Date:** 2026-01-27
**Focus Area:** General Physics & Biomechanics

## Executive Summary

- **Overall Physics Fidelity Score:** 5/10
- **Critical Issues Count:** 6 (Spin Decay, Shaft Stiffness, Gear Effect, MOI, Jerk Units, Code/Doc Mismatch)
- **Confidence in Results:** High

The codebase contains a mix of "research-grade" aspirations and placeholder or heuristic implementations. While the mathematical frameworks (RK4 integration, modular impact models) are sound, the physical parameters and specific sub-models often lack rigor or physical plausibility. Most notably, the shaft flexibility model yields stiffness values an order of magnitude too high, and the primary ball flight simulation lacks spin decay, a critical factor for accuracy.

## Findings by Category

### 1. Mathematical Correctness

**Finding 1.1: Dimensional Inconsistency in Jerk Metric**
- **File:** `src/shared/python/statistical_analysis.py` (Line ~877)
- **Issue:** The `dimensionless_jerk` is calculated as `peak_jerk / peak_acc`.
- **Expected Physics:** Dimensionless jerk should be dimensionless, typically calculated as $\text{Jerk}_{rms} \cdot \text{Duration}^5 / \text{Amplitude}^2$ or similar log-normalized metrics.
- **Actual Implementation:** The implemented formula results in units of frequency ($1/s$), not a dimensionless quantity.
- **Impact:** Misleading metric that cannot be compared across movements of different time scales.
- **Recommended Fix:** Implement standard dimensionless jerk formula. (See Issue 042)

### 2. Physical Plausibility

**Finding 2.1: Unrealistic Shaft Stiffness Magnitude**
- **File:** `src/shared/python/flexible_shaft.py`
- **Issue:** The calculated Bending Stiffness ($EI$) for a standard driver shaft is ~140 $N \cdot m^2$.
- **Expected Physics:** Typical golf shaft $EI$ ranges from 2 to 6 $N \cdot m^2$.
- **Actual Implementation:** $EI$ is derived from geometry (1mm wall thickness) and isotropic modulus (130 GPa), leading to a shaft that is ~30x too stiff.
- **Impact:** The shaft will behave effectively as a rigid body, negating any benefits of the flexible shaft model. Kick timing and deflection will be negligible.
- **Recommended Fix:** Calibrate wall thickness and effective modulus to match empirical EI profiles, or allow direct input of EI profiles. (See Issue 041)

**Finding 2.2: Ambiguous Equipment Parameters**
- **File:** `src/shared/python/equipment.py`
- **Issue:** `flex_stiffness` values are [180.0, 150.0, 120.0].
- **Expected Physics:** Standard stiffness values (EI or torsional k) should align with physical units.
- **Actual Implementation:** Values are unlabelled but appear to be consistent with the overestimate in `flexible_shaft.py`.
- **Impact:** Confirms systemic over-stiffness in default configuration.
- **Recommended Fix:** Update default configuration with realistic values.

### 3. Biomechanics Accuracy

**Finding 3.1: Absolute Value Peak Detection**
- **File:** `src/shared/python/kinematic_sequence.py`
- **Issue:** Peak detection uses `np.abs(velocity)`.
- **Expected Physics:** Downswing peaks are distinct from backswing peaks.
- **Actual Implementation:** While generally robust due to higher downswing speeds, this could theoretically identify a fast takeaway as a sequence peak.
- **Impact:** Low/Negligible.
- **Recommended Fix:** Restrict peak search to the downswing phase or check sign of velocity.

### 4. Ball Flight Physics

**Finding 4.1: Missing Spin Decay**
- **File:** `src/shared/python/ball_flight_physics.py`
- **Issue:** Spin rate (`omega`) is constant throughout the flight.
- **Expected Physics:** Spin decays exponentially due to viscous torque ($\omega(t) = \omega_0 e^{-kt}$).
- **Actual Implementation:** `omega` is a constant scalar in the RK4 loop.
- **Impact:** Critical overestimation of lift and drag late in flight, leading to "ballooning" trajectories and overestimated carry. (See Issue 008)
- **Recommended Fix:** Implement `SPIN_DECAY_RATE_S` from `physics_constants.py` in the integration loop.

**Finding 4.2: Docstring Mismatch**
- **File:** `src/shared/python/ball_flight_physics.py`
- **Issue:** Docstring claims implementation of "exponential spin decay".
- **Actual Implementation:** Feature is missing.
- **Impact:** Misleading documentation.

### 5. Equipment Models

**Finding 5.1: Heuristic Gear Effect**
- **File:** `src/shared/python/impact_model.py`
- **Issue:** Gear effect uses linear heuristic (`-gear_factor * h_offset * speed`).
- **Expected Physics:** Gear effect results from the moment arm of the impact force relative to the CG causing rotation of the clubhead, which alters the tangential velocity at the contact point.
- **Actual Implementation:** Empirical approximation.
- **Impact:** Cannot model specific clubhead CG properties or MOI benefits accurately. (See Issue 011)

**Finding 5.2: Constant Ball MOI**
- **File:** `src/shared/python/impact_model.py`
- **Issue:** Ball MOI is hardcoded as solid sphere ($2/5 MR^2$).
- **Expected Physics:** Modern balls have lower normalized MOI (~0.38).
- **Impact:** Inaccurate spin generation calculation. (See Issue 010)

### 6. Statistical Methods

- Generally robust implementation of advanced metrics (Lyapunov, Entropy).
- **Note:** `compute_jerk_metrics` unit issue (see Finding 1.1).

## Validation Recommendations

1.  **Shaft Stiffness:** Validation against CoolClubs or similar shaft profile databases is required immediately.
2.  **Trajectory:** Compare `ball_flight_physics.py` output against `flight_models.py` (MacDonaldHanzely) to quantify the error from missing spin decay.
3.  **Impact:** Compare heuristic gear effect against a full rigid-body physics simulation (e.g., using `pydrake` or `mujoco` impact solvers if available) to calibrate the heuristic factors.

## Citations Needed

- `ball_flight_physics.py`: Aerodynamic coefficients polynomial source.
- `statistical_analysis.py`: Source for the specific "Dimensionless Jerk" simplification used.
