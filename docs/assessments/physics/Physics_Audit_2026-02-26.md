# Physics Audit Report (2026-02-26)

**Focus Area:** Repository-Wide Physics Implementations

## Executive Summary

The physics implementation across the codebase demonstrates a high degree of modularity and clear design principles, with explicit separation of concerns (e.g., aerodynamics, impact, ground reaction forces). However, several critical physical inaccuracies and implementation gaps have been identified that significantly impact the fidelity of the simulation results.

- **Overall Physics Fidelity Score:** 7/10
- **Critical Issues:** 3 (Impact Model, GRF Fallback, Aerodynamics Double Counting)
- **High Priority Gaps:** 2 (Shaft Torsion, Biomechanics Kinetics)
- **Confidence in Results:** Moderate (High for kinematics, Low for kinetics/forces)

---

## Findings by Category

### 1. Aerodynamics: Double Counting of Lift Forces
- **File:** `src/shared/python/physics/aerodynamics.py` (Line 590)
- **Description:** The `AerodynamicsEngine` sums forces from both `LiftModel` (backspin lift) and `MagnusModel` (spin-induced lateral force). Physically, the Magnus effect *is* the mechanism for spin-induced lift.
- **Expected Physics:** A single unified model for spin-induced aerodynamic forces ($F \propto \omega \times v$).
- **Actual Implementation:** `total = drag + lift + magnus`. If both are enabled (default), lift forces are applied twice for backspin components.
- **Impact:** Significant overestimation of lift, leading to "ballooning" trajectories and unrealistic carry distances.
- **Issue:** [ISSUE_PHYSICS_AERODYNAMICS_DOUBLE_COUNTING.md](../../completist/issues/ISSUE_PHYSICS_AERODYNAMICS_DOUBLE_COUNTING.md)

### 2. Impact Model: Simplified Effective Mass
- **File:** `src/shared/python/physics/impact_model.py` (Line 158)
- **Description:** `RigidBodyImpactModel` uses a scalar effective mass approximation (`1 / (1/m + r^2/I)`) that ignores the full 3D inertia tensor and impact vector direction. Explicitly marked with `FIXME`.
- **Expected Physics:** Full 3D impulse-momentum calculation using the inertia tensor ($J = (M^{-1} + (r \times n)^T I^{-1} (r \times n))^{-1} (1+e) v_{rel}$).
- **Actual Implementation:** Planar approximation valid only for symmetric impacts.
- **Impact:** Inaccurate ball speeds and spin axes for off-center hits (heel/toe).
- **Issue:** [ISSUE_PHYSICS_IMPACT_MODEL.md](../../completist/issues/ISSUE_PHYSICS_IMPACT_MODEL.md)

### 3. Ground Reaction Forces: Incorrect Fallback
- **File:** `src/shared/python/physics/ground_reaction_forces.py` (Line 385)
- **Description:** When engine contact data is unavailable, the fallback mechanism sums static weights (`W=mg`) instead of calculating dynamic forces.
- **Expected Physics:** Inverse dynamics calculation: $F_{GRF} = M(g + a_{com})$.
- **Actual Implementation:** Static weight summation.
- **Impact:** Dynamic vertical forces in a swing (often >1.5x BW) are capped at 1.0x BW, invalidating power transfer analysis.
- **Issue:** [ISSUE_PHYSICS_GRF_FALLBACK.md](../../completist/issues/ISSUE_PHYSICS_GRF_FALLBACK.md)

### 4. Equipment: Missing Torsional Dynamics
- **File:** `src/shared/python/physics/flexible_shaft.py`
- **Description:** The Finite Element beam model implements Euler-Bernoulli bending but explicitly excludes torsion (twisting).
- **Expected Physics:** Coupled bending and torsion to model shaft torque and spine alignment.
- **Actual Implementation:** Bending-only model.
- **Impact:** Cannot predict dispersion due to clubface closing rate variations or "feel" feedback.
- **Issue:** [ISSUE_PHYSICS_SHAFT_TORSION.md](../../completist/issues/ISSUE_PHYSICS_SHAFT_TORSION.md)

### 5. Biomechanics: Missing Kinetic Metrics
- **File:** `src/shared/python/biomechanics/kinematic_sequence.py`
- **Description:** Analysis is purely kinematic (velocities/timing). Kinetic metrics (torques, power flow) are missing.
- **Expected Physics:** Joint torque calculations and inter-segmental power flow ($P = M \cdot \omega$).
- **Actual Implementation:** Velocity peak detection only.
- **Impact:** Unable to explain *why* the sequence occurs or assess injury risk from joint loading.
- **Issue:** [ISSUE_BIOMECHANICS_MISSING_KINETICS.md](../../completist/issues/ISSUE_BIOMECHANICS_MISSING_KINETICS.md)

---

## Validation Recommendations

### Test Cases Needed
1.  **Aerodynamics Validation:**
    -   Compare `AerodynamicsEngine` output (Drag+Lift+Magnus) against standard wind tunnel data (e.g., Bearman & Harvey curves) for a range of spin rates and velocities.
    -   Verify that disabling `LiftModel` while keeping `MagnusModel` yields physically plausible lift-to-drag ratios.

2.  **Impact Validation:**
    -   Simulate off-center impacts (e.g., 20mm toe hit) and compare ball speed retention against TrackMan/FlightScope data or FEA results.
    -   Verify gear effect spin axis tilt matches theoretical predictions for given CG offsets.

3.  **GRF Validation:**
    -   Compare computed GRF time series against force plate data for a standard swing. Check peak vertical force magnitude (>1.5 BW).

### Expert Review Areas
-   **Patent Review:** Re-evaluate `efficiency_score` logic in `pca_analysis.py` (referenced in comments) for infringement risks against Zepp/Blast Motion patents.
-   **Biomechanics:** Review joint angle conventions (Euler vs. Quaternion) in `kinematic_sequence.py` when extending to kinetics to avoid gimbal lock issues.

---

## Citations Needed

Implementations needing academic references:
-   **Aerodynamics:** Bearman, P.W., & Harvey, J.K. (1976). "Golf ball aerodynamics." *Aeronautical Quarterly*, 27(2), 112-122.
-   **Impact:** Cochran, A., & Stobbs, J. (1968). *Search for the Perfect Swing*. Lippincott. (For gear effect and collision physics).
-   **Ball Flight:** Smits, A.J., & Ogg, S. (2004). "Golf ball aerodynamics." *Physics Today*, 57(2).
