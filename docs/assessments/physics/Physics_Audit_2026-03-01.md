# Physics Audit Report (2026-03-01)

**Focus Area:** Repository-Wide Physics Implementations

## Executive Summary

The physics implementation across the codebase demonstrates a high degree of modularity and clear design principles, with explicit separation of concerns (e.g., aerodynamics, impact, ground reaction forces). However, several critical physical inaccuracies and implementation gaps have been identified that significantly impact the fidelity of the simulation results.

- **Overall Physics Fidelity Score:** 7/10
- **Critical Issues:** 3 (Impact Model, GRF Fallback, Aerodynamics Double Counting)
- **High Priority Gaps:** 2 (Shaft Torsion, Biomechanics Kinetics)
- **Confidence in Results:** Moderate (High for kinematics, Low for kinetics/forces)

---

## Findings by Category

### 1. Mathematical Correctness
- **Finding 1.1: Simplified Scalar Effective Mass in Impact Model**
  - **File:** `src/shared/python/physics/impact_model.py` (Line 158)
  - **Issue:** The `RigidBodyImpactModel` calculates effective mass as `1 / (1/m + r^2/I)`. This scalar approximation ignores the full 3D inertia tensor of the clubhead and the directional component of the impact force relative to the center of gravity (CG).
  - **Expected Physics:** Full 3D impulse-momentum calculation using the inertia tensor ($J = (M^{-1} + (r \times n)^T I^{-1} (r \times n))^{-1} (1+e) v_{rel}$).
  - **Actual Implementation:** Scalar approximation.
  - **Impact:** Significantly inaccurate ball speed and spin rates for off-center hits (gear effect), leading to incorrect carry distances and dispersion patterns.
  - **Recommended Fix:** Implement full 3D impulse-momentum equations using the inertia tensor.

- **Finding 1.2: Incorrect GRF Fallback Calculation**
  - **File:** `src/shared/python/physics/ground_reaction_forces.py` (Line 385)
  - **Issue:** When contact data is unavailable, the fallback mechanism sums the scalar weight ($W=mg$) of bodies.
  - **Expected Physics:** Dynamic Ground Reaction Force $F_{GRF} = m(g + a_{com})$.
  - **Actual Implementation:** `total_force[2] += abs(np.sum(g))` (Static weight only).
  - **Impact:** Underestimates peak forces during dynamic swings (which can exceed 2-3x body weight), rendering biomechanical analysis invalid for power transfer.
  - **Recommended Fix:** Implement inverse dynamics to estimate required GRF from body accelerations.

### 2. Physical Plausibility
- **Finding 2.1: Missing Shaft Torsional Dynamics**
  - **File:** `src/shared/python/physics/flexible_shaft.py`
  - **Issue:** The Euler-Bernoulli beam model accounts for bending but explicitly excludes torsion (twisting).
  - **Expected Physics:** Coupled bending and torsion to model shaft torque and spine alignment.
  - **Actual Implementation:** Bending-only model with `TODO: Implement Torsional Dynamics`.
  - **Impact:** Unable to model "spine alignment" effects or the clubface closing rate variations due to shaft torque, which is a primary driver of left/right dispersion.
  - **Recommended Fix:** Add torsional degrees of freedom to the Finite Element model.

### 3. Biomechanics Accuracy
- **Finding 3.1: 1D Kinematic Sequence Analysis**
  - **File:** `src/shared/python/biomechanics/kinematic_sequence.py`
  - **Issue:** Sequence metrics are calculated on 1D velocity magnitudes (`np.abs(velocity_data)`).
  - **Expected Physics:** Rotational velocities should be analyzed as 3D vectors or projected onto anatomical planes (e.g., pelvic rotation vs. tilt).
  - **Actual Implementation:** Velocity peak detection only.
  - **Impact:** Misses out-of-plane rotations which are critical for injury risk assessment (e.g., "early extension").
  - **Recommended Fix:** Implement quaternion-based angular velocity analysis.

- **Finding 3.2: Missing Kinetic Metrics**
  - **File:** `src/shared/python/biomechanics/kinematic_sequence.py`
  - **Issue:** Analysis is purely kinematic (velocities/timing). Kinetic metrics (torques, power flow) are missing.
  - **Expected Physics:** Joint torque calculations and inter-segmental power flow ($P = M \cdot \omega$).
  - **Actual Implementation:** Velocity peak detection only.
  - **Impact:** Unable to explain *why* the sequence occurs or assess injury risk from joint loading.
  - **Recommended Fix:** Add joint torque and power flow calculations.

- **Finding 3.3: Patent Infringement Risk in Efficiency Score**
  - **File:** `src/shared/python/analysis/pca_analysis.py` (Line 160)
  - **Issue:** The `efficiency_score` is calculated as `matches / len(expected_order)`.
  - **Expected Physics:** A distinct efficiency calculation logic.
  - **Actual Implementation:** Generic generic match scoring.
  - **Impact:** This metric and its nomenclature overlap with patent claims from Zepp Labs and Blast Motion regarding "kinematic sequence scoring."
  - **Recommended Fix:** Rename metric to "Sequence Adherence" and consult legal counsel.

### 4. Ball Flight Physics
- **Finding 4.1: Aerodynamics Double Counting of Lift Forces**
  - **File:** `src/shared/python/physics/aerodynamics.py` (Line 590)
  - **Issue:** The `AerodynamicsEngine` sums forces from both `LiftModel` (backspin lift) and `MagnusModel` (spin-induced lateral force). Physically, the Magnus effect *is* the mechanism for spin-induced lift.
  - **Expected Physics:** A single unified model for spin-induced aerodynamic forces ($F \propto \omega \times v$).
  - **Actual Implementation:** `total = drag + lift + magnus`. If both are enabled (default), lift forces are applied twice for backspin components.
  - **Impact:** Significant overestimation of lift, leading to "ballooning" trajectories and unrealistic carry distances.
  - **Recommended Fix:** Consolidate `LiftModel` and `MagnusModel` or ensure mutual exclusivity for the vertical component.

- **Finding 4.2: Missing Environmental Models**
  - **File:** `src/shared/python/physics/ball_flight_physics.py`
  - **Issue:** TODOs for "Environmental Gradient Modeling", "Hydrodynamic Lubrication", "Dimple Geometry Optimization".
  - **Expected Physics:** Robust models that account for varying environmental conditions.
  - **Actual Implementation:** Missing logic.
  - **Impact:** High-fidelity simulation of wind shear and wet weather conditions is impossible.
  - **Recommended Fix:** Implement models for wind shear, rain, and temperature gradients.

### 5. Equipment Models
- **Finding 5.1: Empirical Gear Effect**
  - **File:** `src/shared/python/physics/impact_model.py` (Line 491)
  - **Issue:** Gear effect spin is calculated using a linear coefficient (`gear_factor`) rather than deriving it from the friction and tangential impulse at impact.
  - **Expected Physics:** Spin driven by clubface friction and off-center impulse.
  - **Actual Implementation:** A static factor of `gear_factor = 0.5`.
  - **Impact:** Less accurate prediction of spin axis tilt for complex face geometries (e.g., "Twist Face").
  - **Recommended Fix:** Derive gear effect spin dynamically from impact geometry.

### 6. Statistical Methods
- **Finding 6.1: Lack of Uncertainty Propagation**
  - **Scope:** Entire Physics Module
  - **Issue:** No implementation of Monte Carlo or analytical uncertainty propagation for input parameters (e.g., clubhead speed +/- 2 mph).
  - **Expected Physics:** Inputs should support distributions to reflect sensor noise or human variation.
  - **Actual Implementation:** Deterministic single values.
  - **Impact:** Users receive a single "perfect" number rather than a confidence interval, which is misleading for coaching.
  - **Recommended Fix:** Incorporate Monte Carlo frameworks around physics calculations.

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
