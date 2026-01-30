# Physics Audit Report - 2026-01-30

**Auditor:** PHYSICS AUDITOR (AI Agent)
**Date:** 2026-01-30
**Focus Area:** Core Physics Modules (Ball Flight, Equipment, Biomechanics, Statistics)

## Executive Summary

The physics implementation in the Golf Modeling Suite demonstrates a strong foundation in rigid body dynamics and signal processing but contains several critical deviations from established physical principles and golf-specific empirical data. While the mathematical frameworks (e.g., RK4 integration, Impact Models) are sound in isolation, the parameterization and application of these models frequently rely on simplifications that compromise accuracy for real-world golf scenarios.

*   **Overall Physics Fidelity Score:** 6/10
*   **Critical Issues Count:** 2 (Shaft Stiffness, Spin Decay)
*   **Confidence in Results:** Moderate (High for mathematical correctness, Low for physical parameterization)

## Findings by Category

### 1. Mathematical Correctness

*   **Dimensionless Jerk Dimensionality**
    *   **File:** `src/shared/python/statistical_analysis.py` (Line ~1365)
    *   **Issue:** The `dimensionless_jerk` metric is calculated as `peak_jerk / peak_accel`.
    *   **Expected:** A dimensionless quantity, e.g., $\frac{PeakJerk \cdot Duration}{PeakAccel}$ or $\ln(\int j^2 dt \cdot D^5 / A^2)$.
    *   **Actual:** Units are $[T]^{-1}$ (Frequency).
    *   **Impact:** Metric magnitude depends on time scale, making it incomparable across different movements or durations.
    *   **Recommendation:** Update formula to `(peak_jerk * duration) / peak_accel` or implement the standard cost function form.

### 2. Physical Plausibility

*   **Shaft Bending Stiffness (EI) Overestimation**
    *   **File:** `src/shared/python/flexible_shaft.py`
    *   **Issue:** Default properties use `GRAPHITE_E = 130e9` Pa (130 GPa) with isotropic calculation.
    *   **Expected:** Golf shaft EI varies from ~2 $Nm^2$ (Ladies/Senior) to ~6-7 $Nm^2$ (X-Stiff).
    *   **Actual:** Calculated EI is ~20-140 $Nm^2$ due to high modulus and thick wall assumptions, and ignoring anisotropic composite layups.
    *   **Impact:** Shaft will behave like a rigid steel bar, showing negligible deflection or "kick" during the swing.
    *   **Recommendation:** Use an *effective* Young's Modulus for the beam model (e.g., 50-70 GPa) or allow direct input of EI profiles instead of geometric derivation from high-modulus material properties.

*   **Ball Spin Decay Missing**
    *   **File:** `src/shared/python/ball_flight_physics.py`
    *   **Issue:** The flight dynamics loop updates velocity but treats angular velocity (`omega`) as constant.
    *   **Expected:** Spin decays due to air resistance torque. $\frac{d\omega}{dt} \approx -0.01 \omega$ (approx).
    *   **Actual:** `omega` remains constant throughout the trajectory.
    *   **Impact:** Overestimation of lift (Magnus effect) late in flight, leading to "ballooning" trajectories and inaccurate landing angles/roll.
    *   **Recommendation:** Implement spin decay model (Exponential decay or torque-based).

### 3. Equipment Models

*   **Impact Model Clubhead MOI Simplification**
    *   **File:** `src/shared/python/impact_model.py`
    *   **Issue:** `RigidBodyImpactModel` treats the clubhead as a point mass for linear momentum conservation.
    *   **Expected:** Off-center hits should rotate the clubhead (recoil), reducing the effective mass and energy transferred to the ball.
    *   **Actual:** Uses `m_eff` based on total mass, ignoring MOI and impact location relative to CG.
    *   **Impact:** Overestimates ball speed (smash factor) for off-center hits.
    *   **Recommendation:** Implement full 3D rigid body collision equations including Club MOI tensor.

*   **Gear Effect Heuristics**
    *   **File:** `src/shared/python/impact_model.py`
    *   **Issue:** `compute_gear_effect_spin` uses linear scaling factors (`h_scale=100.0`) without derivation.
    *   **Expected:** Gear effect arises from the moment arm of the friction force relative to CG and the resulting gear ratio between club and ball.
    *   **Actual:** Phenomenological linear heuristic.
    *   **Impact:** High Patent Risk (Zepp/Blast) and potentially inaccurate spin axis results.
    *   **Recommendation:** Derive gear effect from the rigid body collision model (above) rather than using heuristics.

### 4. Biomechanics Accuracy

*   **Kinematic Sequence Metrics**
    *   **File:** `src/shared/python/kinematic_sequence.py`
    *   **Issue:** Focuses only on peak velocities and their order.
    *   **Expected:** Analysis of speed transfer (gain) and deceleration (braking) of proximal segments.
    *   **Actual:** Lacks deceleration metrics. Explicitly checks for specific firing order (Patent Risk).
    *   **Impact:** Limited biomechanical insight; high legal risk.
    *   **Recommendation:** Add Speed Gain and Deceleration metrics. Randomize or generalize the sequence check to avoid patent claims.

*   **Efficiency Calculation**
    *   **File:** `src/shared/python/statistical_analysis.py`
    *   **Issue:** `efficiency_score` uses `peak_club_KE / total_joint_work` but ignores eccentric work.
    *   **Expected:** Net Work or Metabolic Cost proxy.
    *   **Actual:** Uses positive power integral only.
    *   **Impact:** May misrepresent efficiency of swings that rely on stretch-shortening cycles (elastic return).
    *   **Recommendation:** Include eccentric work logic or rename to "Concentric Efficiency".

## Validation Recommendations

1.  **Shaft Frequency Test:** Simulate a cantilevered shaft vibration and check if the fundamental frequency matches standard CPM (Cycles Per Minute) ratings (e.g., 240-270 CPM).
2.  **Trajectory Validation:** Compare `BallFlightSimulator` output against TrackMan Optimizer data for standard launch conditions (e.g., Driver: 160 mph ball speed, 12 deg launch, 2500 rpm spin).
3.  **Impact Conservation:** Verify angular momentum conservation in the Impact Model (currently implicitly violated by the point-mass simplification for the club).

## Citations Needed

*   **Gear Effect:** The heuristic `h_scale=100.0` needs a source or should be replaced by: *Penner, A. R. (2001). "The physics of golf: The convex face of a driver". American Journal of Physics.*
*   **Aerodynamics:** Bearman & Harvey (1976) is cited in constants, but implementation coefficients ($Cd0=0.21$, etc.) should be verified against this source.
