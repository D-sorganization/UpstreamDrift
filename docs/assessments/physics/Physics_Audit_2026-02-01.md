# Physics Audit Report - 2026-02-01

**Date:** 2026-02-01
**Auditor:** Jules (Physics Auditor Agent)
**Focus Area:** General Physics Audit (Ball Flight, Impact, Equipment, Biomechanics)

## Executive Summary

- **Overall Physics Fidelity Score:** 6/10
- **Critical Issues:** 2
- **High Priority Gaps:** 3
- **Confidence in Results:** Medium

The audit reveals a solid mathematical foundation for basic trajectories and rigid body approximations. However, critical gaps exist in the **Impact Model**, which treats the clubhead as a point mass (ignoring Moment of Inertia) and fails to generate spin in the compliant (spring-damper) model. **Ball Flight** physics is missing spin decay, which will significantly overestimate carry distance and curvature for long shots. **Equipment** modeling lacks torsional stiffness and 3D bending, limiting the ability to analyze "droop" and face closure dynamics.

While the code structure is clean and modular, the physics fidelity needs elevation to match research-grade standards, particularly for off-center impacts and realistic environmental effects.

## Findings by Category

### 1. Mathematical Correctness
- **Status:** Good
- **Findings:**
    - RK4 integration in `ball_flight_physics.py` is correctly implemented.
    - Vector operations (cross products for Magnus, dot products for impact) are mathematically sound.
    - **Note:** `SpringDamperImpactModel` force limiting (`max_force = 1e5`) is a numerical safeguard but lacks physical justification; for very high speed impacts, this could clip forces artificially.

### 2. Physical Plausibility
- **Status:** Mixed
- **Findings:**
    - **Issue:** `SpringDamperImpactModel` generates **zero spin**.
        - **File:** `src/shared/python/impact_model.py` (Line ~300)
        - **Description:** The loop calculates force `f_contact` purely along the normal vector `n`. Friction (tangential force) is completely ignored in the integration loop.
        - **Impact:** Any impact simulated with this model will result in a knuckling ball (zero spin), which is physically impossible for a lofted club.
        - **Fix:** Implement a friction model (Coulomb or continuous) within the integration loop to generate tangential force and torque.

    - **Issue:** Air Density does not update with Altitude.
        - **File:** `src/shared/python/ball_flight_physics.py`
        - **Description:** `EnvironmentalConditions` has an `altitude` field, but `simulate_trajectory` calculates `const_term` using `self.environment.air_density` directly. The `air_density` field defaults to sea level (1.225 kg/m³) and is not automatically adjusted based on `altitude`.
        - **Impact:** Simulations at altitude (e.g. Denver) will result in sea-level performance unless the user manually calculates and sets `air_density`.
        - **Fix:** Implement an International Standard Atmosphere (ISA) model to derive density from altitude and temperature in `EnvironmentalConditions`.

### 3. Biomechanics Accuracy
- **Status:** Satisfactory with Gaps
- **Findings:**
    - **Issue:** Efficiency Score uses arbitrary mass.
        - **File:** `src/shared/python/statistical_analysis.py` (Line ~760)
        - **Description:** `ke = 0.5 * 1.0 * (self.club_head_speed**2)`. The output energy calculation assumes a 1.0 kg mass for the clubhead. A typical driver is ~0.2 kg.
        - **Impact:** The `efficiency_score` is a relative metric scaled by magic numbers, not a physically meaningful ratio of Energy Out / Work In.
        - **Fix:** Use actual `club_head_mass` if available, or a more realistic default (0.2 kg).

### 4. Ball Flight Physics
- **Status:** Good Foundation, Missing Decay
- **Findings:**
    - **Issue:** Missing Spin Decay.
        - **File:** `src/shared/python/ball_flight_physics.py`
        - **Description:** `LaunchConditions` sets initial `spin_rate`, and `omega` is passed to the accelerator. However, `omega` is treated as a constant throughout the RK4 integration.
        - **Impact:** Real golf balls lose spin over time (exponential decay). Constant spin overestimates lift and drag late in flight, leading to "ballooning" trajectories and overestimated curve.
        - **Fix:** Add `omega` (or spin vector) to the state vector (making it size 9: pos, vel, spin) and implement spin decay torque equation (`d(omega)/dt = ...`).

### 5. Equipment Models
- **Status:** Simplified
- **Findings:**
    - **Issue:** Clubhead is a Point Mass (No MOI).
        - **File:** `src/shared/python/impact_model.py`
        - **Description:** `RigidBodyImpactModel` uses the 1D collision formula `m_eff = (m_ball * m_club) / (m_ball + m_club)`. It ignores the clubhead's Moment of Inertia (MOI) matrix.
        - **Impact:** Off-center hits (toe/heel) behave like center hits in terms of energy transfer (except for the patched gear effect spin). The clubhead does not twist during collision, which is a key energy loss mechanism in real golf.
        - **Fix:** Implement full 3D rigid body collision using the Impulse-Momentum theorem with Inertia Tensors.

    - **Issue:** Shaft Model is Planar (2D).
        - **File:** `src/shared/python/flexible_shaft.py`
        - **Description:** The Finite Element model uses 2 DOFs per node (deflection, rotation). It does not model torsion (twist about axis) or out-of-plane bending.
        - **Impact:** Cannot simulate "droop" (vertical bending due to CG offset) or face opening/closing due to shaft twist.
        - **Fix:** Upgrade beam elements to 6-DOF (3 translation, 3 rotation) or at least 4-DOF (2 bending, 1 torsion, 1 axial) per node.

### 6. Statistical Methods
- **Status:** Good
- **Findings:**
    - `dimensionless_jerk` is correctly implemented.
    - `compute_sample_entropy` and `compute_lyapunov_exponent` provide advanced nonlinear analysis correctly.

## Validation Recommendations

1.  **Impact Tests:** Create a test case colliding a ball 2cm off-center. Verify that the clubhead rotates post-impact (currently it won't).
2.  **Ball Flight Data:** Compare trajectory apex and landing angle against TrackMan optimizer data for identical launch conditions. Expect the current model to fly higher and land steeper due to lack of spin decay.
3.  **Shaft Torsion:** Apply a torque to the shaft tip in `flexible_shaft.py`. Verify if any rotation occurs (currently it won't, as there is no torsional DOF).

## Citations Needed
- **Gear Effect Factors:** `src/shared/python/impact_model.py` uses `gear_effect_h_scale = 100.0`. This empirical constant needs a citation or derivation.
- **Lift/Drag Coeffs:** `src/shared/python/ball_flight_physics.py` uses specific polynomial coefficients (`cd0=0.21`, etc.). Reference (e.g., Bearman & Harvey, Smits & Smith) should be explicit.
