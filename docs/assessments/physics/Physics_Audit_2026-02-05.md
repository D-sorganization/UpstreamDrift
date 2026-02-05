# Physics Audit Report
**Date:** 2026-02-05
**Auditor:** Jules (Physics Auditor Agent)
**Focus Area:** Core Physics Engines (Ball Flight, Impact, Kinematics)

## Executive Summary

**Overall Physics Fidelity Score:** 5/10
**Critical Issues:** 2
**High Priority Issues:** 2

The core physics engines demonstrate a mix of high-fidelity components (e.g., `AerodynamicsEngine`, `FiniteElementShaftModel`) and significant oversimplifications in critical areas (Impact Physics, Shaft Torsion). While the mathematical implementations of specific sub-components (like RK4 integration or beam stiffness matrices) are correct, the physical models often lack essential degrees of freedom required for accurate golf simulation.

The most critical deficiency is the treatment of the clubhead as a point mass in `impact_model.py`, effectively ignoring Moment of Inertia (MOI) and requiring empirical "magic number" patches for gear effect. This fundamentally compromises the accuracy of off-center strikes.

## Findings by Category

### 1. Equipment Models (Impact & Shaft)

#### **CRITICAL: Point Mass Assumption in Impact Model**
- **File:** `src/shared/python/impact_model.py` (Line 135, `RigidBodyImpactModel`)
- **Issue:** The impact solver uses a 1D impulse-momentum equation for point masses: `m_eff = (m_ball * m_club) / (m_ball + m_club)`. It ignores the clubhead's Moment of Inertia (MOI) and the offset of the impact vector from the Center of Gravity (CG).
- **Impact:** Off-center hits do not naturally generate gear effect or reduce ball speed due to clubhead twisting.
- **Actual Implementation:** Gear effect is added post-hoc via `compute_gear_effect_spin` using arbitrary scaling factors (`gear_effect_h_scale=100.0`).
- **Recommendation:** Implement a full 6-DOF rigid body collision model solving the Newton-Euler equations, incorporating the clubhead's inertia tensor.

#### **HIGH: Missing Torsional Stiffness in Shaft Model**
- **File:** `src/shared/python/flexible_shaft.py`
- **Issue:** The `FiniteElementShaftModel` implements Euler-Bernoulli beam elements which model transverse bending (EI) but ignore torsional stiffness (GJ).
- **Impact:** The model cannot simulate "closing the face" dynamics or the twisting of the shaft due to the clubhead's offset CG (droop/twist). This is a critical factor in delivery accuracy.
- **Recommendation:** Add torsional degrees of freedom (twist angle) to the finite element formulation and `ShaftProperties`.

### 2. Ball Flight Physics

#### **MEDIUM: Inconsistent Aerodynamic Coefficients**
- **File:** `src/shared/python/ball_flight_physics.py` vs `src/shared/python/physics_constants.py`
- **Issue:** `BallProperties` uses hardcoded polynomial coefficients (e.g., `cd0=0.21`) that differ from the centralized constants (e.g., `GOLF_BALL_DRAG_COEFFICIENT=0.25`).
- **Impact:** Simulation results will vary depending on which simulator class (`BallFlightSimulator` vs `EnhancedBallFlightSimulator`) is used, leading to confusion.
- **Recommendation:** Standardize coefficients. The `BallProperties` class should ideally load from `physics_constants.py` or a unified configuration.

#### **LOW: Missing Spin Decay in Basic Simulator**
- **File:** `src/shared/python/ball_flight_physics.py`
- **Issue:** The basic `BallFlightSimulator` treats `omega` (spin) as constant throughout the trajectory in `_solve_rk4_loop`.
- **Impact:** Overestimates lift and drag late in the flight, leading to inaccurate carry distances.
- **Recommendation:** Port the spin decay logic from `AerodynamicsEngine` to the basic simulator or deprecate the basic simulator.

### 3. Biomechanics & Kinematics

#### **HIGH: Hardcoded Kinematic Sequence Order**
- **File:** `src/shared/python/kinematic_sequence.py`
- **Issue:** `KinematicSequenceAnalyzer` hardcodes the expected order as `['Pelvis', 'Torso', 'Arm', 'Club']`.
- **Impact:** This restricts the tool's validity for non-standard swing styles (e.g., short game, elite variations) and creates potential patent risks (TPI patents).
- **Recommendation:** Refactor to allow dynamic definition of segments and expected orders. Rename metrics to avoid trademarked terms if necessary.

### 4. Statistical Methods

#### **LOW: Arbitrary Mass in Efficiency Calculation**
- **File:** `src/shared/python/statistical_analysis.py` (Line 806)
- **Issue:** `efficiency_score` calculates Kinetic Energy using a hardcoded mass of `1.0` kg: `ke = 0.5 * 1.0 * (self.club_head_speed**2)`.
- **Impact:** The score is arbitrary and not physically meaningful as a ratio of real energy.
- **Recommendation:** Use `GOLF_CLUBHEAD_MASS_KG` (approx 0.2 kg) or allow injection of club mass.

## Validation Recommendations

1.  **Impact Physics:** Create a test case comparing `RigidBodyImpactModel` results against the theoretical output of a 2D rigid body collision with known MOI. The current model will fail this validation for off-center hits.
2.  **Shaft Dynamics:** Validate the FE model against a known solution for a cantilevered beam under *torsional* load. The current model will show zero twist.
3.  **Ball Flight:** Compare `BallFlightSimulator` vs `EnhancedBallFlightSimulator` trajectories. They should match when `AerodynamicsConfig` matches the basic settings, but currently diverge due to different coefficient models.

## Citations Needed

-   **Gear Effect Constants:** The values `h_scale=100.0` and `v_scale=50.0` in `impact_model.py` need a source or derivation.
-   **Ball Coefficients:** The polynomial coefficients in `BallProperties` (`cd0=0.21`, etc.) need a citation (likely Bearman & Harvey or similar).
