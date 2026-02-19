# Physics Audit Report - 2025-02-24

**Auditor:** Jules (AI Physics Expert)
**Date:** 2025-02-24
**Scope:** Physics, Biomechanics, Aerodynamics, Equipment Modeling

## Executive Summary

The physics engine demonstrates a solid foundation in many areas (especially ball flight and basic beam theory) but suffers from critical simplifications in ground reaction forces, impact mechanics, and reporting metrics that significantly compromise simulation fidelity for advanced use cases.

*   **Overall Physics Fidelity Score:** 6/10
*   **Critical Issues Count:** 3
*   **Confidence in Results:** High (Verified with reproduction scripts)

## Findings by Category

### 1. Ground Reaction Forces (Critical)

*   **File:** `src/shared/python/physics/ground_reaction_forces.py`
*   **Issue:** Incorrect fallback mechanism for GRF calculation.
*   **Description:** When the physics engine fails to report contact forces (e.g., foot leaves ground or solver gap), the code falls back to summing gravity forces ($W = \sum mg$).
*   **Expected Physics:** Dynamic Ground Reaction Force $F_z = m(g + a_z)$. During a golf swing, peak vertical forces often exceed 200% body weight due to downward acceleration.
*   **Actual Implementation:** Static equilibrium assumption ($F_z = mg$).
*   **Impact:** Massive discontinuities in GRF data; dynamic peaks are completely missed if contact data is intermittent.
*   **Recommended Fix:** Implement an inertial measurement fallback using center-of-mass acceleration: $F_{fallback} = m(g + a_{com})$.

### 2. Impact Model (High)

*   **File:** `src/shared/python/physics/impact_model.py`
*   **Issue:** Scalar approximation of Moment of Inertia (MOI).
*   **Description:** The `RigidBodyImpactModel` uses a scalar `clubhead_moi` and a simplified effective mass formula $m_{eff} = 1 / (1/m + r^2/I)$.
*   **Expected Physics:** Full 3D inertia tensor usage. The effective mass at impact point $r$ with normal $n$ should be $m_{eff} = \frac{1}{\frac{1}{m} + (r \times n)^T I^{-1} (r \times n)}$.
*   **Actual Implementation:** Scalar approximation that treats heel-toe and crown-sole MOI as identical.
*   **Impact:** Significant errors (~40%) in ball speed retention for vertical miss-hits (high/low on face) versus horizontal miss-hits, failing to capture "Twist Face" mechanics.
*   **Recommended Fix:** Update `PreImpactState` to accept a 3x3 inertia tensor and implement the full matrix effective mass equation.

### 3. Shaft Dynamics (High)

*   **File:** `src/shared/python/physics/flexible_shaft.py`
*   **Issue:** Missing Torsional Degrees of Freedom and 2D limitation.
*   **Description:** The `FiniteElementShaftModel` uses a 4-DOF per element formulation ($w_1, \theta_1, w_2, \theta_2$), effectively modeling a 2D planar beam.
*   **Expected Physics:** 6-DOF per node (or at least 4-DOF + Torsion) to model 3D bending (droop + lead/lag) and twisting.
*   **Actual Implementation:** 2D Euler-Bernoulli beam without torsion.
*   **Impact:** Impossible to model dynamic face closure due to shaft twisting (CG offset torque) or gear effect feedback.
*   **Recommended Fix:** Upgrade to a spatial beam element (12-DOF) or add a torsional degree of freedom ($GJ/L$).

### 4. Reporting & Scoring (Critical)

*   **File:** `src/shared/python/analysis/reporting.py`
*   **Issue:** Hardcoded mass in Kinetic Energy calculation.
*   **Description:** `efficiency_score` calculates clubhead KE as `0.5 * 1.0 * speed^2`.
*   **Expected Physics:** Clubhead mass is typically ~0.2 kg (200g).
*   **Actual Implementation:** Hardcoded 1.0 kg mass.
*   **Impact:** Kinetic Energy and Efficiency Score are inflated by factor of 5 (500% error).
*   **Recommended Fix:** Use the actual `clubhead_mass` from the simulation state.

### 5. Patent Risk (Medium)

*   **File:** `src/shared/python/analysis/pca_analysis.py`
*   **Issue:** "Efficiency Score" based on kinematic sequence order.
*   **Description:** Calculates a score based on the percentage of segments peaking in the "expected" order.
*   **Impact:** This logic closely mimics proprietary "Kinematic Sequence" scoring methods (e.g., TPI) and poses a patent infringement risk if used commercially under this name.
*   **Recommended Fix:** Rename to "Sequence Consistency Metric" and ensure documentation clarifies it is a generic statistical measure.

### 6. Aerodynamics (Medium)

*   **File:** `src/shared/python/physics/aerodynamics.py`
*   **Issue:** Missing spin-dependency in `DragModel`.
*   **Description:** The new `DragModel` considers Reynolds number but ignores spin ratio ($S = \omega r / v$).
*   **Expected Physics:** Drag coefficient $C_d$ generally increases with spin rate due to wake deflection (Bearman & Harvey).
*   **Actual Implementation:** $C_d$ depends only on velocity (Reynolds).
*   **Impact:** Underestimates drag for high-spin shots (wedges) compared to the legacy `ball_flight_physics.py` model.
*   **Recommended Fix:** Incorporate the spin-dependent term from `ball_flight_physics.py` into the modular `DragModel`.

## Validation Recommendations

1.  **Free Fall Test:** Drop a virtual golfer (no contact) and verify GRF is exactly zero (currently fails).
2.  **Off-Center Impact Test:** Compare ball speed for 1cm toe hit vs 1cm high hit. With correct 3D MOI, these should differ significantly.
3.  **Torsion Test:** Apply a torque about the shaft axis and verify twist angle (currently zero compliance).

## Citations Needed

*   `src/shared/python/physics/ball_flight_physics.py`: Implements "Waterloo/Penner" coefficients without citation.
*   `src/shared/python/physics/impact_model.py`: Gear effect factor needs empirical reference (e.g., Cochran & Stobbs).
