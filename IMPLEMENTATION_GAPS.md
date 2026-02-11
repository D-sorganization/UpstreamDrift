# Implementation Gaps and Inaccuracies Report

This document outlines implementation gaps, placeholder code, and physical inaccuracies identified during a repository review.

## 1. User Interface (UI)
*   **File:** `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui/Simscape Multibody Data Plotters/Python Version/integrated_golf_gui_r0/golf_gui_application.py`
*   **Issue:** The `SimulinkModelTab` and `ComparisonTab` classes are implemented as placeholders.
*   **Details:** The tabs display labels stating "This tab will support..." but lack functional implementation for loading Simulink data or comparing it with motion capture data.
*   **Action Required:** Implement data loading logic for Simulink exports and visualization/comparison logic.

## 2. Physics / Ground Reaction Forces (GRF)
*   **File:** `src/shared/python/physics/ground_reaction_forces.py`
*   **Issue:** `extract_grf_from_contacts` uses a static gravity fallback when contact data is unavailable.
*   **Details:** The code sums gravity forces (`total_force[2] += abs(np.sum(g))`) as an approximation. This is physically incorrect for dynamic movements like a golf swing where ground reaction forces significantly exceed body weight due to acceleration.
*   **Action Required:** Implement a proper inverse dynamics solver or ensure the physics engine reliably reports contact forces.

## 3. Visualization Backends
*   **File:** `src/unreal_integration/viewer_backends.py`
*   **Issue:** `PyVistaBackend` and `UnrealBridgeBackend` are unimplemented.
*   **Details:** The factory function raises `NotImplementedError` for these backends, despite them being listed in the `BackendType` enum and docstrings.
*   **Action Required:** Implement the `PyVistaBackend` using `pyvista` and the `UnrealBridgeBackend` using a socket or IPC bridge to Unreal Engine.

## 4. Aerodynamics Modeling Inconsistency
*   **Files:** `src/shared/python/physics/ball_flight_physics.py` vs `src/shared/python/physics/aerodynamics.py`
*   **Issue:** Inconsistent aerodynamic models.
*   **Details:**
    *   `BallFlightSimulator` in `ball_flight_physics.py` uses polynomial coefficients (Bearman & Harvey style): `cd = cd0 + spin_ratio * (cd1 + spin_ratio * cd2)`.
    *   `AerodynamicsEngine` in `aerodynamics.py` uses saturation/exponential models: `cl = max_coefficient * (1 - math.exp(-spin_ratio / 0.1))`.
    *   This leads to divergent trajectories between the two simulators.
*   **Action Required:** Standardize on a single aerodynamic model (preferably the more sophisticated `AerodynamicsEngine`) or ensure coefficients are aligned.

## 5. Impact Model Physics
*   **File:** `src/shared/python/physics/impact_model.py`
*   **Issue:** `RigidBodyImpactModel` treats clubhead angular momentum incorrectly.
*   **Details:** The `solve` method calculates post-impact linear velocity but explicitly copies the *pre-impact* angular velocity for the clubhead: `clubhead_angular_velocity=pre_state.clubhead_angular_velocity.copy()`. This ignores the change in angular momentum due to off-center impacts (torque), violating conservation laws.
*   **Action Required:** Update the impact solver to compute the change in clubhead angular velocity based on the impulse and impact offset.

## 6. Statistical Analysis / Reporting
*   **File:** `src/shared/python/analysis/reporting.py`
*   **Issue:** Incorrect mass used in efficiency score calculation.
*   **Details:** The `efficiency_score` calculation uses a hardcoded mass of `1.0` kg for calculating club head kinetic energy (`ke = 0.5 * 1.0 * (self.club_head_speed**2)`). A typical driver head weighs ~0.2 kg. This inflates the calculated kinetic energy and distorts the efficiency metric.
*   **Action Required:** Use a configurable club mass or a realistic default (e.g., 0.2 kg).
