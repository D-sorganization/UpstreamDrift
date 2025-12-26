# Technical Audit: Induced Acceleration Analysis

## Overview
This audit covers the implementation of induced acceleration analysis (IAA) across the MuJoCo, Drake, and Pinocchio physics engines. The analysis is based on the equation of motion:
$M(q)\ddot{q} + C(q,\dot{q})\dot{q} + G(q) = \tau + \tau_{ext}$

## Findings

### 1. MuJoCo Engine (`mujoco_humanoid_golf`)
**File:** `engines/physics_engines/mujoco/python/mujoco_humanoid_golf/inverse_dynamics.py`

#### Critical Issues
*   **Stale Kinematics in IAA:** The `compute_induced_accelerations` method sets `data.qpos` but fails to update kinematics (e.g., `mj_kinematics`) before calling `mj_rne` for the gravity term calculation. This results in gravity forces being calculated for the *previous* robot pose, leading to significant errors.
    *   *Verification:* Confirmed via test script `tests/audit_verification.py`. Gravity torque remained high for a vertical pendulum state because it used the previous horizontal state's kinematics.
*   **Incorrect Attribute Usage:** The code accesses `self.data.qfrc_actuation`, but the correct attribute in current MuJoCo bindings is `self.data.qfrc_actuator`. This causes a runtime `AttributeError`.
*   **Incorrect Solver Signature:** The code calls `mujoco.mj_solveM(model, data, a_g)` (3 args), but the current API requires `mujoco.mj_solveM(model, data, x, y)` (4 args) to solve $Mx = y$.

#### Performance & Code Quality
*   **Redundant Computation:** `mujoco.mj_fullM` is called at the start of the method but the result is discarded/unused. This is an unnecessary $O(n^2)$ operation on potentially stale data.
*   **Optimization:** The method correctly attempts to reuse `qLD` factorization from `mj_forward` for solving, which is $O(n^2)$ vs $O(n^3)$ for full inversion.
*   **Custom ABA:** The standalone `aba.py` implementation in `rigid_body_dynamics/` is mathematically correct, well-tested, and highly optimized (using `out=` parameters and minimal allocations). It serves as a good reference implementation.

### 2. Drake Engine (`drake`)
**File:** `engines/physics_engines/drake/python/src/induced_acceleration.py`

#### Functional Defects
*   **Ignored Control Input:** The `compute_components` method accepts a `tau_app` argument but explicitly ignores it, setting `acc_t = np.zeros_like(acc_g)`. This means control-induced acceleration is never computed.
    ```python
    # Control is zero
    acc_t = np.zeros_like(acc_g)
    ```

#### Performance
*   **Inefficient Solving:** The implementation uses `np.linalg.solve(M, ...)` three times. This likely triggers a dense Cholesky/LU decomposition of $M$ each time (or 3 back-substitutions if optimized by NumPy, but still heavier than reuse).
*   **Correctness:** The logic for Gravity and Velocity components appears mathematically consistent with Drake's `CalcBiasTerm` and `CalcGravityGeneralizedForces` conventions.

### 3. Pinocchio Engine (`pinocchio_golf`)
**File:** `engines/physics_engines/pinocchio/python/pinocchio_golf/induced_acceleration.py`

#### Assessment
*   **Correctness:** The implementation correctly uses the superposition principle of the Articulated Body Algorithm (ABA):
    *   Gravity: `aba(q, 0, 0)` -> $-M^{-1}G$
    *   Velocity: `aba(q, v, 0) - aba(q, 0, 0)` -> $-M^{-1}C\dot{q}$
    *   Control: `aba(q, v, \tau) - aba(q, v, 0)` -> $M^{-1}\tau$
*   **Performance:** Extremely efficient ($O(n)$) as it uses the recursive ABA algorithm directly rather than solving linear systems with the mass matrix.
*   **Safety:** Correctly uses a temporary `_temp_data` structure to avoid side effects on the main simulation state.

## Recommendations

1.  **Fix MuJoCo Stale Kinematics:** Add `mujoco.mj_kinematics(self.model, self.data)` (or `mj_forward` with zero velocity) immediately after setting `qpos` in `compute_induced_accelerations`.
2.  **Update MuJoCo API Usage:**
    *   Rename `qfrc_actuation` to `qfrc_actuator`.
    *   Update `mj_solveM` calls to use the 4-argument signature `(model, data, result_vec, input_force_vec)`.
    *   Remove the unused `mj_fullM` call.
3.  **Implement Drake Control Acceleration:** Update `drake/induced_acceleration.py` to actually use `tau_app`. Calculate `acc_t = np.linalg.solve(M, tau_app)`.
4.  **Standardize Tests:** The current `test_induced_acceleration.py` is weak (skips if modules missing, doesn't verify values deeply). Add the robust logic verification from the audit script to the permanent test suite.

## Conclusion
The Pinocchio implementation is the most robust and efficient. The MuJoCo implementation is currently broken due to API changes and a logic error regarding kinematic updates. The Drake implementation is incomplete (ignores control inputs).
