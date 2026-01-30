---
title: "Physics Gap: Missing Ball Spin Decay"
labels: ["physics-gap", "high"]
assignees: ["physics-lead"]
---

## Description
The ball flight trajectory solver in `src/shared/python/ball_flight_physics.py` treats angular velocity (`omega`) as a constant throughout the flight. The RK4 integration loop (`_solve_rk4_loop`) updates linear velocity but recirculates the initial `omega` value.

In reality, golf ball spin decays due to air friction (viscous torque). The decay rate is approximately exponential or linear with time, often modeled as $\omega(t) = \omega_0 e^{-ct}$ or $\frac{d\omega}{dt} = -0.01\omega$.

## Impact
*   **Overestimation of Lift:** Lift (Magnus force) remains high late in the trajectory.
*   **Ballooning:** Balls will fly higher and stay in the air longer than physically possible.
*   **Landing Roll Error:** The landing spin rate will be equal to the launch spin rate (e.g., 3000 rpm), whereas it should be significantly lower (e.g., 1500 rpm), affecting bounce and roll calculations.

## Affected Files
*   `src/shared/python/ball_flight_physics.py`: `_solve_rk4_loop`, `_flight_dynamics_step`

## Recommended Fix
1.  Add `omega` to the state vector in the RK4 solver (state size 6 -> 7 or separate variable).
2.  Implement a torque term in `_flight_dynamics_step`: $\tau = -C_{decay} \cdot \omega$.
3.  Update `omega` in the integration step.
