# Critical Issue: Missing Ball Spin Decay in Flight Dynamics

**Status:** Open
**Priority:** Critical
**Labels:** physics-error, critical
**Date Identified:** 2026-01-22
**Author:** PHYSICS AUDITOR

## Description
The ball flight physics simulation (`shared/python/ball_flight_physics.py`) fails to model the decay of spin rate over time. The `_flight_dynamics_core` integration loop treats the angular velocity (`omega`) as a constant determined by launch conditions.

Real golf balls experience a viscous torque that causes spin to decay exponentially ($$\omega(t) \approx \omega_0 e^{-kt}$$). Without this, the simulation maintains high spin rates throughout the flight, leading to overestimated lift and drag forces in the latter stages of the trajectory.

## Impact
- **Accuracy:** Significant overestimation of carry distance for high-spin shots.
- **Realism:** Trajectory shape (apex location, descent angle) will be physically incorrect.
- **Simulation:** "Ballooning" effect will be exaggerated.

## Affected Files
- `shared/python/ball_flight_physics.py`

## Recommended Fix
1.  Implement a spin decay differential equation in `_flight_dynamics_core`.
2.  Use the `SPIN_DECAY_RATE_S` constant from `physics_constants.py` (value 0.05/s) or a more sophisticated Reynolds-dependent decay model.
3.  Update the RK4 state vector to include `omega` if it varies dynamically, or apply an analytical decay function if assuming decoupling.

## References
- Penner, A.R. (2003) "The physics of golf"
