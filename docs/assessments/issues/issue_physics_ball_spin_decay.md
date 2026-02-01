---
title: Ball Flight Simulation Lacks Spin Decay
labels: physics-gap, high
---

# Issue Description
The `BallFlightSimulator` treats ball spin (`omega`) as a constant throughout the entire trajectory.

**File:** `src/shared/python/ball_flight_physics.py`
**Line:** `_calculate_accel_core` function

# Expected Physics
Golf balls experience spin decay due to air viscosity. The spin rate decreases exponentially over time. A typical driver shot launching at 2500 rpm might land with < 2000 rpm. High-spin wedge shots decay even faster.

# Actual Implementation
The state vector is size 6 (Position, Velocity). Spin is passed as a constant parameter `omega` to the derivative function. There is no differential equation for `d(omega)/dt`.

# Impact
- **Accuracy:** Overestimates lift and drag in the latter half of the flight.
- **Trajectory:** Resulting trajectories will "balloon" (stay high too long) and land at steeper angles than reality.
- **Curvature:** Slice/Hook curvature will be exaggerated at the end of the flight.

# Recommended Fix
1. Expand the state vector to size 9 (Position, Velocity, Spin Vector).
2. Implement the spin decay moment equation:
   $$ \tau = -c_{decay} \cdot \omega $$
   (or a more complex Reynolds-dependent decay model).
3. Update `_flight_dynamics_step` to compute angular acceleration and update spin.
