# Low Priority Issue: Constant Air Density in Ball Flight

**Status:** Open
**Priority:** Low
**Labels:** physics-gap, low
**Date Identified:** 2026-01-31
**Author:** PHYSICS AUDITOR

## Description

The `BallFlightSimulator` in `src/shared/python/ball_flight_physics.py` calculates a `const_term` (Drag/Lift scaling factor) based on a single, static air density value (`self.environment.air_density`).

```python
const_term = (0.5 * self.environment.air_density * self.ball.cross_sectional_area) / self.ball.mass
```

This constant is used throughout the RK4 integration loop. However, for a driver shot reaching an apex of 30-40 meters (or higher for wedges), air density changes slightly. More importantly, if the simulation supports "course altitude" (e.g., playing in Denver vs. Sea Level), the `EnvironmentalConditions` class has an `altitude` field, but it is **not used** to adjust the `air_density` passed to the solver. The user must manually calculate the density at that altitude and pass it in.

Furthermore, dynamic density variation with height ($z$) is standard in high-fidelity trajectory models (ISA Atmosphere model).

## Impact

- **Accuracy:** Slight error in carry distance for high trajectories.
- **Usability:** Users setting `altitude` in `EnvironmentalConditions` will expect the physics to update automatically, but it does not.

## Affected Files

- `src/shared/python/ball_flight_physics.py`

## Recommended Fix

1.  Update `EnvironmentalConditions` to automatically calculate `air_density` based on `temperature`, `pressure`, and `altitude` if not explicitly provided.
2.  (Optional) Implement dynamic air density $\rho(z)$ inside `_calculate_accel_core` for research-grade precision, though a constant density at launch altitude is usually sufficient for golf (max height < 50m).
