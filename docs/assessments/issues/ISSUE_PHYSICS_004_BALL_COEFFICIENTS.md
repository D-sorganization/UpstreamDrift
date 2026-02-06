---
type: physics-consistency
severity: medium
status: open
created: 2026-02-05
file: src/shared/python/ball_flight_physics.py
---

# Medium: Inconsistent Ball Flight Coefficients

## Description

There is a discrepancy between the aerodynamic coefficients used in different parts of the codebase:

- `src/shared/python/ball_flight_physics.py`: `BallProperties` uses hardcoded polynomial coefficients (e.g., `cd0=0.21`, `cl1=0.38`).
- `src/shared/python/physics_constants.py`: Defines `GOLF_BALL_DRAG_COEFFICIENT = 0.25`.
- `src/shared/python/aerodynamics.py`: `DragModel` defaults to `0.25` (via constants).

This inconsistency means `BallFlightSimulator` (basic) and `EnhancedBallFlightSimulator` (advanced) may produce different trajectories for the "same" ball.

## Expected Behavior

All flight models should reference a single source of truth for physical constants, or explicitly expose configuration for these values.

## Recommended Fix

1.  Update `BallProperties` to accept coefficients in `__init__` and default to values from `physics_constants.py` or a unified config.
2.  If the polynomial model in `ball_flight_physics.py` is superior, move those constants to `physics_constants.py` with proper citations.
