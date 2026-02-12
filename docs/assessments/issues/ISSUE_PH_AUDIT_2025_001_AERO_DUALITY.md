# Critical Issue: Inconsistent Aerodynamic Models

**Status:** Open
**Priority:** High
**Labels:** physics-gap, high
**Date Identified:** 2025-02-04
**Author:** PHYSICS AUDITOR

## Description

The codebase currently maintains two parallel, inconsistent implementations of golf ball aerodynamics:

1.  **Polynomial Model:** Implemented in `BallFlightSimulator` (`src/shared/python/ball_flight_physics.py`). Uses `BallProperties` with `cd0, cd1, cd2` and `cl0, cl1, cl2` (Bearman & Harvey style polynomials).
2.  **Saturation/Exponential Model:** Implemented in `AerodynamicsEngine` (`src/shared/python/aerodynamics.py`) and used by `EnhancedBallFlightSimulator`. Uses `AerodynamicsConfig` with exponential saturation for lift (`1 - exp(-s/0.1)`) and linear saturation for Magnus force.

These two models represent fundamentally different physical approximations. A simulation run with identical launch conditions will produce different trajectories depending on which simulator class is instantiated. This violates the Single Source of Truth principle for physics.

## Impact

-   **Reproducibility:** Users may get different results for the "same" shot depending on whether they use the "basic" or "enhanced" simulator.
-   **Maintenance:** Updates to coefficients in one model (e.g., `BallProperties`) do not propagate to the other (`AerodynamicsConfig`), leading to divergence over time.
-   **Validation:** Validating one model against trackman data does not validate the other.

## Affected Files

-   `src/shared/python/ball_flight_physics.py`
-   `src/shared/python/aerodynamics.py`

## Recommended Fix

1.  **Deprecate Polynomial Logic:** Remove the polynomial coefficient logic (`_calculate_accel_core` with `cd0, cd1...`) from `ball_flight_physics.py`.
2.  **Unify Backend:** Refactor `BallFlightSimulator` to use `AerodynamicsEngine` internally, even for "basic" simulations (perhaps with a default "polynomial-equivalent" configuration if backward compatibility is strictly required, though unified physics is preferred).
3.  **Single Config:** Consolidate `BallProperties` and `AerodynamicsConfig` into a single configuration object that drives the `AerodynamicsEngine`.
