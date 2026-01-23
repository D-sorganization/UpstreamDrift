# High Priority Issue: Oversimplified Golf Ball MOI

**Status:** Open
**Priority:** High
**Labels:** physics-gap, high
**Date Identified:** 2026-01-22
**Author:** PHYSICS AUDITOR

## Description
The impact model (`shared/python/impact_model.py`) hardcodes the golf ball Moment of Inertia (MOI) using the formula for a solid uniform sphere: $$I = \frac{2}{5}MR^2$$.

Modern golf balls are multi-layer constructions with varying densities (dense core, lighter mantle/cover). This results in a Moment of Inertia that differs from a solid sphere. Typically, the normalized MOI coefficient is closer to 0.38-0.39 rather than 0.40.

## Impact
- **Spin Rate:** Inaccurate calculation of spin generation from impact friction.
- **Roll:** Inaccurate modeling of putting roll dynamics.

## Affected Files
- `shared/python/impact_model.py`

## Recommended Fix
1.  Update `GOLF_BALL_MOMENT_INERTIA` to be configurable or use a more accurate default constant (e.g., $$0.38 \cdot MR^2$$).
2.  Allow `BallProperties` to define MOI.
