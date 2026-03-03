---
title: Empirical Gear Effect Calculation is Oversimplified
labels: physics-error, critical
assignees: physics-team
---

## Description
The `compute_gear_effect_spin` function in `src/shared/python/physics/impact_model.py` (Line 491) computes spin from the gear effect for off-center impact using a linear coefficient (`gear_factor = 0.5`) rather than deriving it properly from the physics of the collision (friction and tangential impulse).

## Expected Physics
Spin should be driven dynamically by clubface friction and off-center impulse, incorporating the full 3D interaction.

## Actual Implementation
A static parameter (`gear_factor = 0.5`) is scaled by horizontal and vertical offsets to estimate the imparted spin.

## Impact
Less accurate prediction of spin axis tilt for complex face geometries (e.g., "Twist Face"), leading to incorrect dispersion models for off-center hits.

## Recommended Fix
Derive gear effect spin dynamically from impact geometry and friction properties rather than relying on a static scaling factor.
