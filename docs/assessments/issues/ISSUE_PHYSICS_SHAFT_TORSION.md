---
title: Missing Shaft Torsional Dynamics in Flexible Shaft Model
labels: physics-gap, high
assignees: physics-team
---

## Description
The `FlexibleShaft` model currently uses an Euler-Bernoulli beam formulation that accounts for bending but explicitly ignores torsion (twisting).

## Expected Behavior
The shaft model must include torsional degrees of freedom to accurately simulate the dynamic twisting of the shaft during the swing.

## Impact
Unable to model "spine alignment" effects or the clubface closing rate variations due to shaft torque, which is a primary driver of left/right dispersion.

## Recommended Fix
Add torsional degrees of freedom to the Finite Element model.
