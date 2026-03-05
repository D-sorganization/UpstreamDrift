---
title: Lack of Uncertainty Propagation in Statistical Methods
labels: physics-gap, high
assignees: physics-team
---

## Description
The statistical methods across the Physics Module lack uncertainty propagation for input parameters (e.g., Monte Carlo simulations). Outputs are currently deterministic.

## Expected Behavior
The statistical models should quantify output uncertainty based on variations in input parameters.

## Impact
Users receive a single deterministic result rather than a confidence interval, leading to a misleading sense of precision.

## Recommended Fix
Implement Monte Carlo methods or analytical uncertainty propagation.
