---
title: Lack of Uncertainty Propagation in Statistical Methods
labels: physics-gap, high
assignees: physics-team
---

## Description
Across the entire Physics Module, there is no implementation of Monte Carlo or analytical uncertainty propagation for input parameters (e.g., clubhead speed +/- 2 mph). Inputs are deterministic.

## Expected Physics
Inputs should support distributions to reflect sensor noise or human variation, allowing the output to include a confidence interval.

## Actual Implementation
Deterministic single values are used, yielding single values as output.

## Impact
Users receive a single "perfect" number rather than a confidence interval, which is misleading for coaching and performance assessment.

## Recommended Fix
Incorporate Monte Carlo frameworks around physics calculations to support parameterized uncertainty propagation.
