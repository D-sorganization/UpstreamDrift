---
title: Critical: Hardcoded Mass in Efficiency Score
labels: physics-error, critical
assignees: physics-team
---

## Description
The `compute_swing_profile` method in `src/shared/python/analysis/reporting.py` calculates Kinetic Energy using a hardcoded mass of 1.0 kg (`0.5 * 1.0 * v^2`), whereas a typical driver head is ~0.2 kg.

## Reproduction
1. Run `ReportingMixin.compute_swing_profile` with known clubhead speed.
2. Observe an Efficiency Score that is approximately 5x higher than physically possible.

## Expected Behavior
The calculation should use the actual mass of the clubhead from the simulation state or configuration (default ~0.2 kg).

## Impact
Massively inflates the reported "Efficiency Score", rendering it scientifically invalid and misleading for users.
