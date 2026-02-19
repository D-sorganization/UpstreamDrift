---
title: Critical: Incorrect GRF Fallback Mechanism
labels: physics-error, critical
assignees: physics-team
---

## Description
The `extract_grf_from_contacts` function in `src/shared/python/physics/ground_reaction_forces.py` incorrectly sums gravity forces ($W = mg$) when contact data is unavailable, instead of accounting for dynamic acceleration ($F = m(g + a)$).

## Reproduction
1. Initialize a physics engine simulation in free fall (no contacts).
2. Call `extract_grf_from_contacts`.
3. Observe a non-zero vertical force approximately equal to the body weight.

## Expected Behavior
In free fall, the GRF should be exactly zero. During dynamic movement, the fallback should use center-of-mass acceleration if available.

## Impact
This causes significant discontinuities in GRF data during high-speed swings where contact forces can exceed 200% body weight, leading to incorrect biomechanical analysis.
