---
title: Missing Shaft Torsional Dynamics
labels: physics-gap, high
assignees: physics-team
---

## Description
The `flexible_shaft.py` module currently implements a Finite Element beam model based on Euler-Bernoulli theory, which models bending (flex) but explicitly excludes torsion (twisting).

```python
# TODO: Implement Torsional Dynamics (current Euler-Bernoulli beam model ignores torsional twisting).
# TODO: Support Asymmetric Cross-Sections (modeling spine alignment and manufacturing tolerances).
```

## Impact
- **Dispersion Inaccuracy**: Shaft torque is a primary contributor to clubface closing rate variations. Without it, left/right dispersion predictions are fundamentally limited.
- **Spine Alignment**: Cannot model shaft spine (asymmetric bending stiffness) or installation orientation effects (FLO).
- **Feel**: Torsional feedback is a key component of player "feel" which is missing.

## Recommended Fix
1. Extend the `BeamElement` and `FiniteElementShaftModel` to include torsional degrees of freedom (rotation about the shaft axis).
2. Add shear modulus ($G$) and torsional constant ($J$) to `ShaftProperties`.
3. Couple bending and torsion if modeling asymmetric shafts (spine).
