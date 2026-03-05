---
title: Critical Error in Rigid Body Impact Model Effective Mass Calculation
labels: physics-error, critical
assignees: physics-team
---

## Description
The `RigidBodyImpactModel` in `src/shared/python/physics/impact_model.py` currently uses a scalar effective mass approximation for off-center hits:

```python
return 1.0 / (1.0 / m_club + r_offset**2 / club_moi)
```

This implementation ignores the full 3D inertia tensor of the clubhead and the directional component of the impact force vector relative to the center of gravity (CG).

## Expected Behavior
The impact model should implement the full 3D impulse-momentum equations:
$J = (M^{-1} + (r \times n)^T I^{-1} (r \times n))^{-1} (1+e) v_{rel}$

Where:
- $M^{-1}$ is the inverse mass matrix
- $I^{-1}$ is the inverse inertia tensor
- $n$ is the impact normal
- $r$ is the vector from CG to impact point

## Impact
- Significantly inaccurate ball speed and spin rates for off-center hits (gear effect), leading to incorrect carry distances and dispersion patterns.

## Recommended Fix
1. Modify `PreImpactState` to include the full inertia tensor (3x3 matrix) instead of scalar MOI.
2. Implement the 3D matrix-based impulse calculation in `RigidBodyImpactModel.solve`.
