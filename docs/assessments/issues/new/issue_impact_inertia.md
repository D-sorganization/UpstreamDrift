---
title: High: Inaccurate Effective Mass in Impact Model
labels: physics-error, high
assignees: physics-team
---

## Description
The `RigidBodyImpactModel` in `src/shared/python/physics/impact_model.py` uses a scalar approximation for Moment of Inertia (MOI) and effective mass, ignoring the full 3D inertia tensor.

## Reproduction
1. Simulate an off-center impact with a significant vertical offset (e.g., high on face).
2. Compare the result with a horizontal offset (e.g., toe hit) using a clubhead with distinct $I_{xx}$ and $I_{yy}$ values.
3. Observe identical effective mass calculations due to the scalar approximation $m_{eff} = 1 / (1/m + r^2/I)$.

## Expected Behavior
Vertical and horizontal off-center hits should result in different effective masses and ball speeds due to the different principal moments of inertia of the clubhead (Twist Face effect).

## Impact
Reduces the accuracy of ball speed and launch conditions for miss-hits, particularly for modern drivers with optimized MOI distributions.
