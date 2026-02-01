---
title: Impact Model Ignores Clubhead Moment of Inertia (Point Mass Approximation)
labels: physics-error, critical
---

# Issue Description
The `RigidBodyImpactModel` in `src/shared/python/impact_model.py` treats the clubhead as a point mass, effectively ignoring its Moment of Inertia (MOI).

**File:** `src/shared/python/impact_model.py`
**Line:** ~150 (RigidBodyImpactModel.solve)

# Expected Physics
A golf clubhead is a rigid body with mass and an inertia tensor. Off-center impacts (toe/heel) exert a torque about the Center of Gravity (CG), causing the clubhead to rotate during the collision. This rotation consumes energy that would otherwise go into ball speed, reducing the "Smash Factor" for off-center hits.

# Actual Implementation
The code uses the 1D effective mass formula:
```python
m_eff = (m_ball * m_club) / (m_ball + m_club)
```
This formula is only valid for central impacts where no rotation is induced.

# Impact
- **Accuracy:** Overestimates ball speed (Smash Factor) for off-center hits.
- **Physicality:** The clubhead does not twist post-impact in the simulation, which is a fundamental behavior of golf clubs (and the reason for "High MOI" club designs).

# Recommended Fix
Implement a full 3D impulse-momentum model that includes the Clubhead Inertia Tensor.
The effective mass at the impact point $P$ should be:
$$ \frac{1}{m_{eff}} = \frac{1}{m_{club}} + (r \times n)^T I^{-1} (r \times n) $$
where $r$ is the vector from CG to impact point, $n$ is the impact normal, and $I$ is the inertia tensor.
