---
type: physics-error
severity: critical
status: open
created: 2026-02-05
file: src/shared/python/impact_model.py
---

# Critical: Impact Model Ignores Clubhead MOI

## Description

The `RigidBodyImpactModel` treats the clubhead as a point mass, using a 1D impulse-momentum equation:

```python
m_eff = (m_ball * m_club) / (m_ball + m_club)
```

This completely ignores the Moment of Inertia (MOI) of the clubhead. In reality, an off-center hit causes the clubhead to twist, which:

1.  Reduces the effective mass at the impact point (reducing ball speed).
2.  Imparts gear effect spin to the ball.

Currently, gear effect is added via a separate, empirical function `compute_gear_effect_spin` with arbitrary constants (`h_scale=100.0`), which is not physically robust.

## Expected Behavior

The impact solver should implement a 6-DOF rigid body collision (Newton-Euler equations) that accounts for:

- Clubhead Mass
- Clubhead Inertia Tensor (MOI)
- Impact Location relative to Center of Gravity (CG)

## Recommended Fix

1.  Update `PreImpactState` to include clubhead Inertia Tensor.
2.  Refactor `RigidBodyImpactModel.solve` to compute the impulse $J$ using the rigid body formulation:
    $$ J = \frac{-(1+e)v*{rel} \cdot n}{ \frac{1}{m*{ball}} + \frac{1}{m\_{club}} + (I^{-1} (r \times n)) \times r \cdot n } $$
    Where $r$ is the vector from club CG to impact point.
3.  Derive gear effect spin directly from the tangential impulse components (friction) and clubhead rotation.
