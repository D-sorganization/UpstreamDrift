# Critical Issue: Impact Solver Treats Clubhead as Point Mass

**Status:** Open
**Priority:** Critical
**Labels:** physics-error, critical
**Date Identified:** 2026-01-31
**Author:** PHYSICS AUDITOR

## Description
The `RigidBodyImpactModel` in `src/shared/python/impact_model.py` treats the clubhead as a point mass for the purpose of the impulse-momentum conservation equations.

The solver uses:
```python
m_club = pre_state.clubhead_mass
j = (1 + e) * m_eff * v_approach  # Standard 1D collision
```
This ignores the clubhead's Moment of Inertia (MOI) and the location of the impact relative to the Center of Gravity (CG). In reality, an off-center hit causes the clubhead to rotate, which reduces the effective mass at the impact point and transfers less energy to the ball (ball speed drop-off).

Currently, this effect is completely missing from the core solver. The module relies on a separate "heuristic" (`compute_gear_effect_spin`) to fake the spin results (see Issue 011), but the *velocity* result (ball speed) remains incorrect for off-center hits because it behaves as if the impact was center-face (infinite MOI).

## Impact
- **Ball Speed:** Overestimated for off-center strikes (smash factor remains too high).
- **Forgiveness:** Impossible to model "high MOI" vs "low MOI" clubheads correctly.
- **Accuracy:** The fundamental energy transfer mechanics are flawed for anything other than a perfect sweet-spot strike.

## Affected Files
- `src/shared/python/impact_model.py` (`RigidBodyImpactModel.solve`, `SpringDamperImpactModel.solve`)

## Recommended Fix
1.  Update `PreImpactState` to include Clubhead MOI tensor (or at least principal moments $I_{xx}, I_{zz}$).
2.  Implement full 3D rigid body collision physics:
    - Compute "Effective Mass" at impact point $P$: $1/m_{eff} = 1/m_{ball} + 1/m_{club} + (r \times n)^2 / I_{club}$.
    - Use this $m_{eff}$ in the impulse calculation.
3.  Calculate clubhead post-impact angular velocity changes $\Delta \omega = (r \times J) / I$.
