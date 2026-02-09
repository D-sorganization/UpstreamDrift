# Critical Issue: GRF Implementation Incorrectly Sums Gravity

**Status:** Open
**Priority:** Critical
**Labels:** physics-error, critical
**Date Identified:** 2026-02-16
**Author:** JULES (Gap Analysis)

## Description

The `extract_grf_from_contacts` function in `src/shared/python/ground_reaction_forces.py` contains a critical inaccuracy in how it calculates Ground Reaction Forces (GRF).

Instead of querying the physics engine for actual dynamic contact forces (which would account for impact, acceleration, and weight transfer), the implementation simply sums the magnitude of gravity forces acting on the contacting bodies.

```python
    # Approximate: GRF opposes gravity at contact points
    if len(g) > 0:
        total_force[2] += abs(np.sum(g))
```

This approximation is only valid for a static object in equilibrium. For a dynamic golf swing, where vertical forces can exceed 1.5-2.0x body weight due to acceleration, this implementation completely fails to capture the biomechanics of the swing.

## Impact

- **Force Plate Analysis:** GRF data will be completely flat (equal to system weight) regardless of the golfer's movement.
- **COP Accuracy:** Center of Pressure calculations may be incorrect as they rely on the distribution of vertical force, which is being approximated.
- **Simulation Validity:** Any metrics derived from GRF (like vertical power or efficiency) will be invalid.

## Affected Files

- `src/shared/python/ground_reaction_forces.py` (`extract_grf_from_contacts`)

## Recommended Fix

1.  Update `extract_grf_from_contacts` to retrieve actual contact force vectors from the `PhysicsEngine` interface.
2.  If the physics engine interface does not support direct contact force queries, update the `PhysicsEngine` abstract base class to include a `get_contact_forces(body_name)` method.
3.  Implement this method in the respective engine adapters (MuJoCo, Drake, etc.).
