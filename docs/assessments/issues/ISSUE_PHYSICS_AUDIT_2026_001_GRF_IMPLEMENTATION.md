---
type: physics-error
severity: critical
status: open
created: 2026-02-08
file: src/shared/python/ground_reaction_forces.py
author: PHYSICS AUDITOR
---

# Critical: Ground Reaction Force Extraction is a Placeholder

## Description

The function `extract_grf_from_contacts` in `src/shared/python/ground_reaction_forces.py` contains a placeholder implementation that does not actually calculate Ground Reaction Forces (GRF) from physical contacts.

```python
# For now, estimate contact force from gravity compensation
# This is a simplified approach - real implementation would
# query actual contact forces from the physics engine

# Approximate: GRF opposes gravity at contact points
if len(g) > 0:
    total_force[2] += abs(np.sum(g))
```

It seemingly just sums the gravity force, assuming static equilibrium, and completely ignores:
1.  Dynamic forces (acceleration of body segments).
2.  Actual contact constraints (friction, normal force) from the physics engine.
3.  Center of Pressure (COP) distribution (it defaults to `[0,0,0]` or simplistic average).
4.  Horizontal forces (shear/friction) are ignored.

## Impact

- **Biomechanics:** Any analysis relying on GRF (e.g., efficiency, power generation) is invalid for dynamic movements like a golf swing.
- **Accuracy:** The values will be wrong whenever the golfer is accelerating (which is always during a swing).
- **Compliance:** Fails Guideline E5 (GRF Accuracy).

## Recommended Fix

1.  Implement actual query of contact forces from the physics engine (e.g., MuJoCo `mjData.contact`).
2.  Sum the contact forces for the specified bodies (e.g., feet).
3.  Calculate the COP based on the weighted average of contact points and their normal forces.
4.  Calculate Friction forces (shear) properly.
