---
title: "Physics Gap: Impact Model Ignores Club Moment of Inertia"
labels: ["physics-gap", "high"]
assignees: ["physics-lead"]
---

## Description
The `RigidBodyImpactModel` in `src/shared/python/impact_model.py` uses a simple 1D impulse-momentum formulation based on effective mass ($m_{eff} = \frac{m_{ball} m_{club}}{m_{ball} + m_{club}}$). It treats the clubhead as a point mass $m_{club}$.

This ignores the clubhead's Moment of Inertia (MOI). In reality, off-center impacts cause the clubhead to rotate around its Center of Gravity (CG), significantly reducing the effective mass presented to the ball at the impact point.

## Impact
*   **Smash Factor Error:** Overestimates ball speed for toe and heel hits ("sweet spot" seems infinitely large).
*   **Gear Effect Accuracy:** While gear effect spin is patched in via heuristics, the fundamental loss of linear momentum due to club recoil is missing.

## Affected Files
*   `src/shared/python/impact_model.py`: `RigidBodyImpactModel.solve`

## Recommended Fix
Implement the full 3D rigid body collision equations:
1.  Define Club Inertia Tensor $I_{club}$.
2.  Calculate effective mass at impact point $P$:
    $$ \frac{1}{M_{eff}} = \frac{1}{M_{club}} + (r \times n)^T I_{club}^{-1} (r \times n) $$
    (where $r$ is vector from CG to impact point $P$, $n$ is impact normal).
3.  Use this $M_{eff}$ in the impulse calculation.
