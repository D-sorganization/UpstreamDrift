---
title: "Physics Error: Shaft Bending Stiffness (EI) Significantly Overestimated"
labels: ["physics-error", "critical"]
assignees: ["physics-lead"]
---

## Description
The default `ShaftProperties` in `src/shared/python/flexible_shaft.py` utilize a Young's Modulus (`GRAPHITE_E`) of 130 GPa, assuming isotropic behavior for the graphite material. When combined with the default geometric parameters (Tip 8.5mm, Butt 15mm, Wall 1mm), this results in a calculated Bending Stiffness ($EI$) of approximately 20-140 $Nm^2$.

Real-world golf shafts exhibit an effective $EI$ in the range of 2 $Nm^2$ (Ladies/Senior flex) to 7 $Nm^2$ (X-Stiff/Long Drive), primarily due to anisotropic fiber orientation (bias plies) and lower effective modulus of the composite matrix.

## Impact
The current model simulates a shaft that is 10-50x stiffer than reality. The shaft behaves effectively as a rigid steel bar, preventing the simulation of:
*   Dynamic "kick" at impact.
*   Lag loading during the downswing.
*   Droop and lead/lag deflection effects.

## Affected Files
*   `src/shared/python/flexible_shaft.py`: `create_standard_shaft`, `compute_EI_profile`

## Expected Behavior
The calculated $EI$ profile should average around 3-5 $Nm^2$ for a standard Stiff shaft.

## Recommended Fix
1.  **Quick Fix:** Reduce `GRAPHITE_E` to an effective modulus of ~50 GPa in `create_standard_shaft`.
2.  **Robust Fix:** Change `ShaftProperties` to accept an explicit `EI_profile` array input, bypassing the geometric derivation from isotropic modulus, as EI is the standard industry specification for shafts.
