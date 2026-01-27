# Critical Issue: Unrealistic Shaft Stiffness Magnitude

**Status:** Open
**Priority:** Critical
**Labels:** physics-error, critical
**Date Identified:** 2026-01-27
**Author:** PHYSICS AUDITOR

## Description
The flexible shaft model (`src/shared/python/flexible_shaft.py`) calculates the Bending Stiffness ($EI$) based on geometric inputs: a 1mm wall thickness and an isotropic Young's Modulus of 130 GPa (Graphite).

This calculation yields $EI$ values of approximately 140 $N \cdot m^2$. In reality, golf shafts are thin-walled composite structures with typical $EI$ values ranging from 2 to 6 $N \cdot m^2$. The current implementation results in a shaft that is approximately 30 times too stiff.

Additionally, `src/shared/python/equipment.py` provides default stiffness values of [180.0, 150.0, 120.0], which reinforce this unrealistic magnitude.

## Impact
- **Simulation:** The shaft behaves effectively as a rigid body.
- **Features:** "Kick" velocity, deflection, and droop effects will be negligible.
- **User Experience:** Users selecting "Regular" or "Stiff" flex will see no difference in performance or visualization compared to a steel bar.

## Affected Files
- `src/shared/python/flexible_shaft.py`
- `src/shared/python/equipment.py`

## Recommended Fix
1.  **Calibration:** Update `create_standard_shaft` to use a realistic effective wall thickness (e.g., ~0.1-0.2mm equivalent) or effective modulus to match standard EI profiles.
2.  **Direct Input:** Modify `ShaftProperties` to accept an explicit $EI(x)$ profile array, bypassing the geometric calculation if provided.
3.  **Data Update:** Update `equipment.py` constants to reflect realistic ranges (e.g., [6.0, 4.0, 2.0] $N \cdot m^2$).
