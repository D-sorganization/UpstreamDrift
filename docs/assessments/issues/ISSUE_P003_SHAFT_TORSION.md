# Physics Issue: Missing Shaft Torsion

**ID**: ISSUE_P003
**Category**: Physical Feature Gap
**Status**: Open
**Severity**: High

## Description
The file `src/shared/python/physics/flexible_shaft.py` defines `BeamElement` with 4 degrees of freedom (DOF) per element (2 displacements, 2 rotations), modeling only transverse bending. Torsional twist around the shaft's longitudinal axis is not implemented.

## Physics Gap
Shaft "torque" (torsional stiffness) is a critical specification for golf shafts. During the swing, the center of gravity (CG) of the clubhead is offset from the shaft axis, creating a moment that twists the shaft open on the downswing and closed at impact. This dynamic effect influences the lie angle and face angle at impact, affecting ball direction.

## Impact
-   **Equipment Modeling**: Cannot simulate high-torque vs. low-torque shafts.
-   **Ball Flight**: Misses potential "dynamic lie" effects which can alter launch direction.

## Recommended Fix
1.  **Add Torsional DOF**: Expand `BeamElement` to 6 DOF (add axial rotation at each node).
2.  **Implement Torsional Stiffness**: Add `GJ` (shear modulus * polar moment of inertia) to the element stiffness matrix.
3.  **Update Load Application**: Allow applying moments about the shaft axis.
