---
type: physics-gap
severity: high
status: open
created: 2026-02-05
file: src/shared/python/flexible_shaft.py
---

# High: Shaft Model Missing Torsional Stiffness

## Description
The `FiniteElementShaftModel` implements Euler-Bernoulli beam elements which model transverse bending (EI) but ignore torsional stiffness (GJ).

Golf shafts twist significantly during the swing due to the offset Center of Gravity (CG) of the clubhead. This "twisting" (torque) affects the dynamic face angle at impact. Neglecting this leads to inaccurate predictions of launch direction and spin axis.

## Expected Behavior
The shaft model should include a torsional degree of freedom (twist angle) at each node.

## Recommended Fix
1.  Update `ShaftProperties` to include Shear Modulus ($G$) or Torsional Stiffness ($GJ$).
2.  Update `BeamElement` to include torsional stiffness.
3.  Expand the element stiffness matrix to 12x12 (6 DOF per node) or at least add the torsional DOF to the existing formulation (making it 3 DOF per node: bending deflection, bending slope, twist angle).
4.  Update `apply_load` to handle moments about the shaft axis.
