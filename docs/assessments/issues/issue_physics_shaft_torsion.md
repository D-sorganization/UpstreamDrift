---
title: Shaft Model Lacks Torsion and Out-of-Plane Bending
labels: physics-gap, high
---

# Issue Description
The `FiniteElementShaftModel` implements 2D Euler-Bernoulli beam elements (Planar Bending). It lacks degrees of freedom for Torsion (twist about the axis) and bending in the second transverse plane.

**File:** `src/shared/python/flexible_shaft.py`
**Line:** `FiniteElementShaftModel` class

# Expected Physics
Golf shafts bend in two planes ("Kick" and "Droop") and twist ("Torque") during the swing.
- **Droop:** Caused by the Center of Gravity (CG) of the clubhead being offset from the shaft axis.
- **Twist:** Caused by the CG offset and the "closing" torque required to square the face.
These effects significantly influence the dynamic face angle and dynamic loft at impact.

# Actual Implementation
The model has 2 DOFs per node (transverse deflection `w` and rotation `theta`). It assumes all forces and motion occur in a single plane.

# Impact
- **Accuracy:** Unable to predict dynamic face closure or lie angle changes (droop).
- **Usability:** Cannot be used to analyze "low torque" vs "high torque" shafts.

# Recommended Fix
Upgrade the Finite Element model to use 6-DOF Spatial Beam Elements (or at least 4-DOF: 2 Bending + 1 Axial + 1 Torsion). This requires expanding the stiffness and mass matrices to 12x12 per element (for 6-DOF) and handling 3D coordinate transformations.
