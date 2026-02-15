# Physics Issue: Simplified Impact Effective Mass

**ID**: ISSUE_P005
**Category**: Physics Accuracy / Equipment
**Status**: Open
**Severity**: Medium

## Description
The file `src/shared/python/physics/impact_model.py` uses a scalar approximation for effective mass during off-center hits: `1 / (1/m + r^2/I)`. This formula assumes a 1D impact along a principal axis or ignores the full 3D inertia tensor.

## Physics Violation
The effective mass of a rigid body during an impact is a tensor quantity that depends on the direction of impact relative to the center of mass (CM) and the full inertia matrix ($I$). The formula used is only valid if the impact point is along one of the principal axes of inertia or if the inertia is isotropic.

## Impact
-   **Forgiveness**: Overestimates ball speed on toe/heel hits.
-   **Spin**: Underestimates gear effect spin magnitude due to incorrect torque calculation.

## Recommended Fix
1.  **Implement 3D Inertia Tensor**: Use the full 3x3 inertia matrix `I` for the clubhead.
2.  **Calculate Effective Mass**: Compute the effective mass matrix at the impact point $P$:
    $M_{eff} = (I^{-1} \cdot (r \times n)) \times (r \times n) + \frac{1}{m} \mathbf{1}$ (inverse of this).
    Where $r$ is the vector from CM to $P$, and $n$ is the impact normal.
