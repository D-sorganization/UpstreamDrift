"""
Spatial Algebra Module

This module implements Featherstone's spatial vector algebra for rigid body dynamics.
Spatial vectors are 6D vectors representing motion (velocity, acceleration) and
force (wrenches) in 3D space.

Key concepts:
- Spatial motion vectors: [angular_velocity; linear_velocity]
- Spatial force vectors: [moment; force]
- Spatial transformations (Plücker transforms)
- Spatial cross products
- Spatial inertia matrices

References:
    Featherstone, R. (2008). Rigid Body Dynamics Algorithms.
    Cambridge University Press.
"""

from .inertia import mci, transform_spatial_inertia
from .joints import jcalc
from .spatial_vectors import crf, crm, cross_force, cross_motion, spatial_cross
from .transforms import inv_xtrans, xlt, xrot, xtrans

__all__ = [
    "crf",
    # Spatial cross products
    "crm",
    "cross_force",
    "cross_motion",
    "inv_xtrans",
    # Joints
    "jcalc",
    # Inertia
    "mci",
    "spatial_cross",
    "transform_spatial_inertia",
    "xlt",
    # Transformations
    "xrot",
    "xtrans",
]
