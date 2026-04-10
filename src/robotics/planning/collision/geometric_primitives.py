"""Geometric primitives for collision detection.

This module provides basic geometric shapes and collision algorithms:
- Sphere, Box, Capsule, Cylinder primitives
- Distance computation between primitives
- Collision detection between primitives

Design by Contract:
    All primitives must have positive dimensions.
    All transformations must be valid (finite, proper rotation).
"""

from ._distance_queries import (
    check_primitive_collision,
    compute_primitive_distance,
)
from ._primitive_shapes import Box, Capsule, ConvexHull, Cylinder, Sphere
from ._primitives_base import GeometricPrimitive

__all__ = [
    "GeometricPrimitive",
    "Sphere",
    "Box",
    "Capsule",
    "Cylinder",
    "ConvexHull",
    "compute_primitive_distance",
    "check_primitive_collision",
]
