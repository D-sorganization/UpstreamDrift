"""
Unified inertia calculation for model generation.

This module provides a single, comprehensive inertia calculation
system supporting multiple modes:
- Primitive approximation (analytical formulas)
- Mesh-based with uniform density
- Mesh-based with specified mass
- Manual override
"""

from model_generation.inertia.calculator import (
    InertiaCalculator,
    InertiaMode,
    InertiaResult,
)
from model_generation.inertia.primitives import (
    box_inertia,
    capsule_inertia,
    cylinder_inertia,
    ellipsoid_inertia,
    sphere_inertia,
)
from model_generation.inertia.spatial import (
    mcI,
    spatial_inertia_to_urdf,
    transform_spatial_inertia,
)

__all__ = [
    # Main calculator
    "InertiaCalculator",
    "InertiaMode",
    "InertiaResult",
    # Primitive formulas
    "box_inertia",
    "cylinder_inertia",
    "sphere_inertia",
    "capsule_inertia",
    "ellipsoid_inertia",
    # Spatial inertia
    "mcI",
    "transform_spatial_inertia",
    "spatial_inertia_to_urdf",
]
