"""
Mesh processing module for humanoid character builder.

Provides mesh-based inertia calculation, mesh loading/processing,
and primitive shape fallbacks.
"""

from humanoid_character_builder.mesh.inertia_calculator import (
    InertiaMode,
    InertiaResult,
    MeshInertiaCalculator,
)
from humanoid_character_builder.mesh.mesh_processor import (
    MeshProcessor,
    MeshSegmentResult,
)
from humanoid_character_builder.mesh.primitive_inertia import (
    PrimitiveInertiaCalculator,
    PrimitiveShape,
)

__all__ = [
    "InertiaMode",
    "InertiaResult",
    "MeshInertiaCalculator",
    "PrimitiveInertiaCalculator",
    "PrimitiveShape",
    "MeshProcessor",
    "MeshSegmentResult",
]
