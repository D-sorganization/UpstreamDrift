"""
Mesh processing module for humanoid character builder.

Provides mesh-based inertia calculation, mesh loading/processing,
and primitive shape fallbacks.
"""

from src.shared.python.humanoid_character_builder.mesh.collision_geometry import (
    CollisionGeometry,
    CollisionGeometryGenerator,
)
from src.shared.python.humanoid_character_builder.mesh.inertia_calculator import (
    InertiaMode,
    InertiaResult,
    MeshInertiaCalculator,
)
from src.shared.python.humanoid_character_builder.mesh.mesh_processor import (
    LODGenerationResult,
    LODGenerator,
    LODLevel,
    MeshProcessor,
    MeshSegmentResult,
)
from src.shared.python.humanoid_character_builder.mesh.primitive_inertia import (
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
    "LODGenerator",
    "LODLevel",
    "LODGenerationResult",
    "CollisionGeometry",
    "CollisionGeometryGenerator",
]
