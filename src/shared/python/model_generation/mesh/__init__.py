"""Mesh processing utilities for model generation.

Re-exports mesh processing components from humanoid_character_builder.mesh.
"""

from __future__ import annotations

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
    MeshExportConfig,
    MeshProcessor,
    MeshSegmentResult,
    PrimitiveMeshGenerator,
)
from src.shared.python.humanoid_character_builder.mesh.primitive_inertia import (
    PrimitiveInertiaCalculator,
    PrimitiveShape,
)

__all__: list[str] = [
    "CollisionGeometry",
    "CollisionGeometryGenerator",
    "InertiaMode",
    "InertiaResult",
    "LODGenerationResult",
    "LODGenerator",
    "LODLevel",
    "MeshExportConfig",
    "MeshInertiaCalculator",
    "MeshProcessor",
    "MeshSegmentResult",
    "PrimitiveInertiaCalculator",
    "PrimitiveMeshGenerator",
    "PrimitiveShape",
]
