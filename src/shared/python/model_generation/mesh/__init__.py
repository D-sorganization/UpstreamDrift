"""
Mesh processing utilities for model generation.

Re-exports mesh processing components from humanoid_character_builder.
"""

from __future__ import annotations

from humanoid_character_builder.mesh import (
    CollisionGeometry,
    CollisionGeometryGenerator,
    InertiaMode,
    InertiaResult,
    LODGenerationResult,
    LODGenerator,
    LODLevel,
    MeshInertiaCalculator,
    MeshProcessor,
    MeshSegmentResult,
    PrimitiveInertiaCalculator,
    PrimitiveShape,
)
from humanoid_character_builder.mesh.mesh_processor import (
    MeshExportConfig,
    PrimitiveMeshGenerator,
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