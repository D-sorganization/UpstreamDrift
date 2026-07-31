# mypy: ignore-errors
"""Mesh processing utilities for model generation.

Re-exports mesh processing components from humanoid_character_builder.mesh.
"""

try:
    from shared.python.humanoid_character_builder.mesh import (
        CollisionGeometry,
        CollisionGeometryGenerator,
        MeshExportConfig,
        MeshProcessor,
        MeshSegmentResult,
        PrimitiveMeshGenerator,
    )
    from shared.python.humanoid_character_builder.mesh.lod import (
        LODGenerationResult,
        LODGenerator,
        LODLevel,
    )
    from shared.python.humanoid_character_builder.mesh.mesh_inertia import (
        InertiaMode,
        InertiaResult,
        MeshInertiaCalculator,
        PrimitiveInertiaCalculator,
        PrimitiveShape,
    )
except ImportError:  # pragma: no cover
    pass

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
