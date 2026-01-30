"""
Mesh processing utilities for model generation.

Re-exports mesh processing components from humanoid_character_builder.
"""

try:
    from humanoid_character_builder.mesh.mesh_processor import (
        MeshExportConfig,
        MeshProcessor,
        MeshSegmentResult,
        PrimitiveMeshGenerator,
    )
    from humanoid_character_builder.mesh.inertia_calculator import (
        MeshInertiaCalculator,
    )

    __all__ = [
        "MeshProcessor",
        "MeshExportConfig",
        "MeshSegmentResult",
        "PrimitiveMeshGenerator",
        "MeshInertiaCalculator",
    ]
except ImportError:
    __all__ = []
