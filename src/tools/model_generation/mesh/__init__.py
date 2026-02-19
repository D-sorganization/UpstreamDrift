"""
Mesh processing utilities for model generation.

Re-exports mesh processing components from humanoid_character_builder.
"""

try:
    pass

    __all__ = [
        "MeshProcessor",
        "MeshExportConfig",
        "MeshSegmentResult",
        "PrimitiveMeshGenerator",
        "MeshInertiaCalculator",
    ]
except ImportError:
    __all__ = []
