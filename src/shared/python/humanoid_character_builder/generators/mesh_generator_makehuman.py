"""Compatibility exports for the MakeHuman mesh generator.

The implementation lives in ``_makehuman_generator``. This module remains for
older imports that referenced the pre-split file directly.
"""

from __future__ import annotations

from ._makehuman_generator import MakeHumanMeshGenerator
from ._mesh_types import (
    GeneratedMeshResult,
    MeshGeneratorBackend,
    MeshGeneratorInterface,
)

# Backward compatibility for the old pre-split module, where the MakeHuman
# implementation was accidentally exposed under this name.
PrimitiveMeshGenerator = MakeHumanMeshGenerator

__all__ = [
    "GeneratedMeshResult",
    "MakeHumanMeshGenerator",
    "MeshGeneratorBackend",
    "MeshGeneratorInterface",
    "PrimitiveMeshGenerator",
]
