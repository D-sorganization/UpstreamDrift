"""
Mesh generation interfaces for humanoid character builder.

This module re-exports all public symbols from the mesh generation sub-modules
for backward compatibility. New code should import directly from the
sub-modules:

  - mesh_types: shared types, interfaces, availability flags
  - mesh_primitive: PrimitiveMeshGenerator
  - mesh_makehuman: MakeHumanMeshGenerator
  - mesh_smplx: SMPLXMeshGenerator
  - mesh_factory: MeshGenerator factory
"""

from __future__ import annotations

# Re-export factory
from humanoid_character_builder.generators.mesh_factory import MeshGenerator
from humanoid_character_builder.generators.mesh_makehuman import MakeHumanMeshGenerator

# Re-export backend implementations
from humanoid_character_builder.generators.mesh_primitive import PrimitiveMeshGenerator
from humanoid_character_builder.generators.mesh_smplx import SMPLXMeshGenerator

# Re-export shared types and availability flags
from humanoid_character_builder.generators.mesh_types import (
    SMPLX_AVAILABLE,
    TRIMESH_AVAILABLE,
    GeneratedMeshResult,
    MeshGeneratorBackend,
    MeshGeneratorInterface,
    _smplx_module,
    _trimesh_module,
)

__all__ = [
    # Types
    "GeneratedMeshResult",
    "MeshGeneratorBackend",
    "MeshGeneratorInterface",
    # Availability flags
    "SMPLX_AVAILABLE",
    "TRIMESH_AVAILABLE",
    "_smplx_module",
    "_trimesh_module",
    # Backends
    "MakeHumanMeshGenerator",
    "MeshGenerator",
    "PrimitiveMeshGenerator",
    "SMPLXMeshGenerator",
]
