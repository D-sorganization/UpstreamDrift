"""Thin facade for humanoid mesh-generator backends."""

from __future__ import annotations

from src.shared.python.logging_pkg.logging_config import get_logger

from .mesh_generator_factory import MeshGenerator
from .mesh_generator_makehuman import MakeHumanMeshGenerator
from .mesh_generator_models import (
    GeneratedMeshResult,
    MeshGeneratorBackend,
    MeshGeneratorInterface,
)
from .mesh_generator_primitive import PrimitiveMeshGenerator
from .mesh_generator_smplx import SMPLXMeshGenerator

logger = get_logger(__name__)

try:
    import smplx as _smplx_module  # type: ignore[import-untyped]

    SMPLX_AVAILABLE = True
except ImportError:
    _smplx_module = None  # type: ignore[assignment]
    SMPLX_AVAILABLE = False

try:
    import trimesh as _trimesh_module  # type: ignore[import-untyped]

    TRIMESH_AVAILABLE = True
except ImportError:
    _trimesh_module = None  # type: ignore[assignment]
    TRIMESH_AVAILABLE = False

__all__ = [
    "logger",
    "SMPLX_AVAILABLE",
    "_smplx_module",
    "TRIMESH_AVAILABLE",
    "_trimesh_module",
    "MeshGeneratorBackend",
    "GeneratedMeshResult",
    "MeshGeneratorInterface",
    "PrimitiveMeshGenerator",
    "MakeHumanMeshGenerator",
    "SMPLXMeshGenerator",
    "MeshGenerator",
]
