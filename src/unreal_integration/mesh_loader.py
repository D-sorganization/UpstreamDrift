"""Multi-format mesh loading system for Unreal Engine integration.

This module provides a unified interface for loading 3D mesh files
from various gaming industry formats (GLTF, GLB, FBX, OBJ, etc.).

Design by Contract:
    - Loaders validate file existence and format
    - Loaded meshes maintain vertex/face count invariants
    - Skeleton data maintains hierarchy consistency

Supported Formats:
    - OBJ (Wavefront) - Static geometry
    - GLTF/GLB - Modern standard with PBR materials
    - FBX - Industry standard for rigged characters
    - COLLADA (.dae) - XML-based interchange format
    - STL - Simple geometry (3D printing)
    - PLY - Point cloud and mesh format

Usage:
    from src.unreal_integration.mesh_loader import MeshLoader

    loader = MeshLoader()
    mesh = loader.load("character.gltf")

    # Access mesh data
    logger.info(f"Vertices: {mesh.vertex_count}")
    logger.info(f"Has skeleton: {mesh.has_skeleton}")
"""

from __future__ import annotations

import logging
from pathlib import Path

from ._mesh_parsers import (
    load_collada,
    load_fbx,
    load_gltf,
    load_obj,
    load_ply,
    load_stl,
)
from ._mesh_types import (
    LoadedMesh,
    MeshBone,
    MeshFace,
    MeshFormat,
    MeshLoadError,
    MeshMaterial,
    MeshSkeleton,
    MeshVertex,
    UnsupportedFormatError,
)

__all__ = [
    "LoadedMesh",
    "MeshBone",
    "MeshFace",
    "MeshFormat",
    "MeshLoadError",
    "MeshLoader",
    "MeshMaterial",
    "MeshSkeleton",
    "MeshVertex",
    "UnsupportedFormatError",
]

logger = logging.getLogger(__name__)


class MeshLoader:
    """Universal mesh loader supporting multiple formats.

    Design by Contract:
        Preconditions:
            - load() requires valid file path
            - load() requires supported format

        Postconditions:
            - load() returns valid LoadedMesh
            - Loaded mesh vertex_count == len(vertices)

    Example:
        >>> loader = MeshLoader()
        >>> mesh = loader.load("character.gltf")
        >>> print(f"Loaded {mesh.vertex_count} vertices")
    """

    def __init__(self, enable_cache: bool = True) -> None:
        """Initialize mesh loader.

        Args:
            enable_cache: Whether to cache loaded meshes.
        """
        if not (enable_cache is not None):
            raise ValueError("enable_cache must be provided")
        if not (enable_cache is not None):
            raise ValueError("enable_cache must be provided")
        self.enable_cache = enable_cache
        self._cache: dict[str, tuple[float, LoadedMesh]] = {}

    @property
    def supported_formats(self) -> list[MeshFormat]:
        """Get list of supported formats."""
        return list(MeshFormat)

    @property
    def cache_size(self) -> int:
        """Get number of cached meshes."""
        return len(self._cache)

    def can_load(self, extension: str) -> bool:
        """Check if format is supported.

        Args:
            extension: File extension (e.g., ".obj").

        Returns:
            True if format is supported.
        """
        try:
            MeshFormat.from_extension(extension)
            return True
        except UnsupportedFormatError:
            return False

    def clear_cache(self) -> None:
        """Clear mesh cache."""
        self._cache.clear()

    def load(self, path: str) -> LoadedMesh:
        """Load mesh from file.

        Preconditions:
            - File must exist
            - Format must be supported

        Args:
            path: Path to mesh file.

        Returns:
            Loaded mesh data.

        Raises:
            FileNotFoundError: If file does not exist.
            UnsupportedFormatError: If format is not supported.
            MeshLoadError: If loading fails.
        """
        path_obj = Path(path)

        if not path_obj.exists():
            raise FileNotFoundError(f"Mesh file not found: {path}")

        fmt = MeshFormat.from_extension(path_obj.suffix)

        if self.enable_cache:
            mtime = path_obj.stat().st_mtime
            if path in self._cache:
                cached_mtime, cached_mesh = self._cache[path]
                if cached_mtime >= mtime:
                    logger.debug(f"Using cached mesh: {path}")
                    return cached_mesh

        try:
            if fmt == MeshFormat.OBJ:
                mesh = load_obj(path_obj)
            elif fmt == MeshFormat.STL:
                mesh = load_stl(path_obj)
            elif fmt in (MeshFormat.GLTF, MeshFormat.GLB):
                mesh = load_gltf(path_obj)
            elif fmt == MeshFormat.FBX:
                mesh = load_fbx(path_obj)
            elif fmt == MeshFormat.COLLADA:
                mesh = load_collada(path_obj)
            elif fmt == MeshFormat.PLY:
                mesh = load_ply(path_obj)
            else:
                raise UnsupportedFormatError(path_obj.suffix, path)

            mesh.source_path = path
            mesh.format = fmt

            if self.enable_cache:
                self._cache[path] = (path_obj.stat().st_mtime, mesh)

            logger.info(
                f"Loaded mesh: {path} ({mesh.vertex_count} vertices, {mesh.face_count} faces)"
            )
            return mesh

        except UnsupportedFormatError:
            raise
        except (RuntimeError, TypeError, ValueError) as e:
            raise MeshLoadError(f"Failed to load mesh: {e}", path, e) from e
