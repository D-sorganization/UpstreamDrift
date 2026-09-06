"""Shared types and abstract interface for mesh generation backends.

Internal submodule extracted from mesh_generator.py to keep file size
within the line budget. Import via ``mesh_generator`` (the public module).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from src.shared.python.humanoid_character_builder.core.body_parameters import (
    BodyParameters,
)


class MeshGeneratorBackend(Enum):
    """Available mesh generation backends."""

    PRIMITIVE = "primitive"  # Generate primitive shapes (built-in)
    MAKEHUMAN = "makehuman"  # MakeHuman integration
    SMPLX = "smplx"  # SMPL-X body model
    CUSTOM = "custom"  # Custom mesh provider


@dataclass
class GeneratedMeshResult:
    """Result of mesh generation."""

    # Whether generation was successful
    success: bool

    # Path to generated mesh files (segment name -> path)
    mesh_paths: dict[str, Path] = field(default_factory=dict)

    # Path to collision mesh files
    collision_paths: dict[str, Path] = field(default_factory=dict)

    # Path to texture files
    texture_paths: dict[str, Path] = field(default_factory=dict)

    # Vertex group mapping (for segmentation)
    vertex_groups: dict[str, list[int]] = field(default_factory=dict)

    # Error message if failed
    error_message: str | None = None

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Status indicator string ("success" / "failure")
    solver_status: str | None = None

    def __post_init__(self) -> None:
        """Initialize solver_status if not explicitly provided."""
        if self.solver_status is None:
            self.solver_status = "success" if self.success else "failure"


def segment_mesh_by_range(
    vertices: Any,
    faces: Any,
    start: int,
    end: int,
) -> tuple[Any, Any]:
    """Extract a submesh by vertex index range [start, end)."""
    import numpy as np

    if len(vertices) == 0 or len(faces) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)

    # A face belongs to the segment if all its vertices are in [start, end)
    mask = (
        (faces[:, 0] >= start)
        & (faces[:, 0] < end)
        & (faces[:, 1] >= start)
        & (faces[:, 1] < end)
        & (faces[:, 2] >= start)
        & (faces[:, 2] < end)
    )
    seg_faces = faces[mask]
    if len(seg_faces) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)

    unique_verts, inverse = np.unique(seg_faces, return_inverse=True)
    new_faces = inverse.reshape(seg_faces.shape)
    new_verts = vertices[unique_verts]
    return new_verts, new_faces


class MeshGeneratorInterface(ABC):
    """
    Abstract interface for mesh generation backends.

    Implement this interface to add new mesh generation sources
    (MakeHuman, SMPL, etc.).
    """

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the backend name."""
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend is available (installed, configured)."""
        ...

    @abstractmethod
    def generate(
        self,
        params: BodyParameters,
        output_dir: Path,
        **kwargs: Any,
    ) -> GeneratedMeshResult:
        """
        Generate meshes for the given body parameters.

        Args:
            params: Body parameters
            output_dir: Directory to write mesh files
            **kwargs: Backend-specific options

        Returns:
            GeneratedMeshResult with paths to generated files
        """
        ...

    @abstractmethod
    def get_supported_segments(self) -> list[str]:
        """Return list of segment names this backend can generate."""
        ...
