"""
Shared types and interfaces for mesh generation backends.

Defines the abstract interface, result dataclass, and backend enum used
across all mesh generation sub-modules.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from humanoid_character_builder.core.body_parameters import BodyParameters, GenderModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency availability flags (mock-patchable in tests)
# ---------------------------------------------------------------------------

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


__all__ = [
    "GenderModel",
    "GeneratedMeshResult",
    "MeshGeneratorBackend",
    "MeshGeneratorInterface",
    "SMPLX_AVAILABLE",
    "TRIMESH_AVAILABLE",
    "_smplx_module",
    "_trimesh_module",
    "np",
]
