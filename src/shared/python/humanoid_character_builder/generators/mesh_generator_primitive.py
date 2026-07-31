# ARCHITECTURE_DEBT — tracked as GitHub issue #1937
# This file is 1,173 lines and contains 4 distinct mesh generator backends
# (Primitive, MakeHuman, SMPLX) plus the public MeshGenerator facade.
# Recommended split:
#   mesh_generator_primitive.py  — PrimitiveMeshGenerator
#   mesh_generator_makehuman.py  — MakeHumanMeshGenerator
#   mesh_generator_smplx.py      — SMPLXMeshGenerator
#   mesh_generator.py            — MeshGenerator facade + MeshGeneratorBackend enum
# Risk: low-medium — backends are independent; facade is the only public API.
# Prerequisite: parametrize existing tests over all backends before splitting.

"""
Mesh generation interfaces for humanoid character builder.

This module defines interfaces for mesh generation backends
(MakeHuman, SMPL, etc.) and provides a factory for creating
mesh generators.
"""

from __future__ import annotations  # noqa: E402, F404

import logging  # noqa: E402
from abc import ABC, abstractmethod  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from enum import Enum  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

from shared.python.humanoid_character_builder.core.body_parameters import (
    BodyParameters,  # noqa: E402
)

logger = logging.getLogger(__name__)


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


class PrimitiveMeshGenerator(MeshGeneratorInterface):
    """
    Generate primitive shape meshes (built-in, no external dependencies).

    This is the fallback generator that creates simple geometric shapes
    for each body segment.
    """

    @property
    def backend_name(self) -> str:
        return "primitive"

    @property
    def is_available(self) -> bool:
        # Check if trimesh is available for mesh creation
        try:
            import trimesh  # noqa: F401

            return True
        except ImportError:
            return False

    def generate(
        self,
        params: BodyParameters,
        output_dir: Path,
        **kwargs: Any,
    ) -> GeneratedMeshResult:
        """Generate primitive meshes for body segments."""
        if params is None:
            raise ValueError("params must be provided")
        if not self.is_available:
            return GeneratedMeshResult(
                success=False,
                error_message="trimesh not available for primitive mesh generation",
            )

        import trimesh

        from shared.python.humanoid_character_builder.core.anthropometry import (
            estimate_segment_dimensions,
        )
        from shared.python.humanoid_character_builder.core.segment_definitions import (
            HUMANOID_SEGMENTS,
            GeometryType,
        )

        output_dir = Path(output_dir)
        visual_dir = output_dir / "visual"
        collision_dir = output_dir / "collision"
        visual_dir.mkdir(parents=True, exist_ok=True)
        collision_dir.mkdir(parents=True, exist_ok=True)

        mesh_paths = {}
        collision_paths = {}

        gender_factor = params.get_effective_gender_factor()
        dimensions = estimate_segment_dimensions(params.height_m, gender_factor)

        for segment_name, segment_def in HUMANOID_SEGMENTS.items():
            try:
                dims = dimensions.get(
                    segment_name, {"length": 0.1, "width": 0.05, "depth": 0.05}
                )
                length = dims["length"]
                width = dims["width"]
                depth = dims["depth"]

                # Create mesh based on geometry type
                geom_type = segment_def.visual_geometry.geometry_type

                if geom_type == GeometryType.SPHERE:
                    mesh = trimesh.creation.icosphere(radius=length / 2, subdivisions=2)
                elif geom_type == GeometryType.CYLINDER:
                    radius = (width + depth) / 4
                    mesh = trimesh.creation.cylinder(
                        radius=radius, height=length, sections=16
                    )
                elif geom_type == GeometryType.CAPSULE:
                    radius = (width + depth) / 4
                    cyl_height = max(0.01, length - 2 * radius)
                    mesh = trimesh.creation.capsule(
                        radius=radius, height=cyl_height, count=[8, 8]
                    )
                else:  # BOX or default
                    mesh = trimesh.creation.box(extents=(width, depth, length))

                # Export visual mesh
                visual_path = visual_dir / f"{segment_name}.stl"
                mesh.export(str(visual_path))
                mesh_paths[segment_name] = visual_path

                # Create simplified collision mesh (convex hull)
                collision_mesh = mesh.convex_hull
                collision_path = collision_dir / f"{segment_name}.stl"
                collision_mesh.export(str(collision_path))
                collision_paths[segment_name] = collision_path

            except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
                logger.warning(f"Failed to generate mesh for {segment_name}: {e}")

        return GeneratedMeshResult(
            success=len(mesh_paths) > 0,
            mesh_paths=mesh_paths,
            collision_paths=collision_paths,
            metadata={"backend": "primitive"},
        )

    def get_supported_segments(self) -> list[str]:
        from shared.python.humanoid_character_builder.core.segment_definitions import (
            HUMANOID_SEGMENTS,
        )

        return list(HUMANOID_SEGMENTS.keys())
