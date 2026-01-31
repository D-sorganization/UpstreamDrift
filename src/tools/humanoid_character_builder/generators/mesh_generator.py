"""
Mesh generation interfaces for humanoid character builder.

This module defines interfaces for mesh generation backends
(MakeHuman, SMPL, etc.) and provides a factory for creating
mesh generators.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from humanoid_character_builder.core.body_parameters import BodyParameters

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
        if not self.is_available:
            return GeneratedMeshResult(
                success=False,
                error_message="trimesh not available for primitive mesh generation",
            )

        import trimesh
        from humanoid_character_builder.core.anthropometry import (
            estimate_segment_dimensions,
        )
        from humanoid_character_builder.core.segment_definitions import (
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

            except Exception as e:
                logger.warning(f"Failed to generate mesh for {segment_name}: {e}")

        return GeneratedMeshResult(
            success=len(mesh_paths) > 0,
            mesh_paths=mesh_paths,
            collision_paths=collision_paths,
            metadata={"backend": "primitive"},
        )

    def get_supported_segments(self) -> list[str]:
        from humanoid_character_builder.core.segment_definitions import (
            HUMANOID_SEGMENTS,
        )

        return list(HUMANOID_SEGMENTS.keys())


class MakeHumanMeshGenerator(MeshGeneratorInterface):
    """
    Generate meshes using MakeHuman.

    This is a placeholder for future MakeHuman integration.
    MakeHuman provides high-quality, customizable human meshes
    with proper vertex groups for segmentation.
    """

    def __init__(self, makehuman_path: Path | str | None = None):
        """
        Initialize MakeHuman generator.

        Args:
            makehuman_path: Path to MakeHuman installation
        """
        self.makehuman_path = Path(makehuman_path) if makehuman_path else None

    @property
    def backend_name(self) -> str:
        return "makehuman"

    @property
    def is_available(self) -> bool:
        # Check if MakeHuman is installed
        if self.makehuman_path and self.makehuman_path.exists():
            return True

        # Try to find MakeHuman in common locations
        common_paths = [
            Path("/usr/share/makehuman"),
            Path.home() / "makehuman",
            Path.home() / ".makehuman",
        ]
        for path in common_paths:
            if path.exists():
                self.makehuman_path = path
                return True

        return False

    def generate(
        self,
        params: BodyParameters,
        output_dir: Path,
        **kwargs: Any,
    ) -> GeneratedMeshResult:
        """Generate meshes using MakeHuman."""
        if not self.is_available:
            return GeneratedMeshResult(
                success=False,
                error_message="MakeHuman not found. Please install MakeHuman or provide path.",
            )

        # TODO: Implement MakeHuman integration
        # This would involve:
        # 1. Convert BodyParameters to MakeHuman modifier values
        # 2. Run MakeHuman in scripted mode to generate mesh
        # 3. Export mesh and segment by vertex groups
        # 4. Return paths to generated files

        logger.warning(
            "MakeHuman integration not yet implemented. "
            "Falling back to primitive meshes."
        )

        # Fall back to primitive generator
        primitive_gen = PrimitiveMeshGenerator()
        return primitive_gen.generate(params, output_dir, **kwargs)

    def get_supported_segments(self) -> list[str]:
        # MakeHuman supports all standard humanoid segments
        from humanoid_character_builder.core.segment_definitions import (
            HUMANOID_SEGMENTS,
        )

        return list(HUMANOID_SEGMENTS.keys())

    def _convert_params_to_makehuman(self, params: BodyParameters) -> dict[str, float]:
        """Convert BodyParameters to MakeHuman modifier values."""
        # MakeHuman uses modifiers in range [-1, 1] or [0, 1]
        modifiers = {}

        # Height is handled by overall scale
        # MakeHuman default is ~1.68m, adjust proportionally
        _height_scale = params.height_m / 1.68  # Reserved for future MakeHuman API use

        # Gender (MakeHuman: 0 = female, 1 = male)
        modifiers["macrodetails/Gender"] = params.get_effective_gender_factor()

        # Age (MakeHuman: range depends on modifier)
        modifiers["macrodetails/Age"] = min(
            1.0, max(0.0, params.appearance.age_years / 80.0)
        )

        # Muscularity (MakeHuman: muscle definition)
        modifiers["macrodetails-universal/Muscle"] = params.muscularity

        # Weight/body fat
        modifiers["macrodetails-universal/Weight"] = params.body_fat_factor

        # Proportions
        modifiers["macrodetails-proportions/BodyProportions"] = (
            params.torso_length_factor - 1.0
        )

        return modifiers


class SMPLXMeshGenerator(MeshGeneratorInterface):
    """
    Generate meshes using SMPL-X body model.

    This is a placeholder for future SMPL-X integration.
    SMPL-X provides a differentiable body model learned from
    thousands of 3D body scans.
    """

    @property
    def backend_name(self) -> str:
        return "smplx"

    @property
    def is_available(self) -> bool:
        try:
            import smplx  # noqa: F401

            return True
        except ImportError:
            return False

    def generate(
        self,
        params: BodyParameters,
        output_dir: Path,
        **kwargs: Any,
    ) -> GeneratedMeshResult:
        """Generate meshes using SMPL-X."""
        if not self.is_available:
            return GeneratedMeshResult(
                success=False,
                error_message="smplx package not installed. Install with: pip install smplx",
            )

        # TODO: Implement SMPL-X integration
        logger.warning(
            "SMPL-X integration not yet implemented. "
            "Falling back to primitive meshes."
        )

        primitive_gen = PrimitiveMeshGenerator()
        return primitive_gen.generate(params, output_dir, **kwargs)

    def get_supported_segments(self) -> list[str]:
        # SMPL-X provides full body mesh, needs segmentation
        from humanoid_character_builder.core.segment_definitions import (
            HUMANOID_SEGMENTS,
        )

        return list(HUMANOID_SEGMENTS.keys())


class MeshGenerator:
    """
    Factory class for creating mesh generators.

    Provides a unified interface to multiple mesh generation backends.
    """

    _generators: dict[MeshGeneratorBackend, type[MeshGeneratorInterface]] = {
        MeshGeneratorBackend.PRIMITIVE: PrimitiveMeshGenerator,
        MeshGeneratorBackend.MAKEHUMAN: MakeHumanMeshGenerator,
        MeshGeneratorBackend.SMPLX: SMPLXMeshGenerator,
    }

    @classmethod
    def create(
        cls,
        backend: MeshGeneratorBackend | str = MeshGeneratorBackend.PRIMITIVE,
        **kwargs: Any,
    ) -> MeshGeneratorInterface:
        """
        Create a mesh generator for the specified backend.

        Args:
            backend: Backend to use
            **kwargs: Backend-specific initialization options

        Returns:
            MeshGeneratorInterface instance
        """
        if isinstance(backend, str):
            backend = MeshGeneratorBackend(backend.lower())

        generator_class = cls._generators.get(backend)
        if generator_class is None:
            raise ValueError(f"Unknown backend: {backend}")

        return generator_class(**kwargs)

    @classmethod
    def get_available_backends(cls) -> list[MeshGeneratorBackend]:
        """Return list of available backends."""
        available = []
        for backend, generator_class in cls._generators.items():
            try:
                generator = generator_class()
                if generator.is_available:
                    available.append(backend)
            except Exception:
                pass
        return available

    @classmethod
    def get_best_available(cls) -> MeshGeneratorInterface:
        """
        Get the best available mesh generator.

        Preference order: MakeHuman > SMPL-X > Primitive
        """
        preference = [
            MeshGeneratorBackend.MAKEHUMAN,
            MeshGeneratorBackend.SMPLX,
            MeshGeneratorBackend.PRIMITIVE,
        ]

        for backend in preference:
            try:
                generator = cls.create(backend)
                if generator.is_available:
                    return generator
            except Exception:
                continue

        # Final fallback
        return PrimitiveMeshGenerator()
