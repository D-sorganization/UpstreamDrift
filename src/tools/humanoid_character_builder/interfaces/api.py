"""
Public API for humanoid character builder.

This module provides the main user-facing interface for creating
humanoid characters and exporting them as URDF models.

Example:
    from humanoid_character_builder import CharacterBuilder, BodyParameters

    # Create builder
    builder = CharacterBuilder()

    # Define character
    params = BodyParameters(
        height_m=1.80,
        mass_kg=80.0,
        muscularity=0.7,
    )

    # Build and export
    result = builder.build(params)
    result.export_urdf("./output/my_humanoid")

    # Or use the quick API
    urdf_xml = builder.generate_urdf(params)
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from humanoid_character_builder.core.anthropometry import (
    estimate_segment_dimensions,
    estimate_segment_masses,
)
from humanoid_character_builder.core.body_parameters import (
    BodyParameters,
)
from humanoid_character_builder.core.segment_definitions import (
    HUMANOID_SEGMENTS,
    get_all_segment_names,
)
from humanoid_character_builder.generators.mesh_generator import (
    GeneratedMeshResult,
    MeshGenerator,
    MeshGeneratorBackend,
)
from humanoid_character_builder.generators.urdf_generator import (
    HumanoidURDFGenerator,
    URDFGeneratorConfig,
)
from humanoid_character_builder.mesh.inertia_calculator import (
    InertiaMode,
    InertiaResult,
    MeshInertiaCalculator,
)
from humanoid_character_builder.mesh.primitive_inertia import (
    PrimitiveInertiaCalculator,
    estimate_segment_primitive,
)

logger = logging.getLogger(__name__)


@dataclass
class SegmentMeshInfo:
    """Information about a generated segment mesh."""

    segment_name: str
    visual_mesh_path: Path | None
    collision_mesh_path: Path | None
    mass_kg: float
    inertia: InertiaResult
    dimensions: dict[str, float]


@dataclass
class ExportOptions:
    """Options for exporting character models."""

    # URDF options
    urdf_filename: str = "humanoid.urdf"
    include_collision: bool = True

    # Mesh options
    generate_meshes: bool = True
    mesh_format: str = "stl"  # stl, obj, dae
    mesh_backend: MeshGeneratorBackend = MeshGeneratorBackend.PRIMITIVE

    # Inertia options
    inertia_mode: InertiaMode = InertiaMode.PRIMITIVE_APPROXIMATION
    density_kg_m3: float = 1050.0

    # Output structure
    create_package_structure: bool = True
    mesh_subdirectory: str = "meshes"
    config_subdirectory: str = "config"

    # Include additional files
    save_config: bool = True
    config_format: str = "yaml"  # yaml or json


@dataclass
class CharacterBuildResult:
    """
    Result of character building operation.

    Contains all generated data and provides export methods.
    """

    # Whether build was successful
    success: bool

    # Body parameters used
    params: BodyParameters

    # Generated URDF string
    urdf_xml: str | None = None

    # Segment information
    segments: dict[str, SegmentMeshInfo] = field(default_factory=dict)

    # Mesh generation result
    mesh_result: GeneratedMeshResult | None = None

    # Error message if failed
    error_message: str | None = None

    # Output directory (if exported)
    output_dir: Path | None = None

    def export_urdf(
        self,
        output_dir: Path | str,
        options: ExportOptions | None = None,
    ) -> Path:
        """
        Export the character as a URDF package.

        Args:
            output_dir: Directory to write output files
            options: Export options

        Returns:
            Path to the generated URDF file
        """
        options = options or ExportOptions()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create package structure
        if options.create_package_structure:
            mesh_dir = output_dir / options.mesh_subdirectory
            mesh_dir.mkdir(exist_ok=True)
            (mesh_dir / "visual").mkdir(exist_ok=True)
            (mesh_dir / "collision").mkdir(exist_ok=True)

            config_dir = output_dir / options.config_subdirectory
            config_dir.mkdir(exist_ok=True)

        # Write URDF
        urdf_path = output_dir / options.urdf_filename
        if self.urdf_xml:
            urdf_path.write_text(self.urdf_xml)
            logger.info(f"URDF written to {urdf_path}")

        # Copy mesh files if they exist
        if self.mesh_result and options.generate_meshes:
            mesh_dir = output_dir / options.mesh_subdirectory

            for _seg_name, src_path in self.mesh_result.mesh_paths.items():
                if src_path and src_path.exists():
                    dst_path = mesh_dir / "visual" / src_path.name
                    shutil.copy2(src_path, dst_path)

            for _seg_name, src_path in self.mesh_result.collision_paths.items():
                if src_path and src_path.exists():
                    dst_path = mesh_dir / "collision" / src_path.name
                    shutil.copy2(src_path, dst_path)

        # Save configuration
        if options.save_config:
            config_dir = output_dir / options.config_subdirectory
            config_path = config_dir / f"body_params.{options.config_format}"

            config_data = self.params.to_dict()
            if options.config_format == "yaml":
                config_path.write_text(yaml.dump(config_data, default_flow_style=False))
            else:
                config_path.write_text(json.dumps(config_data, indent=2))

        self.output_dir = output_dir
        return urdf_path

    def get_segment(self, segment_name: str) -> SegmentMeshInfo | None:
        """Get information about a specific segment."""
        return self.segments.get(segment_name)

    def get_all_segments(self) -> list[str]:
        """Get list of all segment names."""
        return list(self.segments.keys())

    def get_total_mass(self) -> float:
        """Get total mass of all segments."""
        return sum(seg.mass_kg for seg in self.segments.values())

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "params": self.params.to_dict(),
            "segment_count": len(self.segments),
            "total_mass": self.get_total_mass(),
            "error_message": self.error_message,
        }


class CharacterBuilder:
    """
    Main interface for building humanoid characters.

    This class provides a high-level API for:
    - Creating characters from body parameters
    - Generating URDF models
    - Exporting meshes
    - Computing inertias

    Example:
        builder = CharacterBuilder()

        # Quick generation
        urdf = builder.generate_urdf(BodyParameters(height_m=1.80))

        # Full build with meshes
        result = builder.build(BodyParameters(height_m=1.80))
        result.export_urdf("./output")

        # Compute single segment inertia
        inertia = builder.compute_segment_inertia("thigh", mass=10.0)
    """

    def __init__(
        self,
        urdf_config: URDFGeneratorConfig | None = None,
        mesh_backend: MeshGeneratorBackend = MeshGeneratorBackend.PRIMITIVE,
    ):
        """
        Initialize the character builder.

        Args:
            urdf_config: Configuration for URDF generation
            mesh_backend: Backend to use for mesh generation
        """
        self.urdf_config = urdf_config or URDFGeneratorConfig()
        self.mesh_backend = mesh_backend

        self._urdf_generator = HumanoidURDFGenerator(self.urdf_config)
        self._mesh_inertia_calc = MeshInertiaCalculator()
        self._primitive_inertia_calc = PrimitiveInertiaCalculator()

    def build(
        self,
        params: BodyParameters,
        generate_meshes: bool = True,
        mesh_output_dir: Path | str | None = None,
    ) -> CharacterBuildResult:
        """
        Build a complete character from body parameters.

        Args:
            params: Body parameters defining the character
            generate_meshes: Whether to generate mesh files
            mesh_output_dir: Directory for mesh files (temp dir if not specified)

        Returns:
            CharacterBuildResult with generated data
        """
        try:
            # Validate parameters
            errors = params.validate()
            if errors:
                logger.warning(f"Parameter validation warnings: {errors}")

            # Generate meshes if requested
            mesh_result = None
            if generate_meshes:
                if mesh_output_dir is None:
                    import tempfile

                    mesh_output_dir = Path(tempfile.mkdtemp(prefix="humanoid_meshes_"))
                else:
                    mesh_output_dir = Path(mesh_output_dir)

                mesh_generator = MeshGenerator.create(self.mesh_backend)
                mesh_result = mesh_generator.generate(params, mesh_output_dir)

            # Generate URDF
            urdf_xml = self._urdf_generator.generate(
                params,
                mesh_dir=mesh_output_dir if mesh_result else None,
            )

            # Build segment info
            segments = self._build_segment_info(params, mesh_result)

            return CharacterBuildResult(
                success=True,
                params=params,
                urdf_xml=urdf_xml,
                segments=segments,
                mesh_result=mesh_result,
            )

        except ImportError as e:
            logger.error(f"Character build failed: {e}")
            return CharacterBuildResult(
                success=False,
                params=params,
                error_message=str(e),
            )

    def generate_urdf(
        self,
        params: BodyParameters,
        output_path: Path | str | None = None,
    ) -> str:
        """
        Generate URDF XML from body parameters.

        This is a quick method that generates URDF without mesh files.

        Args:
            params: Body parameters
            output_path: Optional path to write URDF file

        Returns:
            URDF XML string
        """
        return self._urdf_generator.generate(params, output_path)

    def compute_segment_inertia(
        self,
        segment_name: str,
        mass: float | None = None,
        dimensions: dict[str, float] | None = None,
        mesh_path: Path | str | None = None,
        mode: InertiaMode = InertiaMode.PRIMITIVE_APPROXIMATION,
        density: float = 1050.0,
    ) -> InertiaResult:
        """
        Compute inertia for a single segment.

        Args:
            segment_name: Name of the segment
            mass: Mass in kg (estimated from default if not provided)
            dimensions: Segment dimensions (estimated if not provided)
            mesh_path: Path to mesh file (for mesh-based calculation)
            mode: Inertia calculation mode
            density: Density in kg/m^3 (for uniform density mode)

        Returns:
            InertiaResult with computed inertia
        """
        # Get default dimensions if not provided
        if dimensions is None:
            all_dims = estimate_segment_dimensions(1.75, 0.5)  # Default height, neutral
            dimensions = all_dims.get(
                segment_name, {"length": 0.1, "width": 0.05, "depth": 0.05}
            )

        # Get default mass if not provided
        if mass is None:
            all_masses = estimate_segment_masses(75.0, 0.5)  # Default mass, neutral
            mass = all_masses.get(segment_name, 1.0)

        # Compute based on mode
        if mode in (InertiaMode.MESH_UNIFORM_DENSITY, InertiaMode.MESH_SPECIFIED_MASS):
            if mesh_path:
                if mode == InertiaMode.MESH_SPECIFIED_MASS:
                    return self._mesh_inertia_calc.compute_from_mesh(
                        mesh_path, mass=mass
                    )
                else:
                    return self._mesh_inertia_calc.compute_from_mesh(
                        mesh_path, density=density
                    )
            else:
                logger.warning(
                    f"No mesh path provided for {segment_name}, falling back to primitive"
                )

        # Primitive approximation
        length = dimensions.get("length", 0.1)
        width = dimensions.get("width", 0.05)
        depth = dimensions.get("depth", 0.05)

        shape, shape_dims = estimate_segment_primitive(
            segment_name, length, width, depth
        )
        return self._primitive_inertia_calc.compute(shape, mass, shape_dims)

    def compute_all_inertias(
        self,
        params: BodyParameters,
        mode: InertiaMode = InertiaMode.PRIMITIVE_APPROXIMATION,
        mesh_dir: Path | str | None = None,
    ) -> dict[str, InertiaResult]:
        """
        Compute inertias for all segments.

        Args:
            params: Body parameters
            mode: Inertia calculation mode
            mesh_dir: Directory containing mesh files (for mesh-based)

        Returns:
            Dict mapping segment name to InertiaResult
        """
        gender_factor = params.get_effective_gender_factor()
        masses = estimate_segment_masses(params.mass_kg, gender_factor)
        dimensions = estimate_segment_dimensions(params.height_m, gender_factor)

        inertias = {}
        for segment_name in get_all_segment_names():
            mesh_path = None
            if mesh_dir:
                mesh_path = Path(mesh_dir) / f"{segment_name}.stl"
                if not mesh_path.exists():
                    mesh_path = None

            inertias[segment_name] = self.compute_segment_inertia(
                segment_name,
                mass=masses.get(segment_name),
                dimensions=dimensions.get(segment_name),
                mesh_path=mesh_path,
                mode=mode,
            )

        return inertias

    def _build_segment_info(
        self,
        params: BodyParameters,
        mesh_result: GeneratedMeshResult | None,
    ) -> dict[str, SegmentMeshInfo]:
        """Build segment information dictionary."""
        gender_factor = params.get_effective_gender_factor()
        masses = estimate_segment_masses(params.mass_kg, gender_factor)
        dimensions = estimate_segment_dimensions(params.height_m, gender_factor)

        segments = {}
        for segment_name in HUMANOID_SEGMENTS:
            mass = masses.get(segment_name, 1.0)
            dims = dimensions.get(
                segment_name, {"length": 0.1, "width": 0.05, "depth": 0.05}
            )

            # Get mesh paths if available
            visual_path = None
            collision_path = None
            if mesh_result:
                visual_path = mesh_result.mesh_paths.get(segment_name)
                collision_path = mesh_result.collision_paths.get(segment_name)

            # Compute inertia
            inertia = self.compute_segment_inertia(
                segment_name,
                mass=mass,
                dimensions=dims,
                mesh_path=visual_path,
                mode=self.urdf_config.inertia_mode,
            )

            segments[segment_name] = SegmentMeshInfo(
                segment_name=segment_name,
                visual_mesh_path=visual_path,
                collision_mesh_path=collision_path,
                mass_kg=mass,
                inertia=inertia,
                dimensions=dims,
            )

        return segments

    @staticmethod
    def create_from_preset(
        preset_name: str,
        height_m: float | None = None,
        mass_kg: float | None = None,
    ) -> BodyParameters:
        """
        Create body parameters from a preset.

        Args:
            preset_name: Name of preset (athletic, average, heavy, etc.)
            height_m: Override height
            mass_kg: Override mass

        Returns:
            BodyParameters configured for the preset
        """
        from humanoid_character_builder.presets.loader import load_body_preset

        return load_body_preset(preset_name, height_m=height_m, mass_kg=mass_kg)

    @staticmethod
    def list_presets() -> list[str]:
        """List available body presets."""
        from humanoid_character_builder.presets.loader import list_available_presets

        return list_available_presets()

    @staticmethod
    def list_segments() -> list[str]:
        """List all available segment names."""
        return get_all_segment_names()

    @staticmethod
    def get_segment_definition(segment_name: str) -> dict[str, Any] | None:
        """Get definition for a segment."""
        segment = HUMANOID_SEGMENTS.get(segment_name)
        if segment is None:
            return None

        return {
            "name": segment.name,
            "parent": segment.parent,
            "mass_ratio": segment.mass_ratio,
            "length_ratio": segment.length_ratio,
            "is_end_effector": segment.is_end_effector,
            "vertex_group": segment.vertex_group,
        }


# Convenience functions for quick access
def quick_build(
    height_m: float = 1.75,
    mass_kg: float = 75.0,
    preset: str | None = None,
    output_dir: Path | str | None = None,
) -> CharacterBuildResult:
    """
    Quick function to build a character with minimal configuration.

    Args:
        height_m: Character height in meters
        mass_kg: Character mass in kg
        preset: Optional preset name (overrides height/mass defaults)
        output_dir: Optional output directory for export

    Returns:
        CharacterBuildResult
    """
    builder = CharacterBuilder()

    if preset:
        params = builder.create_from_preset(preset, height_m=height_m, mass_kg=mass_kg)
    else:
        params = BodyParameters(height_m=height_m, mass_kg=mass_kg)

    result = builder.build(params)

    if output_dir and result.success:
        result.export_urdf(output_dir)

    return result


def quick_urdf(
    height_m: float = 1.75,
    mass_kg: float = 75.0,
    preset: str | None = None,
) -> str:
    """
    Quick function to generate URDF XML.

    Args:
        height_m: Character height in meters
        mass_kg: Character mass in kg
        preset: Optional preset name

    Returns:
        URDF XML string
    """
    builder = CharacterBuilder()

    if preset:
        params = builder.create_from_preset(preset, height_m=height_m, mass_kg=mass_kg)
    else:
        params = BodyParameters(height_m=height_m, mass_kg=mass_kg)

    return builder.generate_urdf(params)
