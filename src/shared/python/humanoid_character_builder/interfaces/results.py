"""Result classes for humanoid character builds."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from humanoid_character_builder.core.body_parameters import BodyParameters
from humanoid_character_builder.generators.mesh_generator import GeneratedMeshResult
from humanoid_character_builder.mesh.inertia_calculator import InertiaResult

from humanoid_character_builder.interfaces.options import ExportOptions

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


class BuildErrorCategory(Enum):
    """Category of build error for proper error handling."""

    NONE = "none"
    VALIDATION = "validation"  # Bad parameters
    IO = "io"  # Filesystem/permission errors
    MISSING_BACKEND = "missing_backend"  # Optional dependency missing
    MESH_GENERATION = "mesh_generation"  # Mesh generation failure
    RUNTIME = "runtime"  # Other runtime errors


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

    # Error category for classification
    error_category: BuildErrorCategory = BuildErrorCategory.NONE

    # Output directory (if exported)
    output_dir: Path | None = None

    # Status of the kinematic solver
    solver_status: str = "success"

    def export_urdf(  # noqa: C901
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
        if output_dir is None:
            raise ValueError("output_dir must be provided")
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

            for src_path in self.mesh_result.mesh_paths.values():
                if src_path and src_path.exists():
                    dst_path = mesh_dir / "visual" / src_path.name
                    shutil.copy2(src_path, dst_path)

            for src_path in self.mesh_result.collision_paths.values():
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
            "solver_status": self.solver_status,
        }

    def simulate(self, duration: float = 1.0) -> bool:
        """
        Run a short simulation to verify physics stability.

        Args:
            duration: Simulation duration in seconds

        Returns:
            True if stable, False otherwise.
        """
        if duration is None:
            raise ValueError("duration must be provided")
        if not self.urdf_xml:
            logger.error("No URDF generated to simulate.")
            return False

        from humanoid_character_builder.interfaces.preview import run_simulation

        return run_simulation(self.urdf_xml, duration)

    def preview(self, animate: bool = False) -> None:
        """
        Open visual preview of the character.

        Args:
            animate: If True, applies control signals to joints.
        """
        if animate is None:
            raise ValueError("animate must be provided")
        if not self.urdf_xml:
            logger.error("No URDF generated to preview.")
            return

        from humanoid_character_builder.interfaces.preview import run_preview

        run_preview(self.urdf_xml, animate)
