# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

"""
Standalone URDF generator for humanoid characters.

This module generates complete URDF files from body parameters,
segment definitions, and computed inertias. It is fully self-contained
and does not depend on other Golf Modeling Suite modules.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from humanoid_character_builder.contracts import postcondition, precondition
from humanoid_character_builder.core.anthropometry import (
    estimate_segment_dimensions,
    estimate_segment_masses,
)
from humanoid_character_builder.core.body_parameters import BodyParameters
from humanoid_character_builder.core.model import (
    GeneratedJoint,
    GeneratedLink,
    HumanoidModel,
)
from humanoid_character_builder.core.segment_definitions import (
    HUMANOID_JOINTS,
    HUMANOID_SEGMENTS,
    JointDefinition,
    JointType,
    SegmentDefinition,
)
from humanoid_character_builder.generators._urdf_model_builder import (
    URDFModelBuilder,
    map_joint_type,
)
from humanoid_character_builder.generators._urdf_xml_writer import URDFXMLWriter
from humanoid_character_builder.mesh.inertia_calculator import (
    InertiaMode,
    InertiaResult,
    MeshInertiaCalculator,
)
from humanoid_character_builder.mesh.primitive_inertia import (
    PrimitiveInertiaCalculator,
)

logger = logging.getLogger(__name__)


@dataclass
class URDFGeneratorConfig:
    """Configuration for URDF generation."""

    # Inertia calculation mode
    inertia_mode: InertiaMode = InertiaMode.PRIMITIVE_APPROXIMATION

    # Density for uniform density calculation (kg/m^3)
    default_density: float = 1050.0

    # Mesh paths (relative to URDF or package://)
    mesh_package_name: str | None = None  # e.g., "humanoid_model"
    visual_mesh_dir: str = "meshes/visual"
    collision_mesh_dir: str = "meshes/collision"

    # Use mesh for visual geometry (vs primitives)
    use_mesh_visual: bool = False

    # Use mesh for collision geometry (vs primitives)
    use_mesh_collision: bool = False

    # Generate collision geometry
    generate_collision: bool = True

    # Joint configuration
    default_joint_damping: float = 0.5
    default_joint_friction: float = 0.0

    # URDF formatting
    pretty_print: bool = True
    indent: str = "  "

    # Expand composite joints (gimbal/universal) to multiple revolute joints
    expand_composite_joints: bool = True

    # Include comments in URDF
    include_comments: bool = True


class HumanoidURDFGenerator:
    """
    Generate URDF files for humanoid characters.

    This is a standalone generator that creates complete URDF files
    from body parameters. It handles:
    - Scaling segments based on height/mass
    - Computing inertias (mesh-based, primitive, or manual)
    - Generating links and joints
    - Expanding composite joints (gimbal, universal)
    - Outputting valid URDF XML
    """

    def __init__(self, config: URDFGeneratorConfig | None = None):
        """
        Initialize the generator.

        Args:
            config: Generator configuration
        """
        self.config = config or URDFGeneratorConfig()
        self.mesh_inertia_calc = MeshInertiaCalculator(self.config.default_density)
        self.primitive_inertia_calc = PrimitiveInertiaCalculator()

        # Generated data
        self._links: dict[str, GeneratedLink] = {}
        self._joints: list[GeneratedJoint] = []
        self._materials: dict[str, tuple[float, float, float, float]] = {}
        self._model_builder = URDFModelBuilder(
            config=self.config,
            mesh_inertia_calc=self.mesh_inertia_calc,
            primitive_inertia_calc=self.primitive_inertia_calc,
            links=self._links,
            joints=self._joints,
            materials=self._materials,
        )
        self._xml_writer = URDFXMLWriter(self.config)

    @precondition(
        lambda params: params is not None,
        "params must not be None",
    )
    @precondition(
        lambda params: params.height_m > 0,
        "Height must be positive",
    )
    @precondition(
        lambda params: params.mass_kg > 0,
        "Mass must be positive",
    )
    @postcondition(
        lambda result: len(result.links) > 0,
        "Model must have at least one link",
    )
    def build_model(
        self,
        params: BodyParameters,
        mesh_dir: Path | str | None = None,
    ) -> HumanoidModel:
        """
        Build HumanoidModel from body parameters.

        Args:
            params: Body parameters
            mesh_dir: Optional directory containing mesh files

        Returns:
            HumanoidModel instance
        """
        # Validate parameters
        if not (params is not None):
            raise ValueError("params must be provided")
        errors = params.validate()
        if errors:
            logger.warning(f"Parameter validation warnings: {errors}")

        # Clear previous generation
        self._links.clear()
        self._joints.clear()
        self._materials.clear()

        # Compute scaled dimensions and masses
        gender_factor = params.get_effective_gender_factor()
        segment_masses = estimate_segment_masses(params.mass_kg, gender_factor)
        segment_dimensions = estimate_segment_dimensions(params.height_m, gender_factor)

        # Apply proportion factors
        segment_dimensions = self._apply_proportion_factors(segment_dimensions, params)

        # Generate materials
        self._generate_materials(params)

        # Generate links
        for segment_name, segment_def in HUMANOID_SEGMENTS.items():
            self._generate_link(
                segment_name,
                segment_def,
                params,
                segment_masses.get(segment_name, 1.0),
                segment_dimensions.get(
                    segment_name, {"length": 0.1, "width": 0.05, "depth": 0.05}
                ),
                gender_factor,
                mesh_dir,
            )

        # Generate joints
        for joint_name, joint_def in HUMANOID_JOINTS.items():
            self._generate_joint(joint_name, joint_def, segment_dimensions)

        return HumanoidModel(self._links, self._joints)

    @precondition(
        lambda params: params is not None,
        "params must not be None",
    )
    @precondition(
        lambda params: params.height_m > 0,
        "Height must be positive",
    )
    @precondition(
        lambda params: params.mass_kg > 0,
        "Mass must be positive",
    )
    @postcondition(
        lambda result: len(result) > 0,
        "URDF output must not be empty",
    )
    def generate(
        self,
        params: BodyParameters,
        output_path: Path | str | None = None,
        mesh_dir: Path | str | None = None,
    ) -> str:
        """
        Generate URDF from body parameters.

        Args:
            params: Body parameters
            output_path: Optional path to write URDF file
            mesh_dir: Optional directory containing mesh files

        Returns:
            URDF XML string
        """
        if not (params is not None):
            raise ValueError("params must be provided")
        self.build_model(params, mesh_dir)

        # Build URDF XML
        urdf_xml = self._build_urdf_xml(params.name)

        # Write to file if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(urdf_xml)
            logger.info(f"URDF written to {output_path}")

        return urdf_xml

    def _apply_proportion_factors(
        self,
        dimensions: dict[str, dict[str, float]],
        params: BodyParameters,
    ) -> dict[str, dict[str, float]]:
        """Apply proportion factors to segment dimensions."""
        return self._model_builder.apply_proportion_factors(dimensions, params)

    def _generate_materials(self, params: BodyParameters) -> None:
        """Generate material definitions."""
        self._model_builder.generate_materials(params)

    def _generate_link(
        self,
        segment_name: str,
        segment_def: SegmentDefinition,
        params: BodyParameters,
        mass: float,
        dimensions: dict[str, float],
        gender_factor: float,
        mesh_dir: Path | str | None,
    ) -> None:
        """Generate a single URDF link."""
        self._model_builder.generate_link(
            segment_name,
            segment_def,
            params,
            mass,
            dimensions,
            gender_factor,
            mesh_dir,
        )

    def _compute_segment_inertia(
        self,
        segment_name: str,
        segment_def: SegmentDefinition,
        seg_params: Any,
        mass: float,
        dimensions: dict[str, float],
        gender_factor: float,
        mesh_dir: Path | str | None,
    ) -> InertiaResult:
        """Compute inertia for a segment."""
        return self._model_builder.compute_segment_inertia(
            segment_name,
            segment_def,
            seg_params,
            mass,
            dimensions,
            gender_factor,
            mesh_dir,
        )

    def _create_geometry_dict(
        self,
        segment_def: SegmentDefinition,
        dimensions: dict[str, float],
        is_collision: bool,
    ) -> dict[str, Any]:
        """Create geometry specification dictionary."""
        return self._model_builder.create_geometry_dict(
            segment_def,
            dimensions,
            is_collision,
        )

    def _generate_joint(
        self,
        joint_name: str,
        joint_def: JointDefinition,
        dimensions: dict[str, dict[str, float]],
    ) -> None:
        """Generate URDF joint(s) from joint definition."""
        self._model_builder.generate_joint(joint_name, joint_def, dimensions)

    def _generate_single_joint(
        self,
        joint_name: str,
        joint_def: JointDefinition,
    ) -> None:
        """Generate a single URDF joint."""
        self._model_builder.generate_single_joint(joint_name, joint_def)

    def _expand_composite_joint(
        self,
        joint_name: str,
        joint_def: JointDefinition,
        dimensions: dict[str, dict[str, float]],
    ) -> None:
        """Expand composite joint into multiple revolute joints."""
        del dimensions
        self._model_builder.expand_composite_joint(joint_name, joint_def)

    def _map_joint_type(self, joint_type: JointType) -> str:
        """Map internal joint type to URDF joint type string."""
        return map_joint_type(joint_type)

    def _build_urdf_xml(self, robot_name: str) -> str:
        """Build the complete URDF XML."""
        return self._xml_writer.build_urdf_xml(
            robot_name,
            self._materials,
            self._links,
            self._joints,
        )

    def _add_link_element(self, root: ET.Element, link: GeneratedLink) -> None:
        """Add a link element to the URDF."""
        self._xml_writer.add_link_element(root, link)

    def _add_geometry_element(self, parent: ET.Element, geom: dict[str, Any]) -> None:
        """Add geometry element."""
        self._xml_writer.add_geometry_element(parent, geom)

    def _add_joint_element(self, root: ET.Element, joint: GeneratedJoint) -> None:
        """Add a joint element to the URDF."""
        self._xml_writer.add_joint_element(root, joint)


def generate_humanoid_urdf(
    params: BodyParameters,
    output_path: Path | str | None = None,
    config: URDFGeneratorConfig | None = None,
) -> str:
    """
    Convenience function to generate humanoid URDF.

    Args:
        params: Body parameters
        output_path: Optional path to write URDF
        config: Generator configuration

    Returns:
        URDF XML string
    """
    if not (params is not None):
        raise ValueError("params must be provided")
    generator = HumanoidURDFGenerator(config)
    return cast(str, generator.generate(params, output_path))
