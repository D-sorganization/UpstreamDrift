"""
Standalone URDF generator for humanoid characters.

This module generates complete URDF files from body parameters,
segment definitions, and computed inertias. It is fully self-contained
and does not depend on other Golf Modeling Suite modules.

Internal implementation is decomposed into focused sub-modules:

- :mod:`urdf_config`       -- URDFGeneratorConfig dataclass
- :mod:`urdf_geometry`     -- geometry dict creation and XML rendering
- :mod:`urdf_joints`       -- joint-type mapping and composite expansion
- :mod:`urdf_xml_builder`  -- full XML tree assembly

Public API (HumanoidURDFGenerator, URDFGeneratorConfig,
generate_humanoid_urdf) is fully preserved.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET  # stdlib retained for Element/SubElement
from pathlib import Path
from typing import Any, cast

import defusedxml.ElementTree as DefusedET  # noqa: S314  # Security: defusedxml prevents XML attacks
from humanoid_character_builder.contracts import postcondition, precondition
from humanoid_character_builder.core.anthropometry import (
    estimate_segment_dimensions,
    estimate_segment_masses,
    get_com_location,
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
    SegmentDefinition,
)
from humanoid_character_builder.generators._link_generation import (
    apply_proportion_factors,
    generate_link,
    generate_materials,
)
from humanoid_character_builder.generators.urdf_config import URDFGeneratorConfig
from humanoid_character_builder.generators.urdf_geometry import (
    add_geometry_element,
    create_geometry_dict,
)
from humanoid_character_builder.generators.urdf_joints import (
    expand_composite_joint,
    generate_joint,
    generate_single_joint,
    map_joint_type,
)
from humanoid_character_builder.generators.urdf_xml_builder import (
    add_joint_element as _add_joint_element,
)
from humanoid_character_builder.generators.urdf_xml_builder import (
    add_link_element as _add_link_element,
)
from humanoid_character_builder.generators.urdf_xml_builder import (
    build_urdf_xml,
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
from src.shared.python.body_part_viz.contracts import BodyPartShape

logger = logging.getLogger(__name__)

# Re-export URDFGeneratorConfig so existing callers that import it from this
# module continue to work without change.
__all__ = [
    "HumanoidURDFGenerator",
    "URDFGeneratorConfig",
    "generate_humanoid_urdf",
]


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

    def __init__(self, config: URDFGeneratorConfig | None = None) -> None:
        """
        Initialize the generator.

        Args:
            config: Generator configuration
        """
        self.config = config or URDFGeneratorConfig()
        self.mesh_inertia_calc = MeshInertiaCalculator(self.config.default_density)
        self.primitive_inertia_calc = PrimitiveInertiaCalculator()
        self._links: dict[str, GeneratedLink] = {}
        self._joints: list[GeneratedJoint] = []
        self._materials: dict[str, tuple[float, float, float, float]] = {}

    @precondition(lambda params: params is not None, "params must not be None")
    @precondition(lambda params: params.height_m > 0, "Height must be positive")
    @precondition(lambda params: params.mass_kg > 0, "Mass must be positive")
    @postcondition(
        lambda result: len(result.links) > 0, "Model must have at least one link"
    )
    def build_model(
        self, params: BodyParameters, mesh_dir: Path | str | None = None
    ) -> HumanoidModel:
        """
        Build HumanoidModel from body parameters.

        Args:
            params: Body parameters
            mesh_dir: Optional directory containing mesh files

        Returns:
            HumanoidModel instance
        """
        if not (params is not None):
            raise ValueError("params must be provided")
        errors = params.validate()
        if errors:
            logger.warning(f"Parameter validation warnings: {errors}")

        self._links.clear()
        self._joints.clear()
        self._materials.clear()

        gender_factor = params.get_effective_gender_factor()
        segment_masses = estimate_segment_masses(params.mass_kg, gender_factor)
        segment_dimensions = estimate_segment_dimensions(params.height_m, gender_factor)

        segment_dimensions = apply_proportion_factors(segment_dimensions, params)

        self._materials = generate_materials(params)

        for segment_name, segment_def in HUMANOID_SEGMENTS.items():
            visual_shape = self._lookup_library_shape(segment_name)
            self._links[segment_name] = generate_link(
                segment_name,
                segment_def,
                params,
                segment_masses.get(segment_name, 1.0),
                segment_dimensions.get(
                    segment_name, {"length": 0.1, "width": 0.05, "depth": 0.05}
                ),
                gender_factor,
                mesh_dir,
                self.config.inertia_mode,
                self.mesh_inertia_calc,
                self.primitive_inertia_calc,
                self.config.generate_collision,
                visual_shape=visual_shape,
            )

        for joint_name, joint_def in HUMANOID_JOINTS.items():
            extra_links, joints = generate_joint(
                joint_name,
                joint_def,
                self.config.expand_composite_joints,
            )
            for link in extra_links:
                self._links[link.name] = link
            self._joints.extend(joints)

        return HumanoidModel(self._links, self._joints)

    def _lookup_library_shape(self, segment_name: str) -> BodyPartShape | None:
        """Resolve a per-segment :class:`BodyPartShape` from the configured
        :class:`ShapeLibrary`, if any. Strips left/right side prefixes so
        the bundled default library (which has one ``upper_arm`` entry,
        not two) covers both sides.
        """
        library = getattr(self.config, "shape_library", None)
        if library is None:
            return None
        candidates = (segment_name,)
        for prefix in ("left_", "right_"):
            if segment_name.startswith(prefix):
                candidates = (segment_name, segment_name[len(prefix) :])
                break
        for name in candidates:
            if name in library.names():
                return library.get(name)
        return None

    @precondition(lambda params: params is not None, "params must not be None")
    @precondition(lambda params: params.height_m > 0, "Height must be positive")
    @precondition(lambda params: params.mass_kg > 0, "Mass must be positive")
    @postcondition(
        lambda result: _is_valid_xml(result), "Generated URDF must be valid XML"
    )
    @postcondition(lambda result: len(result) > 0, "URDF output must not be empty")
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

        urdf_xml = build_urdf_xml(
            params.name,
            self._links,
            self._joints,
            self._materials,
            self.config.pretty_print,
            self.config.indent,
        )

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(urdf_xml)
            logger.info(f"URDF written to {output_path}")
        return urdf_xml

    def _apply_proportion_factors(
        self, dimensions: dict[str, dict[str, float]], params: BodyParameters
    ) -> dict[str, dict[str, float]]:
        """Apply proportion factors to segment dimensions."""
        if dimensions is None:
            raise ValueError("dimensions must be provided")
        scaled = {}
        for seg_name, dims in dimensions.items():
            scaled_dims = dims.copy()
            seg_lower = seg_name.lower()
            if "arm" in seg_lower or "hand" in seg_lower:
                scaled_dims["length"] *= params.arm_length_factor
            elif "thigh" in seg_lower or "shin" in seg_lower or "foot" in seg_lower:
                scaled_dims["length"] *= params.leg_length_factor
            elif "thorax" in seg_lower or "lumbar" in seg_lower:
                scaled_dims["length"] *= params.torso_length_factor
                scaled_dims["width"] *= params.shoulder_width_factor
            elif "pelvis" in seg_lower:
                scaled_dims["width"] *= params.hip_width_factor
            elif "head" in seg_lower:
                for key in scaled_dims:
                    scaled_dims[key] *= params.head_scale_factor
            elif "neck" in seg_lower:
                scaled_dims["length"] *= params.neck_length_factor
            width_factor = 1.0 + 0.2 * params.muscularity + 0.3 * params.body_fat_factor
            scaled_dims["width"] = scaled_dims.get("width", 0.05) * width_factor
            scaled_dims["depth"] = scaled_dims.get("depth", 0.05) * width_factor
            seg_params = params.get_segment_params(seg_name)
            scale = seg_params.scale.as_tuple()
            scaled_dims["width"] *= scale[0]
            scaled_dims["depth"] *= scale[1]
            scaled_dims["length"] *= scale[2]
            scaled[seg_name] = scaled_dims
        return scaled

    def _generate_materials(self, params: BodyParameters) -> None:
        """Generate material definitions."""
        if params is None:
            raise ValueError("params must be provided")
        skin = params.appearance.skin_tone
        self._materials["skin"] = skin.as_tuple()
        self._materials["default"] = (0.7, 0.7, 0.7, 1.0)

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
        if segment_name is None:
            raise ValueError("segment_name must be provided")
        seg_params = params.get_segment_params(segment_name)
        final_mass = seg_params.mass_kg if seg_params.has_mass_override() else mass
        inertia = self._compute_segment_inertia(
            segment_name,
            segment_def,
            seg_params,
            final_mass,
            dimensions,
            gender_factor,
            mesh_dir,
        )
        visual_geom = create_geometry_dict(segment_def, dimensions, is_collision=False)
        collision_geom = None
        if self.config.generate_collision:
            collision_geom = create_geometry_dict(
                segment_def, dimensions, is_collision=True
            )
        length = dimensions.get("length", 0.1)
        com = get_com_location(segment_name, length, gender_factor)
        self._links[segment_name] = GeneratedLink(
            name=segment_name,
            mass=final_mass,
            inertia=inertia,
            visual_geometry=visual_geom,
            collision_geometry=collision_geom,
            origin_xyz=com,
            origin_rpy=(0.0, 0.0, 0.0),
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
        if segment_name is None:
            raise ValueError("segment_name must be provided")
        if seg_params.has_inertia_override():
            override = seg_params.inertia_override
            return MeshInertiaCalculator.create_manual_inertia(
                ixx=override.get("ixx", 0.01),
                iyy=override.get("iyy", 0.01),
                izz=override.get("izz", 0.01),
                mass=mass,
                ixy=override.get("ixy", 0.0),
                ixz=override.get("ixz", 0.0),
                iyz=override.get("iyz", 0.0),
            )
        if (
            self.config.inertia_mode
            in (InertiaMode.MESH_UNIFORM_DENSITY, InertiaMode.MESH_SPECIFIED_MASS)
            and mesh_dir
        ):
            mesh_path = Path(mesh_dir) / f"{segment_name}.stl"
            if mesh_path.exists():
                try:
                    if self.config.inertia_mode == InertiaMode.MESH_SPECIFIED_MASS:
                        return self.mesh_inertia_calc.compute_from_mesh(
                            mesh_path, mass=mass
                        )
                    return self.mesh_inertia_calc.compute_from_mesh(mesh_path)
                except (KeyError, ValueError, TypeError) as e:
                    logger.warning(
                        f"Mesh inertia calculation failed for {segment_name}: {e}"
                    )
        length = dimensions.get("length", 0.1)
        width = dimensions.get("width", 0.05)
        depth = dimensions.get("depth", 0.05)
        shape, shape_dims = estimate_segment_primitive(
            segment_name, length, width, depth
        )
        return self.primitive_inertia_calc.compute(shape, mass, shape_dims)

    def _generate_joint(
        self,
        joint_name: str,
        joint_def: JointDefinition,
        dimensions: dict[str, dict[str, float]],
    ) -> None:
        """Generate URDF joint(s) from joint definition."""
        extra_links, joints = generate_joint(
            joint_name, joint_def, self.config.expand_composite_joints
        )
        for link in extra_links:
            self._links[link.name] = link
        self._joints.extend(joints)

    # Backward-compat shims
    def _generate_single_joint(
        self, joint_name: str, joint_def: JointDefinition
    ) -> None:
        """Generate a single URDF joint (backward-compat shim)."""
        self._joints.append(generate_single_joint(joint_name, joint_def))

    def _expand_composite_joint(
        self,
        joint_name: str,
        joint_def: JointDefinition,
        dimensions: dict[str, dict[str, float]],
    ) -> None:
        """Expand composite joint (backward-compat shim)."""
        extra_links, joints = expand_composite_joint(joint_name, joint_def)
        for link in extra_links:
            self._links[link.name] = link
        self._joints.extend(joints)

    def _map_joint_type(self, joint_type: Any) -> str:
        """Map joint type (backward-compat shim)."""
        return map_joint_type(joint_type)

    def _build_urdf_xml(self, robot_name: str) -> str:
        """Build URDF XML (backward-compat shim)."""
        return build_urdf_xml(
            robot_name=robot_name,
            links=self._links,
            joints=self._joints,
            materials=self._materials,
            pretty_print=self.config.pretty_print,
            indent=self.config.indent,
        )

    def _add_link_element(self, root: ET.Element, link: GeneratedLink) -> None:
        """Add link element (backward-compat shim)."""
        _add_link_element(root, link)

    def _add_geometry_element(self, parent: ET.Element, geom: dict[str, Any]) -> None:
        """Add geometry element (backward-compat shim)."""
        add_geometry_element(parent, geom)

    def _add_joint_element(self, root: ET.Element, joint: GeneratedJoint) -> None:
        """Add joint element (backward-compat shim)."""
        _add_joint_element(root, joint)

    def _create_geometry_dict(
        self,
        segment_def: SegmentDefinition,
        dimensions: dict[str, float],
        is_collision: bool,
    ) -> dict[str, Any]:
        """Create geometry dict (backward-compat shim)."""
        return create_geometry_dict(segment_def, dimensions, is_collision)


def _is_valid_xml(xml_str: str) -> bool:
    """Return True if *xml_str* is parseable XML."""
    try:
        DefusedET.fromstring(xml_str)  # nosec B314 — parsing self-generated URDF, not untrusted input
        return True
    except ET.ParseError:
        return False


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
