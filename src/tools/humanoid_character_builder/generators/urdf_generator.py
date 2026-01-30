"""
Standalone URDF generator for humanoid characters.

This module generates complete URDF files from body parameters,
segment definitions, and computed inertias. It is fully self-contained
and does not depend on other Golf Modeling Suite modules.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.dom import minidom

from humanoid_character_builder.core.anthropometry import (
    estimate_segment_dimensions,
    estimate_segment_masses,
    get_com_location,
    estimate_segment_inertia_from_gyration,
)
from humanoid_character_builder.core.body_parameters import BodyParameters
from humanoid_character_builder.core.segment_definitions import (
    HUMANOID_JOINTS,
    HUMANOID_SEGMENTS,
    GeometryType,
    JointDefinition,
    JointType,
    SegmentDefinition,
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


@dataclass
class GeneratedLink:
    """Generated URDF link data."""

    name: str
    mass: float
    inertia: InertiaResult
    visual_geometry: dict[str, Any]
    collision_geometry: dict[str, Any] | None
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]


@dataclass
class GeneratedJoint:
    """Generated URDF joint data."""

    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]
    axis: tuple[float, float, float]
    limits: dict[str, float] | None
    dynamics: dict[str, float]


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
        # Validate parameters
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
        segment_dimensions = self._apply_proportion_factors(
            segment_dimensions, params
        )

        # Generate materials
        self._generate_materials(params)

        # Generate links
        for segment_name, segment_def in HUMANOID_SEGMENTS.items():
            self._generate_link(
                segment_name,
                segment_def,
                params,
                segment_masses.get(segment_name, 1.0),
                segment_dimensions.get(segment_name, {"length": 0.1, "width": 0.05, "depth": 0.05}),
                gender_factor,
                mesh_dir,
            )

        # Generate joints
        for joint_name, joint_def in HUMANOID_JOINTS.items():
            self._generate_joint(joint_name, joint_def, segment_dimensions)

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
        scaled = {}

        for seg_name, dims in dimensions.items():
            scaled_dims = dims.copy()

            # Apply segment-specific factors
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

            # Apply muscularity/body fat to widths
            width_factor = 1.0 + 0.2 * params.muscularity + 0.3 * params.body_fat_factor
            scaled_dims["width"] = scaled_dims.get("width", 0.05) * width_factor
            scaled_dims["depth"] = scaled_dims.get("depth", 0.05) * width_factor

            # Apply individual segment overrides
            seg_params = params.get_segment_params(seg_name)
            scale = seg_params.scale.as_tuple()
            scaled_dims["width"] *= scale[0]
            scaled_dims["depth"] *= scale[1]
            scaled_dims["length"] *= scale[2]

            scaled[seg_name] = scaled_dims

        return scaled

    def _generate_materials(self, params: BodyParameters) -> None:
        """Generate material definitions."""
        # Skin material
        skin = params.appearance.skin_tone
        self._materials["skin"] = skin.as_tuple()

        # Default material
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
        seg_params = params.get_segment_params(segment_name)

        # Determine mass
        if seg_params.has_mass_override():
            final_mass = seg_params.mass_kg
        else:
            final_mass = mass

        # Compute inertia
        inertia = self._compute_segment_inertia(
            segment_name,
            segment_def,
            seg_params,
            final_mass,
            dimensions,
            gender_factor,
            mesh_dir,
        )

        # Generate geometry
        visual_geom = self._create_geometry_dict(
            segment_def, dimensions, is_collision=False
        )
        collision_geom = None
        if self.config.generate_collision:
            collision_geom = self._create_geometry_dict(
                segment_def, dimensions, is_collision=True
            )

        # Get center of mass
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
        # Check for manual override
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

        # Try mesh-based calculation
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
                    else:
                        return self.mesh_inertia_calc.compute_from_mesh(mesh_path)
                except Exception as e:
                    logger.warning(
                        f"Mesh inertia calculation failed for {segment_name}: {e}"
                    )

        # Fall back to primitive approximation
        length = dimensions.get("length", 0.1)
        width = dimensions.get("width", 0.05)
        depth = dimensions.get("depth", 0.05)

        shape, shape_dims = estimate_segment_primitive(segment_name, length, width, depth)
        return self.primitive_inertia_calc.compute(shape, mass, shape_dims)

    def _create_geometry_dict(
        self,
        segment_def: SegmentDefinition,
        dimensions: dict[str, float],
        is_collision: bool,
    ) -> dict[str, Any]:
        """Create geometry specification dictionary."""
        geom_spec = (
            segment_def.get_collision_geometry()
            if is_collision
            else segment_def.visual_geometry
        )

        length = dimensions.get("length", 0.1)
        width = dimensions.get("width", 0.05)
        depth = dimensions.get("depth", 0.05)

        # Scale dimensions based on geometry type
        if geom_spec.geometry_type == GeometryType.BOX:
            return {
                "type": "box",
                "size": (width, depth, length),
            }
        elif geom_spec.geometry_type == GeometryType.CYLINDER:
            radius = (width + depth) / 4
            return {
                "type": "cylinder",
                "radius": radius,
                "length": length,
            }
        elif geom_spec.geometry_type == GeometryType.SPHERE:
            radius = length / 2
            return {
                "type": "sphere",
                "radius": radius,
            }
        elif geom_spec.geometry_type == GeometryType.CAPSULE:
            radius = (width + depth) / 4
            return {
                "type": "cylinder",  # URDF doesn't have capsule, use cylinder
                "radius": radius,
                "length": max(0.01, length - 2 * radius),
            }
        elif geom_spec.geometry_type == GeometryType.MESH:
            return {
                "type": "mesh",
                "filename": geom_spec.mesh_path,
                "scale": geom_spec.mesh_scale,
            }
        else:
            # Default to box
            return {
                "type": "box",
                "size": (width, depth, length),
            }

    def _generate_joint(
        self,
        joint_name: str,
        joint_def: JointDefinition,
        dimensions: dict[str, dict[str, float]],
    ) -> None:
        """Generate URDF joint(s) from joint definition."""
        if joint_def.is_composite() and self.config.expand_composite_joints:
            # Expand to multiple revolute joints
            self._expand_composite_joint(joint_name, joint_def, dimensions)
        else:
            self._generate_single_joint(joint_name, joint_def)

    def _generate_single_joint(
        self,
        joint_name: str,
        joint_def: JointDefinition,
    ) -> None:
        """Generate a single URDF joint."""
        # Map joint type
        urdf_type = self._map_joint_type(joint_def.joint_type)

        # Get limits for non-fixed joints
        limits = None
        if urdf_type in ("revolute", "prismatic"):
            limits = joint_def.limits.as_dict()

        self._joints.append(
            GeneratedJoint(
                name=joint_name,
                joint_type=urdf_type,
                parent=joint_def.parent_segment,
                child=joint_def.child_segment,
                origin_xyz=joint_def.origin_xyz,
                origin_rpy=joint_def.origin_rpy,
                axis=joint_def.axis,
                limits=limits,
                dynamics={
                    "damping": joint_def.damping,
                    "friction": joint_def.friction,
                },
            )
        )

    def _expand_composite_joint(
        self,
        joint_name: str,
        joint_def: JointDefinition,
        dimensions: dict[str, dict[str, float]],
    ) -> None:
        """Expand composite joint into multiple revolute joints."""
        if joint_def.joint_type == JointType.GIMBAL:
            axes = [
                joint_def.axis,
                joint_def.secondary_axis or (0.0, 1.0, 0.0),
                joint_def.tertiary_axis or (1.0, 0.0, 0.0),
            ]
            suffixes = ["_z", "_y", "_x"]
        elif joint_def.joint_type == JointType.UNIVERSAL:
            axes = [
                joint_def.axis,
                joint_def.secondary_axis or (0.0, 1.0, 0.0),
            ]
            suffixes = ["_1", "_2"]
        else:
            # Not composite, generate single joint
            self._generate_single_joint(joint_name, joint_def)
            return

        # Create intermediate links for composite joints
        parent = joint_def.parent_segment

        for i, (axis, suffix) in enumerate(zip(axes, suffixes)):
            is_last = i == len(axes) - 1
            child = joint_def.child_segment if is_last else f"{joint_name}{suffix}_link"

            # Create intermediate link if not last
            if not is_last:
                self._links[child] = GeneratedLink(
                    name=child,
                    mass=0.001,  # Minimal mass for intermediate link
                    inertia=InertiaResult.create_default(0.001),
                    visual_geometry={"type": "sphere", "radius": 0.001},
                    collision_geometry=None,
                    origin_xyz=(0.0, 0.0, 0.0),
                    origin_rpy=(0.0, 0.0, 0.0),
                )

            # Origin only for first joint
            origin_xyz = joint_def.origin_xyz if i == 0 else (0.0, 0.0, 0.0)
            origin_rpy = joint_def.origin_rpy if i == 0 else (0.0, 0.0, 0.0)

            self._joints.append(
                GeneratedJoint(
                    name=f"{joint_name}{suffix}",
                    joint_type="revolute",
                    parent=parent,
                    child=child,
                    origin_xyz=origin_xyz,
                    origin_rpy=origin_rpy,
                    axis=axis,
                    limits=joint_def.limits.as_dict(),
                    dynamics={
                        "damping": joint_def.damping,
                        "friction": joint_def.friction,
                    },
                )
            )

            parent = child

    def _map_joint_type(self, joint_type: JointType) -> str:
        """Map internal joint type to URDF joint type string."""
        mapping = {
            JointType.FIXED: "fixed",
            JointType.REVOLUTE: "revolute",
            JointType.CONTINUOUS: "continuous",
            JointType.PRISMATIC: "prismatic",
            JointType.FLOATING: "floating",
            JointType.PLANAR: "planar",
            JointType.UNIVERSAL: "revolute",  # Expanded separately
            JointType.GIMBAL: "revolute",  # Expanded separately
            JointType.SPHERICAL: "revolute",  # Expanded separately
        }
        return mapping.get(joint_type, "fixed")

    def _build_urdf_xml(self, robot_name: str) -> str:
        """Build the complete URDF XML."""
        root = ET.Element("robot", name=robot_name)

        # Add materials
        for mat_name, rgba in self._materials.items():
            material = ET.SubElement(root, "material", name=mat_name)
            ET.SubElement(
                material,
                "color",
                rgba=f"{rgba[0]:.4f} {rgba[1]:.4f} {rgba[2]:.4f} {rgba[3]:.4f}",
            )

        # Add links
        for link_data in self._links.values():
            self._add_link_element(root, link_data)

        # Add joints
        for joint_data in self._joints:
            self._add_joint_element(root, joint_data)

        # Format XML
        if self.config.pretty_print:
            xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(
                indent=self.config.indent
            )
            # Remove extra blank lines
            lines = [line for line in xml_str.split("\n") if line.strip()]
            return "\n".join(lines)
        else:
            return ET.tostring(root, encoding="unicode")

    def _add_link_element(self, root: ET.Element, link: GeneratedLink) -> None:
        """Add a link element to the URDF."""
        link_elem = ET.SubElement(root, "link", name=link.name)

        # Inertial
        inertial = ET.SubElement(link_elem, "inertial")
        ET.SubElement(
            inertial,
            "origin",
            xyz=f"{link.origin_xyz[0]:.6f} {link.origin_xyz[1]:.6f} {link.origin_xyz[2]:.6f}",
            rpy=f"{link.origin_rpy[0]:.6f} {link.origin_rpy[1]:.6f} {link.origin_rpy[2]:.6f}",
        )
        ET.SubElement(inertial, "mass", value=f"{link.mass:.6f}")

        inertia = link.inertia
        ET.SubElement(
            inertial,
            "inertia",
            ixx=f"{inertia.ixx:.8f}",
            ixy=f"{inertia.ixy:.8f}",
            ixz=f"{inertia.ixz:.8f}",
            iyy=f"{inertia.iyy:.8f}",
            iyz=f"{inertia.iyz:.8f}",
            izz=f"{inertia.izz:.8f}",
        )

        # Visual
        visual = ET.SubElement(link_elem, "visual")
        ET.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")
        self._add_geometry_element(visual, link.visual_geometry)
        ET.SubElement(visual, "material", name="skin")

        # Collision
        if link.collision_geometry:
            collision = ET.SubElement(link_elem, "collision")
            ET.SubElement(collision, "origin", xyz="0 0 0", rpy="0 0 0")
            self._add_geometry_element(collision, link.collision_geometry)

    def _add_geometry_element(
        self, parent: ET.Element, geom: dict[str, Any]
    ) -> None:
        """Add geometry element."""
        geometry = ET.SubElement(parent, "geometry")

        geom_type = geom["type"]
        if geom_type == "box":
            size = geom["size"]
            ET.SubElement(
                geometry, "box", size=f"{size[0]:.6f} {size[1]:.6f} {size[2]:.6f}"
            )
        elif geom_type == "cylinder":
            ET.SubElement(
                geometry,
                "cylinder",
                radius=f"{geom['radius']:.6f}",
                length=f"{geom['length']:.6f}",
            )
        elif geom_type == "sphere":
            ET.SubElement(geometry, "sphere", radius=f"{geom['radius']:.6f}")
        elif geom_type == "mesh":
            scale = geom.get("scale", (1.0, 1.0, 1.0))
            ET.SubElement(
                geometry,
                "mesh",
                filename=geom["filename"],
                scale=f"{scale[0]:.6f} {scale[1]:.6f} {scale[2]:.6f}",
            )

    def _add_joint_element(self, root: ET.Element, joint: GeneratedJoint) -> None:
        """Add a joint element to the URDF."""
        joint_elem = ET.SubElement(root, "joint", name=joint.name, type=joint.joint_type)

        ET.SubElement(joint_elem, "parent", link=joint.parent)
        ET.SubElement(joint_elem, "child", link=joint.child)
        ET.SubElement(
            joint_elem,
            "origin",
            xyz=f"{joint.origin_xyz[0]:.6f} {joint.origin_xyz[1]:.6f} {joint.origin_xyz[2]:.6f}",
            rpy=f"{joint.origin_rpy[0]:.6f} {joint.origin_rpy[1]:.6f} {joint.origin_rpy[2]:.6f}",
        )

        if joint.joint_type != "fixed":
            ET.SubElement(
                joint_elem,
                "axis",
                xyz=f"{joint.axis[0]:.6f} {joint.axis[1]:.6f} {joint.axis[2]:.6f}",
            )

        if joint.limits and joint.joint_type in ("revolute", "prismatic"):
            ET.SubElement(
                joint_elem,
                "limit",
                lower=f"{joint.limits['lower']:.6f}",
                upper=f"{joint.limits['upper']:.6f}",
                effort=f"{joint.limits['effort']:.2f}",
                velocity=f"{joint.limits['velocity']:.2f}",
            )

        if joint.dynamics:
            ET.SubElement(
                joint_elem,
                "dynamics",
                damping=f"{joint.dynamics['damping']:.4f}",
                friction=f"{joint.dynamics['friction']:.4f}",
            )


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
    generator = HumanoidURDFGenerator(config)
    return generator.generate(params, output_path)
