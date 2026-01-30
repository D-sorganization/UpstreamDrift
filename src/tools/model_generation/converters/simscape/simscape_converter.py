"""
SimScape Multibody to URDF converter.

Converts MATLAB SimScape Multibody models to URDF format,
enabling use of models created in Simulink with ROS and other robotics frameworks.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from model_generation.converters.simscape.mdl_parser import (
    MDLParser,
    SimscapeBlock,
    SimscapeBlockType,
    SimscapeModel,
)
from model_generation.core.types import (
    Geometry,
    GeometryType,
    Inertia,
    Joint,
    JointDynamics,
    JointLimits,
    JointType,
    Link,
    Material,
    Origin,
)

logger = logging.getLogger(__name__)


@dataclass
class ConversionConfig:
    """Configuration for SimScape to URDF conversion."""

    # Output settings
    robot_name: str | None = None  # Override robot name
    output_dir: Path | None = None  # Output directory for meshes

    # Unit conversion (SimScape can use various units)
    length_unit: str = "m"  # Input length unit: m, cm, mm, in
    mass_unit: str = "kg"  # Input mass unit: kg, g, lb
    angle_unit: str = "rad"  # Input angle unit: rad, deg

    # Geometry options
    include_visual: bool = True
    include_collision: bool = True
    collision_from_visual: bool = True  # Use visual geometry for collision
    default_material_color: tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0)

    # Joint options
    default_joint_damping: float = 0.1
    default_joint_friction: float = 0.0
    default_effort_limit: float = 1000.0
    default_velocity_limit: float = 10.0

    # Handling unknown blocks
    skip_unknown: bool = True
    create_dummy_links: bool = True  # Create links for unknown bodies


@dataclass
class ConversionResult:
    """Result of SimScape to URDF conversion."""

    success: bool
    links: list[Link] = field(default_factory=list)
    joints: list[Joint] = field(default_factory=list)
    materials: dict[str, Material] = field(default_factory=dict)
    robot_name: str = "converted_robot"
    urdf_string: str | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    source_model: SimscapeModel | None = None


class SimscapeToURDFConverter:
    """
    Convert SimScape Multibody models to URDF.

    This converter handles the translation of:
    - SimScape Solids (Brick, Cylinder, Sphere) -> URDF Links with geometry
    - SimScape Joints -> URDF Joints
    - SimScape Rigid Transforms -> Origin offsets
    - SimScape Inertia blocks -> URDF Inertial elements

    Limitations:
    - Complex parametric geometry may need manual adjustment
    - MATLAB expressions in parameters are not evaluated
    - Some SimScape features (constraints, actuators) are not directly convertible

    Example:
        converter = SimscapeToURDFConverter()
        result = converter.convert("/path/to/robot.slx")

        if result.success:
            with open("robot.urdf", "w") as f:
                f.write(result.urdf_string)
    """

    # Unit conversion factors to meters/kg/radians
    LENGTH_FACTORS = {
        "m": 1.0,
        "cm": 0.01,
        "mm": 0.001,
        "in": 0.0254,
        "ft": 0.3048,
    }

    MASS_FACTORS = {
        "kg": 1.0,
        "g": 0.001,
        "lb": 0.453592,
        "oz": 0.0283495,
    }

    ANGLE_FACTORS = {
        "rad": 1.0,
        "deg": math.pi / 180.0,
    }

    # SimScape joint type to URDF joint type mapping
    JOINT_TYPE_MAP = {
        SimscapeBlockType.REVOLUTE_JOINT: JointType.REVOLUTE,
        SimscapeBlockType.PRISMATIC_JOINT: JointType.PRISMATIC,
        SimscapeBlockType.WELD_JOINT: JointType.FIXED,
        SimscapeBlockType.SPHERICAL_JOINT: JointType.GIMBAL,  # Approximation
        SimscapeBlockType.UNIVERSAL_JOINT: JointType.UNIVERSAL,
        SimscapeBlockType.CYLINDRICAL_JOINT: JointType.REVOLUTE,  # Approximation
        SimscapeBlockType.PLANAR_JOINT: JointType.PLANAR,
        SimscapeBlockType.GIMBAL_JOINT: JointType.GIMBAL,
        SimscapeBlockType.SIX_DOF_JOINT: JointType.FLOATING,
    }

    def __init__(self, config: ConversionConfig | None = None):
        """
        Initialize converter.

        Args:
            config: Conversion configuration
        """
        self.config = config or ConversionConfig()
        self._parser = MDLParser()
        self._link_counter = 0
        self._joint_counter = 0

    def convert(
        self,
        source: str | Path,
        output_path: Path | None = None,
    ) -> ConversionResult:
        """
        Convert a SimScape model file to URDF.

        Args:
            source: Path to MDL/SLX file
            output_path: Optional path to write URDF file

        Returns:
            ConversionResult with converted model
        """
        result = ConversionResult(success=False)

        try:
            # Parse SimScape model
            simscape_model = self._parser.parse(source)
            result.source_model = simscape_model
            result.warnings.extend(simscape_model.warnings)

            # Set robot name
            result.robot_name = self.config.robot_name or simscape_model.name

            # Convert to URDF elements
            self._convert_model(simscape_model, result)

            if result.errors:
                return result

            # Generate URDF string
            result.urdf_string = self._generate_urdf(result)
            result.success = True

            # Write to file if requested
            if output_path and result.urdf_string:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(result.urdf_string)
                logger.info(f"Wrote URDF to {output_path}")

        except Exception as e:
            result.errors.append(f"Conversion failed: {e}")
            logger.exception("Conversion error")

        return result

    def convert_string(
        self,
        content: str,
        format: str = "mdl",
    ) -> ConversionResult:
        """
        Convert SimScape model from string content.

        Args:
            content: Model content as string
            format: Format ('mdl' or 'xml')

        Returns:
            ConversionResult
        """
        result = ConversionResult(success=False)

        try:
            simscape_model = self._parser.parse_string(content, format)
            result.source_model = simscape_model
            result.warnings.extend(simscape_model.warnings)

            result.robot_name = self.config.robot_name or simscape_model.name

            self._convert_model(simscape_model, result)

            if not result.errors:
                result.urdf_string = self._generate_urdf(result)
                result.success = True

        except Exception as e:
            result.errors.append(f"Conversion failed: {e}")
            logger.exception("Conversion error")

        return result

    def _convert_model(
        self,
        model: SimscapeModel,
        result: ConversionResult,
    ) -> None:
        """Convert SimScape model to URDF elements."""
        # Reset counters
        self._link_counter = 0
        self._joint_counter = 0

        # Build connection graph
        connection_map = self._build_connection_map(model)

        # Convert body blocks to links
        body_to_link: dict[str, str] = {}  # SimScape block name -> URDF link name

        for body_block in model.get_body_blocks():
            link = self._convert_body_to_link(body_block)
            if link:
                result.links.append(link)
                body_to_link[body_block.full_path] = link.name

        # If no bodies found, try to infer from joints
        if not result.links:
            result.warnings.append("No body blocks found, inferring from joints")
            self._infer_links_from_joints(model, result, body_to_link)

        # Add world frame as base link if needed
        if not any(
            link.name == "world" or link.name == "base_link" for link in result.links
        ):
            base_link = Link(
                name="base_link",
                inertia=Inertia(ixx=0.001, iyy=0.001, izz=0.001, mass=0.001),
            )
            result.links.insert(0, base_link)
            body_to_link["world"] = "base_link"
            body_to_link["WorldFrame"] = "base_link"

        # Convert joint blocks
        for joint_block in model.get_joint_blocks():
            joint = self._convert_joint_block(
                joint_block, model, body_to_link, connection_map
            )
            if joint:
                result.joints.append(joint)

        # Handle rigid transforms as fixed joints
        for transform_block in model.get_transform_blocks():
            joint = self._convert_transform_to_joint(
                transform_block, model, body_to_link, connection_map
            )
            if joint:
                result.joints.append(joint)

        # Validate and fix orphan links
        self._connect_orphan_links(result, body_to_link)

        # Create default material
        default_material = Material(
            name="default_material",
            color=self.config.default_material_color,
        )
        result.materials["default_material"] = default_material

    def _build_connection_map(
        self, model: SimscapeModel
    ) -> dict[str, list[tuple[str, str]]]:
        """Build a map of block connections."""
        # Maps block name to list of (connected_block, port) tuples
        connection_map: dict[str, list[tuple[str, str]]] = {}

        for conn in model.connections:
            # Forward connection
            if conn.source_block not in connection_map:
                connection_map[conn.source_block] = []
            connection_map[conn.source_block].append(
                (conn.dest_block, conn.source_port)
            )

            # Reverse connection
            if conn.dest_block not in connection_map:
                connection_map[conn.dest_block] = []
            connection_map[conn.dest_block].append((conn.source_block, conn.dest_port))

        return connection_map

    def _convert_body_to_link(self, block: SimscapeBlock) -> Link | None:
        """Convert a SimScape body block to URDF Link."""
        # Get or generate link name
        link_name = self._sanitize_name(block.name)

        # Get mass
        mass = self._get_mass(block)

        # Get inertia
        inertia = self._get_inertia(block, mass)

        # Get geometry
        visual_geometry = None
        collision_geometry = None

        if self.config.include_visual:
            visual_geometry = self._get_geometry(block)

        if self.config.include_collision:
            if self.config.collision_from_visual and visual_geometry:
                collision_geometry = visual_geometry
            else:
                collision_geometry = self._get_geometry(block)

        # Create link
        link = Link(
            name=link_name,
            inertia=inertia,
            visual_geometry=visual_geometry,
            collision_geometry=collision_geometry,
            visual_material=Material(
                name="default_material",
                color=self.config.default_material_color,
            ),
        )

        return link

    def _get_mass(self, block: SimscapeBlock) -> float:
        """Extract mass from body block."""
        mass = 1.0  # Default

        # Try common parameter names
        for param_name in ["Mass", "m", "mass", "MassValue"]:
            if param_name in block.parameters:
                mass = block.get_param_float(param_name, 1.0)
                break

        # Apply unit conversion
        mass *= self.MASS_FACTORS.get(self.config.mass_unit, 1.0)

        return max(0.001, mass)  # Ensure positive mass

    def _get_inertia(self, block: SimscapeBlock, mass: float) -> Inertia:
        """Extract inertia from body block."""
        # Check for explicit inertia tensor
        inertia_param = block.parameters.get("Inertia") or block.parameters.get(
            "MomentOfInertia"
        )

        if inertia_param:
            # Parse inertia matrix [ixx iyy izz ixy ixz iyz] or similar
            values = inertia_param.as_vector()
            if len(values) >= 3:
                ixx, iyy, izz = values[0], values[1], values[2]
                ixy = values[3] if len(values) > 3 else 0.0
                ixz = values[4] if len(values) > 4 else 0.0
                iyz = values[5] if len(values) > 5 else 0.0
                return Inertia(
                    ixx=ixx, iyy=iyy, izz=izz, ixy=ixy, ixz=ixz, iyz=iyz, mass=mass
                )

        # Calculate from geometry
        geometry = self._get_geometry(block)
        if geometry:
            return self._inertia_from_geometry(geometry, mass)

        # Default minimal inertia
        return Inertia(ixx=0.01, iyy=0.01, izz=0.01, mass=mass)

    def _get_geometry(self, block: SimscapeBlock) -> Geometry | None:
        """Extract geometry from body block."""
        block_type = block.block_type
        length_scale = self.LENGTH_FACTORS.get(self.config.length_unit, 1.0)

        if block_type == SimscapeBlockType.BRICK_SOLID:
            # Get dimensions [x y z]
            dims = block.get_param_vector("Dimensions", (0.1, 0.1, 0.1))
            if len(dims) >= 3:
                return Geometry.box(
                    dims[0] * length_scale,
                    dims[1] * length_scale,
                    dims[2] * length_scale,
                )

        elif block_type == SimscapeBlockType.CYLINDER_SOLID:
            radius = block.get_param_float("Radius", 0.05) * length_scale
            length = block.get_param_float("Length", 0.1) * length_scale
            return Geometry.cylinder(radius, length)

        elif block_type == SimscapeBlockType.SPHERE_SOLID:
            radius = block.get_param_float("Radius", 0.05) * length_scale
            return Geometry.sphere(radius)

        elif block_type == SimscapeBlockType.SOLID:
            # Generic solid - try to determine shape
            shape = block.get_param("Shape", "box").lower()
            if "cylinder" in shape:
                radius = block.get_param_float("Radius", 0.05) * length_scale
                length = block.get_param_float("Length", 0.1) * length_scale
                return Geometry.cylinder(radius, length)
            elif "sphere" in shape:
                radius = block.get_param_float("Radius", 0.05) * length_scale
                return Geometry.sphere(radius)
            else:
                dims = block.get_param_vector("Dimensions", (0.1, 0.1, 0.1))
                if len(dims) >= 3:
                    return Geometry.box(
                        dims[0] * length_scale,
                        dims[1] * length_scale,
                        dims[2] * length_scale,
                    )

        # Default box for unknown geometries
        if self.config.create_dummy_links:
            return Geometry.box(0.05, 0.05, 0.05)

        return None

    def _inertia_from_geometry(self, geometry: Geometry, mass: float) -> Inertia:
        """Calculate inertia from geometry."""
        if geometry.geometry_type == GeometryType.BOX:
            return Inertia.from_box(mass, *geometry.dimensions[:3])
        elif geometry.geometry_type == GeometryType.CYLINDER:
            return Inertia.from_cylinder(
                mass, geometry.dimensions[0], geometry.dimensions[1]
            )
        elif geometry.geometry_type == GeometryType.SPHERE:
            return Inertia.from_sphere(mass, geometry.dimensions[0])
        else:
            return Inertia(ixx=0.01, iyy=0.01, izz=0.01, mass=mass)

    def _convert_joint_block(
        self,
        block: SimscapeBlock,
        model: SimscapeModel,
        body_to_link: dict[str, str],
        connection_map: dict[str, list[tuple[str, str]]],
    ) -> Joint | None:
        """Convert a SimScape joint block to URDF Joint."""
        # Determine joint type
        urdf_joint_type = self.JOINT_TYPE_MAP.get(block.block_type, JointType.FIXED)

        # Find connected bodies (parent/child)
        connections = connection_map.get(block.full_path, [])

        parent_link = None
        child_link = None

        # SimScape joints typically have Base (B) and Follower (F) ports
        for connected_block, port in connections:
            link_name = body_to_link.get(connected_block)
            if link_name:
                # Determine if this is base or follower
                port_lower = port.lower() if port else ""
                if "b" in port_lower or "base" in port_lower or parent_link is None:
                    if parent_link is None:
                        parent_link = link_name
                    else:
                        child_link = link_name
                else:
                    child_link = link_name

        if not parent_link or not child_link:
            # Try to find any connected links
            for connected_block, _ in connections:
                link_name = body_to_link.get(connected_block)
                if link_name:
                    if parent_link is None:
                        parent_link = link_name
                    elif child_link is None and link_name != parent_link:
                        child_link = link_name

        if not parent_link or not child_link:
            logger.warning(f"Could not find parent/child for joint {block.name}")
            return None

        # Get joint origin from transforms
        origin = self._get_joint_origin(block)

        # Get axis
        axis = self._get_joint_axis(block)

        # Get limits
        limits = self._get_joint_limits(block, urdf_joint_type)

        # Generate joint name
        joint_name = self._sanitize_name(block.name)

        joint = Joint(
            name=joint_name,
            joint_type=urdf_joint_type,
            parent=parent_link,
            child=child_link,
            origin=origin,
            axis=axis,
            limits=limits,
            dynamics=JointDynamics(
                damping=self.config.default_joint_damping,
                friction=self.config.default_joint_friction,
            ),
        )

        return joint

    def _get_joint_origin(self, block: SimscapeBlock) -> Origin:
        """Extract joint origin from block parameters."""
        length_scale = self.LENGTH_FACTORS.get(self.config.length_unit, 1.0)
        angle_scale = self.ANGLE_FACTORS.get(self.config.angle_unit, 1.0)

        # Try common parameter names for position
        xyz = (0.0, 0.0, 0.0)
        for param_name in ["Position", "Offset", "Translation", "xyz"]:
            if param_name in block.parameters:
                pos = block.get_param_vector(param_name, (0.0, 0.0, 0.0))
                if len(pos) >= 3:
                    xyz = (
                        pos[0] * length_scale,
                        pos[1] * length_scale,
                        pos[2] * length_scale,
                    )
                    break

        # Try common parameter names for rotation
        rpy = (0.0, 0.0, 0.0)
        for param_name in ["Rotation", "Orientation", "rpy", "Angles"]:
            if param_name in block.parameters:
                rot = block.get_param_vector(param_name, (0.0, 0.0, 0.0))
                if len(rot) >= 3:
                    rpy = (
                        rot[0] * angle_scale,
                        rot[1] * angle_scale,
                        rot[2] * angle_scale,
                    )
                    break

        return Origin(xyz=xyz, rpy=rpy)

    def _get_joint_axis(self, block: SimscapeBlock) -> tuple[float, float, float]:
        """Extract joint axis from block parameters."""
        # Try common parameter names
        for param_name in ["Axis", "JointAxis", "RotationAxis"]:
            if param_name in block.parameters:
                axis = block.get_param_vector(param_name, (0.0, 0.0, 1.0))
                if len(axis) >= 3:
                    # Normalize
                    mag = (axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2) ** 0.5
                    if mag > 0:
                        return (axis[0] / mag, axis[1] / mag, axis[2] / mag)

        # Check axis direction parameter
        axis_dir = block.get_param("AxisDirection", "").lower()
        if "x" in axis_dir:
            return (1.0, 0.0, 0.0)
        elif "y" in axis_dir:
            return (0.0, 1.0, 0.0)

        # Default to Z axis
        return (0.0, 0.0, 1.0)

    def _get_joint_limits(
        self,
        block: SimscapeBlock,
        joint_type: JointType,
    ) -> JointLimits | None:
        """Extract joint limits from block parameters."""
        if joint_type in (JointType.FIXED, JointType.FLOATING, JointType.CONTINUOUS):
            return None

        angle_scale = self.ANGLE_FACTORS.get(self.config.angle_unit, 1.0)
        length_scale = self.LENGTH_FACTORS.get(self.config.length_unit, 1.0)

        lower = -math.pi
        upper = math.pi

        # Check for limit parameters
        if "LowerLimit" in block.parameters:
            lower = block.get_param_float("LowerLimit", lower)
        if "UpperLimit" in block.parameters:
            upper = block.get_param_float("UpperLimit", upper)

        # Apply scaling
        if joint_type == JointType.PRISMATIC:
            lower *= length_scale
            upper *= length_scale
        else:
            lower *= angle_scale
            upper *= angle_scale

        return JointLimits(
            lower=lower,
            upper=upper,
            effort=self.config.default_effort_limit,
            velocity=self.config.default_velocity_limit,
        )

    def _convert_transform_to_joint(
        self,
        block: SimscapeBlock,
        model: SimscapeModel,
        body_to_link: dict[str, str],
        connection_map: dict[str, list[tuple[str, str]]],
    ) -> Joint | None:
        """Convert a RigidTransform block to a fixed URDF joint."""
        # Find connected bodies
        connections = connection_map.get(block.full_path, [])

        parent_link = None
        child_link = None

        for connected_block, _port in connections:
            link_name = body_to_link.get(connected_block)
            if link_name:
                if parent_link is None:
                    parent_link = link_name
                elif child_link is None and link_name != parent_link:
                    child_link = link_name

        if not parent_link or not child_link:
            return None

        # Get transform
        origin = self._get_joint_origin(block)

        joint_name = self._sanitize_name(f"{block.name}_fixed")

        return Joint(
            name=joint_name,
            joint_type=JointType.FIXED,
            parent=parent_link,
            child=child_link,
            origin=origin,
        )

    def _infer_links_from_joints(
        self,
        model: SimscapeModel,
        result: ConversionResult,
        body_to_link: dict[str, str],
    ) -> None:
        """Create links inferred from joint connections."""
        link_names: set[str] = set()

        for joint_block in model.get_joint_blocks():
            connections = model.get_connections_from(
                joint_block.full_path
            ) + model.get_connections_to(joint_block.full_path)

            for conn in connections:
                for block_name in [conn.source_block, conn.dest_block]:
                    if block_name == joint_block.full_path:
                        continue

                    link_name = self._sanitize_name(block_name)
                    if link_name not in link_names:
                        link_names.add(link_name)
                        body_to_link[block_name] = link_name

                        link = Link(
                            name=link_name,
                            inertia=Inertia(ixx=0.01, iyy=0.01, izz=0.01, mass=1.0),
                            visual_geometry=Geometry.box(0.05, 0.05, 0.05),
                        )
                        result.links.append(link)
                        result.warnings.append(f"Inferred link: {link_name}")

    def _connect_orphan_links(
        self,
        result: ConversionResult,
        body_to_link: dict[str, str],
    ) -> None:
        """Connect any orphan links to base with fixed joints."""
        # Find links that are not children of any joint
        child_links = {j.child for j in result.joints}
        root_candidates = [
            link.name for link in result.links if link.name not in child_links
        ]

        if len(root_candidates) <= 1:
            return  # One or zero roots is fine

        # Find the most likely root (base_link or first)
        root_link = "base_link"
        if root_link not in root_candidates:
            root_link = root_candidates[0]

        # Connect other orphans to root
        for link_name in root_candidates:
            if link_name != root_link:
                joint = Joint(
                    name=f"{root_link}_to_{link_name}_fixed",
                    joint_type=JointType.FIXED,
                    parent=root_link,
                    child=link_name,
                    origin=Origin(),
                )
                result.joints.append(joint)
                result.warnings.append(
                    f"Connected orphan link '{link_name}' to '{root_link}'"
                )

    def _sanitize_name(self, name: str) -> str:
        """Sanitize name for URDF (no special characters)."""
        # Replace path separators and special chars
        sanitized = name.replace("/", "_").replace("\\", "_")
        sanitized = sanitized.replace(" ", "_").replace("-", "_")
        sanitized = "".join(c for c in sanitized if c.isalnum() or c == "_")

        # Ensure doesn't start with number
        if sanitized and sanitized[0].isdigit():
            sanitized = "link_" + sanitized

        return sanitized or f"unnamed_{self._link_counter}"

    def _generate_urdf(self, result: ConversionResult) -> str:
        """Generate URDF XML string from conversion result."""
        from model_generation.builders.urdf_writer import URDFWriter

        writer = URDFWriter(pretty_print=True)
        return writer.write(
            result.robot_name,
            result.links,
            result.joints,
            result.materials,
        )


def convert_simscape_to_urdf(
    source: str | Path,
    output_path: Path | None = None,
    robot_name: str | None = None,
    **config_kwargs: Any,
) -> ConversionResult:
    """
    Convenience function to convert SimScape model to URDF.

    Args:
        source: Path to MDL/SLX file
        output_path: Optional output file path
        robot_name: Optional robot name override
        **config_kwargs: Additional ConversionConfig parameters

    Returns:
        ConversionResult
    """
    config = ConversionConfig(robot_name=robot_name, **config_kwargs)
    converter = SimscapeToURDFConverter(config)
    return converter.convert(source, output_path)
