"""
MJCF (MuJoCo) format converter.

This module provides bidirectional conversion between URDF and MJCF formats.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import defusedxml.ElementTree as ET
from model_generation.converters.urdf_parser import ParsedModel, URDFParser
from model_generation.core.types import (
    Geometry,
    GeometryType,
    Inertia,
    Joint,
    JointType,
    Link,
    Origin,
)

from src.shared.python.core.contracts import precondition

logger = logging.getLogger(__name__)


# Joint type mappings
URDF_TO_MJCF_JOINT = {
    JointType.FIXED: None,  # No joint in MuJoCo
    JointType.REVOLUTE: "hinge",
    JointType.CONTINUOUS: "hinge",
    JointType.PRISMATIC: "slide",
    JointType.FLOATING: "free",
    JointType.PLANAR: "slide",  # Approximation
}

MJCF_TO_URDF_JOINT = {
    "hinge": JointType.REVOLUTE,
    "slide": JointType.PRISMATIC,
    "free": JointType.FLOATING,
    "ball": JointType.GIMBAL,  # Approximation
}


@dataclass
class MJCFConfig:
    """Configuration for MJCF conversion."""

    # Compiler settings
    angle_unit: str = "radian"
    coordinate: str = "local"
    inertiafromgeom: bool = False

    # Option settings
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    timestep: float = 0.002

    # Visual settings
    include_visuals: bool = True
    include_collisions: bool = True

    # Asset handling
    mesh_dir: str = "meshes"
    texture_dir: str = "textures"

    # Default properties
    default_friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
    default_damping: float = 0.5


class MJCFConverter:
    """
    Convert between URDF and MJCF formats.

    MJCF (MuJoCo XML) is a hierarchical format where bodies are nested.
    URDF uses a flat structure with explicit parent-child relationships.
    """

    def __init__(self, config: MJCFConfig | None = None) -> None:
        """
        Initialize converter.

        Args:
            config: Conversion configuration
        """
        self.config = config or MJCFConfig()
        self._urdf_parser = URDFParser()

    @precondition(
        lambda self, source, output_path=None: source is not None,
        "Source must not be None",
    )
    def urdf_to_mjcf(
        self,
        source: str | Path | ParsedModel,
        output_path: Path | None = None,
    ) -> str:
        """
        Convert URDF to MJCF format.

        Args:
            source: URDF file path, XML string, or ParsedModel
            output_path: Optional path to save output

        Returns:
            MJCF XML string
        """
        # Parse URDF if needed
        if isinstance(source, ParsedModel):
            model = source
        else:
            model = self._urdf_parser.parse(source)

        # Build MJCF structure
        mjcf_xml = self._build_mjcf(model)

        if output_path:
            Path(output_path).write_text(mjcf_xml)

        return mjcf_xml

    @precondition(
        lambda self, source, output_path=None: source is not None,
        "Source must not be None",
    )
    def mjcf_to_urdf(
        self,
        source: str | Path,
        output_path: Path | None = None,
    ) -> str:
        """
        Convert MJCF to URDF format.

        Args:
            source: MJCF file path or XML string
            output_path: Optional path to save output

        Returns:
            URDF XML string
        """
        # Load MJCF
        if isinstance(source, Path) or (
            isinstance(source, str) and not source.strip().startswith("<")
        ):
            xml_string = Path(source).read_text()
        else:
            xml_string = source

        # Parse MJCF
        root = ET.fromstring(xml_string)
        model = self._parse_mjcf(root)

        # Convert to URDF
        urdf_xml = model.to_urdf()

        if output_path:
            Path(output_path).write_text(urdf_xml)

        return urdf_xml

    def _build_mjcf(self, model: ParsedModel) -> str:
        """Build MJCF XML from parsed model."""
        lines = []
        lines.append(f'<mujoco model="{model.name}">')

        # Compiler
        lines.append("  <compiler")
        lines.append(f'    angle="{self.config.angle_unit}"')
        lines.append(f'    coordinate="{self.config.coordinate}"')
        if self.config.inertiafromgeom:
            lines.append('    inertiafromgeom="true"')
        lines.append("  />")

        # Option
        g = self.config.gravity
        lines.append("  <option")
        lines.append(f'    gravity="{g[0]} {g[1]} {g[2]}"')
        lines.append(f'    timestep="{self.config.timestep}"')
        lines.append("  />")

        # Default
        lines.append("  <default>")
        f = self.config.default_friction
        lines.append(f'    <geom friction="{f[0]} {f[1]} {f[2]}"/>')
        lines.append(f'    <joint damping="{self.config.default_damping}"/>')
        lines.append("  </default>")

        # Assets (materials, meshes)
        if model.materials or any(
            link.visual_geometry
            and link.visual_geometry.geometry_type == GeometryType.MESH
            for link in model.links
        ):
            lines.append("  <asset>")
            for mat in model.materials.values():
                rgba = " ".join(f"{v:.4f}" for v in mat.color)
                lines.append(f'    <material name="{mat.name}" rgba="{rgba}"/>')

            # Collect mesh references
            for link in model.links:
                if (
                    link.visual_geometry
                    and link.visual_geometry.geometry_type == GeometryType.MESH
                ):
                    filename = link.visual_geometry.mesh_filename
                    mesh_name = Path(filename).stem if filename else link.name
                    lines.append(f'    <mesh name="{mesh_name}" file="{filename}"/>')
            lines.append("  </asset>")

        # Worldbody
        lines.append("  <worldbody>")

        # Build body hierarchy
        root_link = model.get_root_link()
        if root_link:
            body_lines = self._build_mjcf_body(model, root_link.name, 2)
            lines.extend(body_lines)

        lines.append("  </worldbody>")

        # Actuators (for non-fixed joints)
        actuator_lines = []
        for joint in model.joints:
            if joint.joint_type not in (JointType.FIXED,):
                actuator_lines.append(
                    f'    <motor joint="{joint.name}" name="{joint.name}_motor"/>'
                )

        if actuator_lines:
            lines.append("  <actuator>")
            lines.extend(actuator_lines)
            lines.append("  </actuator>")

        lines.append("</mujoco>")

        return "\n".join(lines)

    def _build_mjcf_body(
        self,
        model: ParsedModel,
        link_name: str,
        indent_level: int,
    ) -> list[str]:
        """Recursively build body element."""
        lines: list[str] = []
        indent = "  " * indent_level

        link = model.get_link(link_name)
        if not link:
            return lines

        # Find joint connecting this link (if not root)
        connecting_joint = None
        for joint in model.joints:
            if joint.child == link_name:
                connecting_joint = joint
                break

        # Body element with position
        pos = "0 0 0"
        if connecting_joint:
            xyz = connecting_joint.origin.xyz
            pos = f"{xyz[0]:.6g} {xyz[1]:.6g} {xyz[2]:.6g}"

        lines.append(f'{indent}<body name="{link_name}" pos="{pos}">')

        # Inertial
        if link.inertia.mass > 0:
            com = link.inertia.center_of_mass
            com_str = f"{com[0]:.6g} {com[1]:.6g} {com[2]:.6g}"

            if link.inertia.is_diagonal():
                inertia_str = (
                    f"{link.inertia.ixx:.6g} {link.inertia.iyy:.6g} "
                    f"{link.inertia.izz:.6g}"
                )
                lines.append(
                    f'{indent}  <inertial mass="{link.inertia.mass:.6g}" '
                    f'pos="{com_str}" diaginertia="{inertia_str}"/>'
                )
            else:
                inertia_str = (
                    f"{link.inertia.ixx:.6g} {link.inertia.iyy:.6g} "
                    f"{link.inertia.izz:.6g} {link.inertia.ixy:.6g} "
                    f"{link.inertia.ixz:.6g} {link.inertia.iyz:.6g}"
                )
                lines.append(
                    f'{indent}  <inertial mass="{link.inertia.mass:.6g}" '
                    f'pos="{com_str}" fullinertia="{inertia_str}"/>'
                )

        # Joint (if not root)
        if connecting_joint and connecting_joint.joint_type != JointType.FIXED:
            mjcf_type = URDF_TO_MJCF_JOINT.get(connecting_joint.joint_type, "hinge")
            axis = connecting_joint.axis
            axis_str = f"{axis[0]:.6g} {axis[1]:.6g} {axis[2]:.6g}"

            joint_line = (
                f'{indent}  <joint name="{connecting_joint.name}" '
                f'type="{mjcf_type}" axis="{axis_str}"'
            )

            if connecting_joint.limits:
                joint_line += (
                    f' range="{connecting_joint.limits.lower:.6g} '
                    f'{connecting_joint.limits.upper:.6g}"'
                )

            if connecting_joint.dynamics:
                joint_line += f' damping="{connecting_joint.dynamics.damping:.6g}"'

            joint_line += "/>"
            lines.append(joint_line)

        # Geometry (visual)
        if link.visual_geometry and self.config.include_visuals:
            geom_line = self._build_mjcf_geom(
                link.visual_geometry,
                link.visual_origin,
                link.visual_material,
                indent + "  ",
            )
            lines.append(geom_line)

        # Child bodies
        for child_name in model.get_children(link_name):
            child_lines = self._build_mjcf_body(model, child_name, indent_level + 1)
            lines.extend(child_lines)

        lines.append(f"{indent}</body>")

        return lines

    def _build_mjcf_geom(
        self,
        geometry: Geometry,
        origin: Origin,
        material: Any,
        indent: str,
    ) -> str:
        """Build geometry element."""
        pos = origin.xyz
        pos_str = f"{pos[0]:.6g} {pos[1]:.6g} {pos[2]:.6g}"

        if geometry.geometry_type == GeometryType.BOX:
            size = geometry.dimensions
            # MuJoCo uses half-sizes
            half_size = f"{size[0] / 2:.6g} {size[1] / 2:.6g} {size[2] / 2:.6g}"
            return f'{indent}<geom type="box" size="{half_size}" pos="{pos_str}"/>'

        elif geometry.geometry_type == GeometryType.CYLINDER:
            radius, length = geometry.dimensions
            # MuJoCo cylinder: radius and half-length
            return (
                f'{indent}<geom type="cylinder" '
                f'size="{radius:.6g} {length / 2:.6g}" pos="{pos_str}"/>'
            )

        elif geometry.geometry_type == GeometryType.SPHERE:
            radius = geometry.dimensions[0]
            return f'{indent}<geom type="sphere" size="{radius:.6g}" pos="{pos_str}"/>'

        elif geometry.geometry_type == GeometryType.CAPSULE:
            radius, length = geometry.dimensions
            return (
                f'{indent}<geom type="capsule" '
                f'size="{radius:.6g} {length / 2:.6g}" pos="{pos_str}"/>'
            )

        elif geometry.geometry_type == GeometryType.MESH:
            mesh_name = (
                Path(geometry.mesh_filename).stem if geometry.mesh_filename else "mesh"
            )
            return f'{indent}<geom type="mesh" mesh="{mesh_name}" pos="{pos_str}"/>'

        else:
            return f'{indent}<geom type="box" size="0.05 0.05 0.05" pos="{pos_str}"/>'

    def _parse_mjcf(self, root: ET.Element) -> ParsedModel:
        """Parse MJCF into ParsedModel."""
        model_name = root.get("model", "mjcf_model")

        links: list[Link] = []
        joints: list[Joint] = []
        materials: dict[str, Any] = {}

        # Parse assets
        asset_elem = root.find("asset")
        if asset_elem is not None:
            for mat_elem in asset_elem.findall("material"):
                name = mat_elem.get("name", "default")
                rgba_str = mat_elem.get("rgba", "0.8 0.8 0.8 1.0")
                rgba = tuple(float(v) for v in rgba_str.split())
                materials[name] = type(
                    "Material", (), {"name": name, "color": rgba, "texture": None}
                )()

        # Parse worldbody
        worldbody = root.find("worldbody")
        if worldbody is not None:
            self._parse_mjcf_body(worldbody, None, links, joints)

        # Convert materials to proper type
        from model_generation.core.types import Material

        proper_materials = {}
        for name, mat in materials.items():
            proper_materials[name] = Material(
                name=mat.name, color=mat.color, texture=mat.texture
            )

        return ParsedModel(
            name=model_name,
            links=links,
            joints=joints,
            materials=proper_materials,
        )

    def _parse_mjcf_body(
        self,
        elem: ET.Element,
        parent_name: str | None,
        links: list[Link],
        joints: list[Joint],
    ) -> None:
        """Recursively parse body elements."""
        for body_elem in elem.findall("body"):
            body_name = body_elem.get("name", f"body_{len(links)}")

            # Parse position
            pos_str = body_elem.get("pos", "0 0 0")
            pos_parts = [float(v) for v in pos_str.split()]
            pos: tuple[float, float, float] = (
                pos_parts[0],
                pos_parts[1],
                pos_parts[2] if len(pos_parts) > 2 else 0.0,
            )

            # Parse inertial
            inertia = Inertia(ixx=0.1, iyy=0.1, izz=0.1, mass=1.0)
            inertial_elem = body_elem.find("inertial")
            if inertial_elem is not None:
                mass = float(inertial_elem.get("mass", 1.0))

                com_str = inertial_elem.get("pos", "0 0 0")
                com_parts = [float(v) for v in com_str.split()]
                com: tuple[float, float, float] = (
                    com_parts[0],
                    com_parts[1],
                    com_parts[2] if len(com_parts) > 2 else 0.0,
                )

                diag_str = inertial_elem.get("diaginertia")
                full_str = inertial_elem.get("fullinertia")

                if diag_str:
                    diag = [float(v) for v in diag_str.split()]
                    inertia = Inertia(
                        ixx=diag[0],
                        iyy=diag[1],
                        izz=diag[2],
                        mass=mass,
                        center_of_mass=com,
                    )
                elif full_str:
                    full = [float(v) for v in full_str.split()]
                    inertia = Inertia(
                        ixx=full[0],
                        iyy=full[1],
                        izz=full[2],
                        ixy=full[3] if len(full) > 3 else 0,
                        ixz=full[4] if len(full) > 4 else 0,
                        iyz=full[5] if len(full) > 5 else 0,
                        mass=mass,
                        center_of_mass=com,
                    )
                else:
                    inertia = Inertia(
                        ixx=0.1, iyy=0.1, izz=0.1, mass=mass, center_of_mass=com
                    )

            # Create link
            link = Link(name=body_name, inertia=inertia)
            links.append(link)

            # Parse joint and create URDF joint to parent
            if parent_name:
                joint_elem = body_elem.find("joint")
                if joint_elem is not None:
                    joint_name = joint_elem.get("name", f"{parent_name}_to_{body_name}")
                    mjcf_type = joint_elem.get("type", "hinge")
                    joint_type = MJCF_TO_URDF_JOINT.get(mjcf_type, JointType.REVOLUTE)

                    axis_str = joint_elem.get("axis", "0 0 1")
                    axis_parts = [float(v) for v in axis_str.split()]
                    axis: tuple[float, float, float] = (
                        axis_parts[0],
                        axis_parts[1],
                        axis_parts[2] if len(axis_parts) > 2 else 0.0,
                    )

                    joint = Joint(
                        name=joint_name,
                        joint_type=joint_type,
                        parent=parent_name,
                        child=body_name,
                        origin=Origin(xyz=pos),
                        axis=axis,
                    )
                    joints.append(joint)
                else:
                    # Fixed joint
                    joint = Joint(
                        name=f"{parent_name}_to_{body_name}",
                        joint_type=JointType.FIXED,
                        parent=parent_name,
                        child=body_name,
                        origin=Origin(xyz=pos),
                    )
                    joints.append(joint)

            # Recurse into children
            self._parse_mjcf_body(body_elem, body_name, links, joints)
