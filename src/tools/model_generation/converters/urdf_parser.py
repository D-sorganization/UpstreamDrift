"""
URDF parser for loading and editing existing URDF files.

This module provides comprehensive parsing of URDF files into
editable data structures.
"""

from __future__ import annotations

import logging
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

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
class ParsedModel:
    """Result of parsing a URDF file."""

    # Robot name
    name: str

    # Parsed links
    links: list[Link] = field(default_factory=list)

    # Parsed joints
    joints: list[Joint] = field(default_factory=list)

    # Material definitions
    materials: dict[str, Material] = field(default_factory=dict)

    # Original XML (for text editing)
    original_xml: str | None = None

    # Source file path
    source_path: Path | None = None

    # Parse warnings/errors
    warnings: list[str] = field(default_factory=list)

    # Whether file is read-only (from library)
    read_only: bool = False

    def get_link(self, name: str) -> Link | None:
        """Get link by name."""
        for link in self.links:
            if link.name == name:
                return link
        return None

    def get_joint(self, name: str) -> Joint | None:
        """Get joint by name."""
        for joint in self.joints:
            if joint.name == name:
                return joint
        return None

    def get_root_link(self) -> Link | None:
        """Get the root (base) link."""
        child_names = {j.child for j in self.joints}
        for link in self.links:
            if link.name not in child_names:
                return link
        return self.links[0] if self.links else None

    def get_children(self, link_name: str) -> list[str]:
        """Get child link names."""
        return [j.child for j in self.joints if j.parent == link_name]

    def get_parent(self, link_name: str) -> str | None:
        """Get parent link name."""
        for j in self.joints:
            if j.child == link_name:
                return j.parent
        return None

    def get_subtree(self, link_name: str) -> list[str]:
        """Get all links in subtree rooted at link_name."""
        result = [link_name]
        queue = [link_name]
        while queue:
            current = queue.pop(0)
            children = self.get_children(current)
            result.extend(children)
            queue.extend(children)
        return result

    def to_urdf(self, pretty_print: bool = True) -> str:
        """Convert back to URDF XML."""
        from model_generation.builders.urdf_writer import URDFWriter

        writer = URDFWriter(pretty_print=pretty_print)
        return writer.write(self.name, self.links, self.joints, self.materials)

    def copy(self) -> ParsedModel:
        """Create a deep copy."""
        return ParsedModel(
            name=self.name,
            links=[Link.from_dict(l.to_dict()) for l in self.links],
            joints=[Joint.from_dict(j.to_dict()) for j in self.joints],
            materials={
                k: Material.from_dict(v.to_dict()) for k, v in self.materials.items()
            },
            original_xml=self.original_xml,
            source_path=self.source_path,
            warnings=self.warnings.copy(),
            read_only=False,  # Copy is editable
        )


class URDFParser:
    """
    Parse URDF files into editable data structures.

    Features:
    - Full URDF parsing (links, joints, materials)
    - Mesh path resolution
    - Validation during parsing
    - Preserves original XML for text editing
    """

    def __init__(self, resolve_meshes: bool = True):
        """
        Initialize parser.

        Args:
            resolve_meshes: If True, attempt to resolve mesh file paths
        """
        self.resolve_meshes = resolve_meshes

    def parse(
        self,
        source: str | Path,
        read_only: bool = False,
    ) -> ParsedModel:
        """
        Parse a URDF file.

        Args:
            source: Path to URDF file or XML string
            read_only: If True, mark model as read-only

        Returns:
            ParsedModel with parsed contents
        """
        # Determine if source is file or string
        source_path = None
        if isinstance(source, Path) or (
            isinstance(source, str) and not source.strip().startswith("<")
        ):
            source_path = Path(source)
            if not source_path.exists():
                raise FileNotFoundError(f"URDF file not found: {source_path}")
            xml_string = source_path.read_text()
        else:
            xml_string = source

        # Parse XML
        try:
            root = ET.fromstring(xml_string)
        except ET.ParseError as e:
            raise ValueError(f"Invalid URDF XML: {e}")

        if root.tag != "robot":
            raise ValueError(f"Expected 'robot' root element, got '{root.tag}'")

        robot_name = root.get("name", "unnamed_robot")

        # Parse materials first
        materials = {}
        for mat_elem in root.findall("material"):
            material = self._parse_material(mat_elem)
            if material:
                materials[material.name] = material

        # Parse links
        links = []
        warnings = []
        for link_elem in root.findall("link"):
            try:
                link = self._parse_link(link_elem, materials, source_path)
                links.append(link)
            except Exception as e:
                warnings.append(f"Failed to parse link: {e}")

        # Parse joints
        joints = []
        for joint_elem in root.findall("joint"):
            try:
                joint = self._parse_joint(joint_elem)
                joints.append(joint)
            except Exception as e:
                warnings.append(f"Failed to parse joint: {e}")

        return ParsedModel(
            name=robot_name,
            links=links,
            joints=joints,
            materials=materials,
            original_xml=xml_string,
            source_path=source_path,
            warnings=warnings,
            read_only=read_only,
        )

    def parse_string(self, xml_string: str, read_only: bool = False) -> ParsedModel:
        """Parse URDF from XML string."""
        return self.parse(xml_string, read_only=read_only)

    def _parse_link(
        self,
        elem: ET.Element,
        materials: dict[str, Material],
        base_path: Path | None,
    ) -> Link:
        """Parse a link element."""
        name = elem.get("name")
        if not name:
            raise ValueError("Link missing 'name' attribute")

        # Parse inertial
        inertia = Inertia(ixx=0.1, iyy=0.1, izz=0.1, mass=1.0)
        inertial_elem = elem.find("inertial")
        if inertial_elem is not None:
            inertia = self._parse_inertial(inertial_elem)

        # Parse visual
        visual_geometry = None
        visual_origin = Origin()
        visual_material = None

        visual_elem = elem.find("visual")
        if visual_elem is not None:
            origin_elem = visual_elem.find("origin")
            if origin_elem is not None:
                visual_origin = self._parse_origin(origin_elem)

            geom_elem = visual_elem.find("geometry")
            if geom_elem is not None:
                visual_geometry = self._parse_geometry(geom_elem, base_path)

            mat_elem = visual_elem.find("material")
            if mat_elem is not None:
                mat_name = mat_elem.get("name")
                if mat_name and mat_name in materials:
                    visual_material = materials[mat_name]
                else:
                    visual_material = self._parse_material(mat_elem)

        # Parse collision
        collision_geometry = None
        collision_origin = Origin()

        collision_elem = elem.find("collision")
        if collision_elem is not None:
            origin_elem = collision_elem.find("origin")
            if origin_elem is not None:
                collision_origin = self._parse_origin(origin_elem)

            geom_elem = collision_elem.find("geometry")
            if geom_elem is not None:
                collision_geometry = self._parse_geometry(geom_elem, base_path)

        return Link(
            name=name,
            inertia=inertia,
            visual_geometry=visual_geometry,
            visual_origin=visual_origin,
            visual_material=visual_material,
            collision_geometry=collision_geometry,
            collision_origin=collision_origin,
        )

    def _parse_joint(self, elem: ET.Element) -> Joint:
        """Parse a joint element."""
        name = elem.get("name")
        if not name:
            raise ValueError("Joint missing 'name' attribute")

        joint_type_str = elem.get("type", "fixed")
        try:
            joint_type = JointType(joint_type_str)
        except ValueError:
            logger.warning(f"Unknown joint type '{joint_type_str}', using fixed")
            joint_type = JointType.FIXED

        # Parent and child
        parent_elem = elem.find("parent")
        child_elem = elem.find("child")
        if parent_elem is None or child_elem is None:
            raise ValueError(f"Joint '{name}' missing parent or child")

        parent = parent_elem.get("link", "")
        child = child_elem.get("link", "")

        # Origin
        origin = Origin()
        origin_elem = elem.find("origin")
        if origin_elem is not None:
            origin = self._parse_origin(origin_elem)

        # Axis
        axis = (0.0, 0.0, 1.0)
        axis_elem = elem.find("axis")
        if axis_elem is not None:
            xyz_str = axis_elem.get("xyz", "0 0 1")
            axis = tuple(float(v) for v in xyz_str.split())

        # Limits
        limits = None
        limit_elem = elem.find("limit")
        if limit_elem is not None:
            limits = JointLimits(
                lower=float(limit_elem.get("lower", -math.pi)),
                upper=float(limit_elem.get("upper", math.pi)),
                effort=float(limit_elem.get("effort", 1000)),
                velocity=float(limit_elem.get("velocity", 10)),
            )

        # Dynamics
        dynamics = JointDynamics()
        dynamics_elem = elem.find("dynamics")
        if dynamics_elem is not None:
            dynamics = JointDynamics(
                damping=float(dynamics_elem.get("damping", 0.5)),
                friction=float(dynamics_elem.get("friction", 0.0)),
            )

        return Joint(
            name=name,
            joint_type=joint_type,
            parent=parent,
            child=child,
            origin=origin,
            axis=axis,
            limits=limits,
            dynamics=dynamics,
        )

    def _parse_inertial(self, elem: ET.Element) -> Inertia:
        """Parse inertial element."""
        # Origin (COM)
        com = (0.0, 0.0, 0.0)
        origin_elem = elem.find("origin")
        if origin_elem is not None:
            xyz_str = origin_elem.get("xyz", "0 0 0")
            com = tuple(float(v) for v in xyz_str.split())

        # Mass
        mass = 1.0
        mass_elem = elem.find("mass")
        if mass_elem is not None:
            mass = float(mass_elem.get("value", 1.0))

        # Inertia
        inertia_elem = elem.find("inertia")
        if inertia_elem is not None:
            return Inertia(
                ixx=float(inertia_elem.get("ixx", 0.1)),
                iyy=float(inertia_elem.get("iyy", 0.1)),
                izz=float(inertia_elem.get("izz", 0.1)),
                ixy=float(inertia_elem.get("ixy", 0.0)),
                ixz=float(inertia_elem.get("ixz", 0.0)),
                iyz=float(inertia_elem.get("iyz", 0.0)),
                mass=mass,
                center_of_mass=com,
            )
        else:
            return Inertia(ixx=0.1, iyy=0.1, izz=0.1, mass=mass, center_of_mass=com)

    def _parse_origin(self, elem: ET.Element) -> Origin:
        """Parse origin element."""
        xyz_str = elem.get("xyz", "0 0 0")
        rpy_str = elem.get("rpy", "0 0 0")

        xyz = tuple(float(v) for v in xyz_str.split())
        rpy = tuple(float(v) for v in rpy_str.split())

        return Origin(xyz=xyz, rpy=rpy)

    def _parse_geometry(self, elem: ET.Element, base_path: Path | None) -> Geometry:
        """Parse geometry element."""
        # Box
        box_elem = elem.find("box")
        if box_elem is not None:
            size_str = box_elem.get("size", "0.1 0.1 0.1")
            size = tuple(float(v) for v in size_str.split())
            return Geometry(geometry_type=GeometryType.BOX, dimensions=size)

        # Cylinder
        cylinder_elem = elem.find("cylinder")
        if cylinder_elem is not None:
            radius = float(cylinder_elem.get("radius", 0.05))
            length = float(cylinder_elem.get("length", 0.1))
            return Geometry(
                geometry_type=GeometryType.CYLINDER, dimensions=(radius, length)
            )

        # Sphere
        sphere_elem = elem.find("sphere")
        if sphere_elem is not None:
            radius = float(sphere_elem.get("radius", 0.05))
            return Geometry(geometry_type=GeometryType.SPHERE, dimensions=(radius,))

        # Mesh
        mesh_elem = elem.find("mesh")
        if mesh_elem is not None:
            filename = mesh_elem.get("filename", "")
            scale_str = mesh_elem.get("scale", "1 1 1")
            scale = tuple(float(v) for v in scale_str.split())

            # Resolve mesh path
            if self.resolve_meshes and base_path and filename:
                resolved = self._resolve_mesh_path(filename, base_path)
                if resolved:
                    filename = str(resolved)

            return Geometry(
                geometry_type=GeometryType.MESH,
                mesh_filename=filename,
                mesh_scale=scale,
            )

        # Default
        return Geometry(geometry_type=GeometryType.BOX, dimensions=(0.1, 0.1, 0.1))

    def _parse_material(self, elem: ET.Element) -> Material | None:
        """Parse material element."""
        name = elem.get("name")
        if not name:
            return None

        color = (0.8, 0.8, 0.8, 1.0)
        color_elem = elem.find("color")
        if color_elem is not None:
            rgba_str = color_elem.get("rgba", "0.8 0.8 0.8 1.0")
            color = tuple(float(v) for v in rgba_str.split())

        texture = None
        texture_elem = elem.find("texture")
        if texture_elem is not None:
            texture = texture_elem.get("filename")

        return Material(name=name, color=color, texture=texture)

    def _resolve_mesh_path(self, filename: str, base_path: Path) -> Path | None:
        """Resolve mesh file path."""
        # Handle package:// URLs
        if filename.startswith("package://"):
            # Strip package:// prefix
            package_path = filename[10:]
            # Try to find in common locations
            for search_dir in [base_path.parent, base_path.parent.parent]:
                candidate = search_dir / package_path
                if candidate.exists():
                    return candidate
            return None

        # Handle relative paths
        if not Path(filename).is_absolute():
            candidate = base_path.parent / filename
            if candidate.exists():
                return candidate

        return Path(filename) if Path(filename).exists() else None
