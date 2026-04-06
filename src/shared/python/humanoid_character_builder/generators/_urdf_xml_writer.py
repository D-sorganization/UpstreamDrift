"""Internal helpers for URDF XML serialization."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Any

from humanoid_character_builder.core.model import GeneratedJoint, GeneratedLink

if TYPE_CHECKING:
    from humanoid_character_builder.generators.urdf_generator import URDFGeneratorConfig


class URDFXMLWriter:
    """Serialize generated humanoid model data into URDF XML."""

    def __init__(self, config: URDFGeneratorConfig) -> None:
        self.config = config

    def build_urdf_xml(
        self,
        robot_name: str,
        materials: dict[str, tuple[float, float, float, float]],
        links: dict[str, GeneratedLink],
        joints: list[GeneratedJoint],
    ) -> str:
        """Build the complete URDF XML document."""
        if not (robot_name is not None):
            raise ValueError("robot_name must be provided")
        root = ET.Element("robot", name=robot_name)

        for material_name, rgba in materials.items():
            material = ET.SubElement(root, "material", name=material_name)
            ET.SubElement(
                material,
                "color",
                rgba=f"{rgba[0]:.4f} {rgba[1]:.4f} {rgba[2]:.4f} {rgba[3]:.4f}",
            )

        for link in links.values():
            self.add_link_element(root, link)

        for joint in joints:
            self.add_joint_element(root, joint)

        if self.config.pretty_print:
            ET.indent(root, space=self.config.indent)
        return ET.tostring(root, encoding="unicode")

    def add_link_element(self, root: ET.Element, link: GeneratedLink) -> None:
        """Append a URDF link element."""
        if not (root is not None):
            raise ValueError("root must be provided")
        link_elem = ET.SubElement(root, "link", name=link.name)

        inertial = ET.SubElement(link_elem, "inertial")
        ET.SubElement(
            inertial,
            "origin",
            xyz=f"{link.origin_xyz[0]:.6f} {link.origin_xyz[1]:.6f} {link.origin_xyz[2]:.6f}",
            rpy=f"{link.origin_rpy[0]:.6f} {link.origin_rpy[1]:.6f} {link.origin_rpy[2]:.6f}",
        )
        ET.SubElement(inertial, "mass", value=f"{link.mass:.6f}")
        ET.SubElement(
            inertial,
            "inertia",
            ixx=f"{link.inertia.ixx:.8f}",
            ixy=f"{link.inertia.ixy:.8f}",
            ixz=f"{link.inertia.ixz:.8f}",
            iyy=f"{link.inertia.iyy:.8f}",
            iyz=f"{link.inertia.iyz:.8f}",
            izz=f"{link.inertia.izz:.8f}",
        )

        visual = ET.SubElement(link_elem, "visual")
        ET.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")
        self.add_geometry_element(visual, link.visual_geometry)
        ET.SubElement(visual, "material", name="skin")

        if link.collision_geometry:
            collision = ET.SubElement(link_elem, "collision")
            ET.SubElement(collision, "origin", xyz="0 0 0", rpy="0 0 0")
            self.add_geometry_element(collision, link.collision_geometry)

    def add_geometry_element(self, parent: ET.Element, geom: dict[str, Any]) -> None:
        """Append a URDF geometry element."""
        if not (parent is not None):
            raise ValueError("parent must be provided")
        geometry = ET.SubElement(parent, "geometry")

        geom_type = geom["type"]
        if geom_type == "box":
            size = geom["size"]
            ET.SubElement(
                geometry,
                "box",
                size=f"{size[0]:.6f} {size[1]:.6f} {size[2]:.6f}",
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

    def add_joint_element(self, root: ET.Element, joint: GeneratedJoint) -> None:
        """Append a URDF joint element."""
        if not (root is not None):
            raise ValueError("root must be provided")
        joint_elem = ET.SubElement(
            root, "joint", name=joint.name, type=joint.joint_type
        )
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
