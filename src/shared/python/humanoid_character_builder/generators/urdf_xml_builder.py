"""
URDF XML assembly helpers.

Handles constructing the robot XML tree from generated links and joints
and serialising it to a string.

Extracted from urdf_generator.py to isolate XML-serialisation concerns.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from humanoid_character_builder.core.model import GeneratedJoint, GeneratedLink
from humanoid_character_builder.generators.urdf_geometry import add_geometry_element


def build_urdf_xml(
    robot_name: str,
    links: dict[str, GeneratedLink],
    joints: list[GeneratedJoint],
    materials: dict[str, tuple[float, float, float, float]],
    pretty_print: bool = True,
    indent: str = "  ",
) -> str:
    """Build the complete URDF XML string.

    Args:
        robot_name: Name attribute for the <robot> element.
        links: Mapping of link-name -> GeneratedLink.
        joints: Ordered list of GeneratedJoint instances.
        materials: Mapping of material-name -> RGBA tuple.
        pretty_print: If True, apply ET.indent for human-readable output.
        indent: Indentation string used when pretty_print is True.

    Returns:
        URDF XML as a unicode string (no XML declaration header).
    """
    if not (robot_name is not None):
        raise ValueError("robot_name must be provided")
    root = ET.Element("robot", name=robot_name)

    # Add materials
    for mat_name, rgba in materials.items():
        material = ET.SubElement(root, "material", name=mat_name)
        ET.SubElement(
            material,
            "color",
            rgba=f"{rgba[0]:.4f} {rgba[1]:.4f} {rgba[2]:.4f} {rgba[3]:.4f}",
        )

    # Add links
    for link_data in links.values():
        _add_link_element(root, link_data)

    # Add joints
    for joint_data in joints:
        _add_joint_element(root, joint_data)

    if pretty_print:
        ET.indent(root, space=indent)
    return ET.tostring(root, encoding="unicode")


def _add_link_element(root: ET.Element, link: GeneratedLink) -> None:
    """Add a <link> element to the robot XML root."""
    if not (root is not None):
        raise ValueError("root must be provided")
    link_elem = ET.SubElement(root, "link", name=link.name)

    # Inertial
    inertial = ET.SubElement(link_elem, "inertial")
    ET.SubElement(
        inertial,
        "origin",
        xyz=(
            f"{link.origin_xyz[0]:.6f} "
            f"{link.origin_xyz[1]:.6f} "
            f"{link.origin_xyz[2]:.6f}"
        ),
        rpy=(
            f"{link.origin_rpy[0]:.6f} "
            f"{link.origin_rpy[1]:.6f} "
            f"{link.origin_rpy[2]:.6f}"
        ),
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
    add_geometry_element(visual, link.visual_geometry)
    ET.SubElement(visual, "material", name="skin")

    # Collision
    if link.collision_geometry:
        collision = ET.SubElement(link_elem, "collision")
        ET.SubElement(collision, "origin", xyz="0 0 0", rpy="0 0 0")
        add_geometry_element(collision, link.collision_geometry)


def _add_joint_element(root: ET.Element, joint: GeneratedJoint) -> None:
    """Add a <joint> element to the robot XML root."""
    if not (root is not None):
        raise ValueError("root must be provided")
    joint_elem = ET.SubElement(root, "joint", name=joint.name, type=joint.joint_type)

    ET.SubElement(joint_elem, "parent", link=joint.parent)
    ET.SubElement(joint_elem, "child", link=joint.child)
    ET.SubElement(
        joint_elem,
        "origin",
        xyz=(
            f"{joint.origin_xyz[0]:.6f} "
            f"{joint.origin_xyz[1]:.6f} "
            f"{joint.origin_xyz[2]:.6f}"
        ),
        rpy=(
            f"{joint.origin_rpy[0]:.6f} "
            f"{joint.origin_rpy[1]:.6f} "
            f"{joint.origin_rpy[2]:.6f}"
        ),
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


# Public aliases used by backward-compat shims in urdf_generator
add_link_element = _add_link_element
add_joint_element = _add_joint_element
