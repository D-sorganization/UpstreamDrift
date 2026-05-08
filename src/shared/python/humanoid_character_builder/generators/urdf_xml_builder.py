"""
URDF XML assembly helpers.

Handles constructing the robot XML tree from generated links and joints
and serialising it to a string.

Extracted from urdf_generator.py to isolate XML-serialisation concerns.
"""

from __future__ import annotations

from typing import Any
import xml.etree.ElementTree as ET  # stdlib retained for Element/SubElement

from humanoid_character_builder.core.model import GeneratedJoint, GeneratedLink
from humanoid_character_builder.generators.urdf_geometry import add_geometry_element

from model_generation.builders.urdf_writer import URDFWriter
from model_generation.core.types import (
    Geometry,
    Inertia,
    Joint,
    JointDynamics,
    JointLimits,
    JointType,
    Link,
    Material,
    Origin,
)


def _compute_collision_exclusions(
    links: dict[str, GeneratedLink],
    joints: list[GeneratedJoint],
) -> list[tuple[str, str]]:
    """Compute collision exclusion pairs for adjacent segments.

    Builds a parent-child map from joints and excludes collisions between
    direct parent-child pairs (connected by joints).

    Args:
        links: Mapping of link-name -> GeneratedLink.
        joints: Ordered list of GeneratedJoint instances.

    Returns:
        List of (link1, link2) tuples representing excluded collision pairs.
    """
    if links is None:
        raise ValueError("links must be provided")
    if joints is None:
        raise ValueError("joints must be provided")
    exclusions: set[tuple[str, str]] = set()

    # Build parent-child relationships and exclude direct parent-child pairs
    for joint in joints:
        pair = tuple(sorted((joint.parent, joint.child)))
        exclusions.add(pair)

    return list(exclusions)


def _convert_geometry(geom: dict[str, Any]) -> Geometry:
    gtype = geom.get("type")
    if gtype == "box":
        return Geometry.box(*geom["size"])
    if gtype == "cylinder":
        return Geometry.cylinder(geom["radius"], geom["length"])
    if gtype == "sphere":
        return Geometry.sphere(geom["radius"])
    if gtype == "capsule":
        return Geometry.capsule(geom["radius"], geom["length"])
    if gtype == "mesh":
        scale = geom.get("scale", (1.0, 1.0, 1.0))
        return Geometry.mesh(geom["filename"], scale)
    return Geometry.box(0.1, 0.1, 0.1)


def build_urdf_xml(
    robot_name: str,
    links: dict[str, GeneratedLink],
    joints: list[GeneratedJoint],
    materials: dict[str, tuple[float, float, float, float]],
    pretty_print: bool = True,
    indent: str = "  ",
    add_collision_exclusions: bool = True,
) -> str:
    """Build the complete URDF XML string.

    Args:
        robot_name: Name attribute for the <robot> element.
        links: Mapping of link-name -> GeneratedLink.
        joints: Ordered list of GeneratedJoint instances.
        materials: Mapping of material-name -> RGBA tuple.
        pretty_print: If True, apply ET.indent for human-readable output.
        indent: Indentation string used when pretty_print is True.
        add_collision_exclusions: If True, add <gazebo> disable_collisions for adjacent links.

    Returns:
        URDF XML as a unicode string (no XML declaration header).
    """
    # Convert GeneratedLink to model_generation Link
    canonical_links = []
    for link_data in links.values():
        canonical_link = Link(
            name=link_data.name,
            inertia=Inertia(
                ixx=link_data.inertia.ixx,
                iyy=link_data.inertia.iyy,
                izz=link_data.inertia.izz,
                ixy=link_data.inertia.ixy,
                ixz=link_data.inertia.ixz,
                iyz=link_data.inertia.iyz,
                mass=link_data.mass,
                center_of_mass=link_data.origin_xyz,
            ),
        )
        if link_data.visual_geometry:
            canonical_link.visual_geometry = _convert_geometry(
                link_data.visual_geometry
            )
            canonical_link.visual_material = Material(name="skin")
        if link_data.collision_geometry:
            canonical_link.collision_geometry = _convert_geometry(
                link_data.collision_geometry
            )
        canonical_links.append(canonical_link)

    # Convert GeneratedJoint to model_generation Joint
    canonical_joints = []
    for joint_data in joints:
        try:
            jtype = JointType(joint_data.joint_type)
        except ValueError:
            jtype = JointType.FIXED

        canonical_joint = Joint(
            name=joint_data.name,
            joint_type=jtype,
            parent=joint_data.parent,
            child=joint_data.child,
            origin=Origin(xyz=joint_data.origin_xyz, rpy=joint_data.origin_rpy),
            axis=joint_data.axis,
        )
        if joint_data.limits and joint_data.joint_type in ("revolute", "prismatic"):
            canonical_joint.limits = JointLimits(
                lower=joint_data.limits["lower"],
                upper=joint_data.limits["upper"],
                effort=joint_data.limits["effort"],
                velocity=joint_data.limits["velocity"],
            )
        if joint_data.dynamics:
            canonical_joint.dynamics = JointDynamics(
                damping=joint_data.dynamics.get("damping", 0.0),
                friction=joint_data.dynamics.get("friction", 0.0),
            )
        canonical_joints.append(canonical_joint)

    # Convert materials
    canonical_materials = {
        name: Material(name=name, color=rgba) for name, rgba in materials.items()
    }

    writer = URDFWriter(
        pretty_print=pretty_print,
        indent=indent,
        add_collision_exclusions=add_collision_exclusions,
    )

    xml_str = writer.write(
        robot_name=robot_name,
        links=canonical_links,
        joints=canonical_joints,
        materials=canonical_materials,
    )

    # Strip XML declaration if present to match backward compatibility
    if xml_str.startswith("<?xml"):
        xml_str = xml_str.split("?>", 1)[1].strip()

    return xml_str


def _add_link_element(root: ET.Element, link: GeneratedLink) -> None:
    """Add a <link> element to the robot XML root."""
    if root is None:
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
    if root is None:
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
