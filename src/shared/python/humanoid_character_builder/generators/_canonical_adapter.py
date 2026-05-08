"""Adapter from humanoid-domain types to model_generation canonical types.

Per ADR 0007 (Option B - Layer), the canonical URDF XML emitter is
``model_generation.builders.urdf_writer.URDFWriter``. The humanoid
character builder's domain types (``GeneratedLink``, ``GeneratedJoint``)
must be converted to ``model_generation.core.types`` (``Link``, ``Joint``)
before they can be passed to the canonical writer.

This module provides the conversion. It is the structural foundation for
issue #4601 (extract canonical URDF emitter): once consumers route through
``write_humanoid_urdf_via_canonical(...)``, the legacy
``humanoid_character_builder.generators._urdf_xml_writer.URDFXMLWriter``
can be deprecated.

The conversion is intentionally lossy in the same direction as URDF
itself (e.g. visual+collision geometry collapse to single dict);
round-tripping is not a goal of this adapter.

Tested in ``tests/unit/tools/humanoid_character_builder/test_canonical_adapter.py``.
"""

from __future__ import annotations

from typing import Any

from humanoid_character_builder.core.model import GeneratedJoint, GeneratedLink
from model_generation.core.types import (
    Geometry,
    GeometryType,
    Inertia,
    Joint,
    JointDynamics,
    JointLimits,
    JointType,
    Link,
    Origin,
)

#: HCB joint-type strings → canonical model_generation JointType members.
_JOINT_TYPE_MAP: dict[str, JointType] = {
    "revolute": JointType.REVOLUTE,
    "prismatic": JointType.PRISMATIC,
    "fixed": JointType.FIXED,
    "continuous": JointType.CONTINUOUS,
    "floating": JointType.FLOATING,
    "planar": JointType.PLANAR,
}


def _geometry_dict_to_canonical(geom: dict[str, Any] | None) -> Geometry | None:
    """Convert HCB's free-form geometry dict to a canonical Geometry."""
    if geom is None:
        return None
    gtype = geom.get("type", "").lower()
    if gtype == "box":
        sx, sy, sz = geom.get("size", (0.1, 0.1, 0.1))
        return Geometry.box(sx, sy, sz)
    if gtype == "cylinder":
        return Geometry.cylinder(geom.get("radius", 0.05), geom.get("length", 0.1))
    if gtype == "sphere":
        return Geometry.sphere(geom.get("radius", 0.05))
    if gtype == "capsule":
        return Geometry.capsule(geom.get("radius", 0.05), geom.get("length", 0.1))
    if gtype == "mesh":
        # Geometry.mesh signature is (filename, scale)
        return Geometry.mesh(geom.get("filename", ""), geom.get("scale", (1, 1, 1)))
    # Unknown geometry type — fall back to a tiny box so URDFWriter still
    # produces output. Matches the legacy emitter's lenience.
    return Geometry(geometry_type=GeometryType.BOX, size=(0.01, 0.01, 0.01))


def _generated_link_to_canonical(g: GeneratedLink) -> Link:
    """Convert a humanoid GeneratedLink to a canonical Link."""
    inertia = Inertia(
        ixx=g.inertia.ixx,
        iyy=g.inertia.iyy,
        izz=g.inertia.izz,
        ixy=g.inertia.ixy,
        ixz=g.inertia.ixz,
        iyz=g.inertia.iyz,
        mass=g.mass,
        center_of_mass=g.inertia.center_of_mass,
    )
    visual = _geometry_dict_to_canonical(g.visual_geometry)
    collision = _geometry_dict_to_canonical(g.collision_geometry)
    origin = Origin(xyz=g.origin_xyz, rpy=g.origin_rpy)
    # MG Link uses separate visual_origin / collision_origin fields; the
    # HCB GeneratedLink combines them into one origin (URDF allows them to
    # differ but HCB never produces different ones), so we use the same
    # origin for both.
    return Link(
        name=g.name,
        inertia=inertia,
        visual_geometry=visual,
        visual_origin=origin,
        collision_geometry=collision,
        collision_origin=origin,
    )


def _generated_joint_to_canonical(g: GeneratedJoint) -> Joint:
    """Convert a humanoid GeneratedJoint to a canonical Joint."""
    jtype = _JOINT_TYPE_MAP.get(g.joint_type.lower(), JointType.REVOLUTE)
    limits: JointLimits | None = None
    if g.limits:
        limits = JointLimits(
            lower=g.limits.get("lower", 0.0),
            upper=g.limits.get("upper", 0.0),
            effort=g.limits.get("effort", 0.0),
            velocity=g.limits.get("velocity", 0.0),
        )
    dynamics = JointDynamics(
        damping=g.dynamics.get("damping", 0.0) if g.dynamics else 0.0,
        friction=g.dynamics.get("friction", 0.0) if g.dynamics else 0.0,
    )
    return Joint(
        name=g.name,
        joint_type=jtype,
        parent=g.parent,
        child=g.child,
        origin=Origin(xyz=g.origin_xyz, rpy=g.origin_rpy),
        axis=g.axis,
        limits=limits,
        dynamics=dynamics,
    )


def to_canonical_lists(
    links: dict[str, GeneratedLink],
    joints: list[GeneratedJoint],
) -> tuple[list[Link], list[Joint]]:
    """Convert HCB link/joint collections to canonical Link[]/Joint[].

    Args:
        links: HCB link mapping (name -> GeneratedLink).
        joints: HCB joint list.

    Returns:
        ``(canonical_links, canonical_joints)`` ready to pass to
        ``model_generation.builders.urdf_writer.URDFWriter.write(...)``.
    """
    return (
        [_generated_link_to_canonical(g) for g in links.values()],
        [_generated_joint_to_canonical(g) for g in joints],
    )


def write_humanoid_urdf_via_canonical(
    robot_name: str,
    links: dict[str, GeneratedLink],
    joints: list[GeneratedJoint],
    materials: dict[str, Any] | None = None,
    pretty_print: bool = True,
) -> str:
    """Write a humanoid URDF using the canonical model_generation writer.

    This is the structural foundation for #4601 (canonical URDF emitter).
    It converts HCB domain types to canonical types and delegates emission
    to ``model_generation.builders.urdf_writer.URDFWriter``.

    Behavior is **not yet byte-identical** to the legacy
    ``humanoid_character_builder.generators._urdf_xml_writer.URDFXMLWriter``;
    a follow-up issue will measure and align the diff before flipping the
    default. Until then, callers that need byte-identical legacy output
    should keep using the legacy writer.

    Args:
        robot_name: Name attribute on ``<robot>``.
        links: HCB link mapping.
        joints: HCB joint list.
        materials: Optional materials dict (passed through to canonical writer).
        pretty_print: If True, indent the output.

    Returns:
        URDF XML string.
    """
    from model_generation.builders.urdf_writer import URDFWriter

    canon_links, canon_joints = to_canonical_lists(links, joints)
    writer = URDFWriter(pretty_print=pretty_print)
    return writer.write(robot_name, canon_links, canon_joints, materials or {})
