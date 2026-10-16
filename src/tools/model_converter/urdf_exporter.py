"""URDF model exporter for canonical golfer specification.

Generates a valid, Pinocchio- and Drake-compatible URDF from CanonicalModel,
preserving all required invariants:
- Pelvis floating base root link
- Decomposed multi-DOF joints (universal -> 2 revolute; gimbal -> 3 revolute)
- mid_hands virtual frame welded to thorax3 at grip centre
- club_shaft welded to mid_hands
- Valid inertial, visual, and collision tags
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET  # noqa: S405  # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml  # build-only

from defusedxml import minidom
from tools.model_converter.schema_validator import (
    CanonicalModel,
    JointDof,
    JointDof as JointDofSpec,
    RootBody,
    Segment,
)

logger = logging.getLogger(__name__)

HEADER_COMMENT = """
  ============================================================================
  golfer.urdf - Pinocchio & Drake golf humanoid with rigid-club attachment
  ============================================================================
  Scope: forward-simulation model of full body + club. The club (shaft + head)
  is welded to a virtual `mid_hands` frame placed at the geometric centre of
  the grip (midpoint of hands in the address pose).
  Floating base / free-flyer is attached at the pelvis root at load time.

  Generated automatically by tools/model_converter/build_models.py from:
  src/engines/physics_engines/pinocchio/models/spec/golfer_canonical.yaml
"""


def _format_floats(values: tuple[float, ...] | list[float]) -> str:
    return " ".join(f"{v:g}" for v in values)


def _add_inertial(
    parent_elem: ET.Element,
    mass: float,
    ixx: float,
    iyy: float,
    izz: float,
    ixy: float = 0.0,
    ixz: float = 0.0,
    iyz: float = 0.0,
) -> None:
    inertial = ET.SubElement(parent_elem, "inertial")
    mass_elem = ET.SubElement(inertial, "mass")
    mass_elem.set("value", f"{mass:g}")
    inertia_elem = ET.SubElement(inertial, "inertia")
    inertia_elem.set("ixx", f"{ixx:g}")
    inertia_elem.set("ixy", f"{ixy:g}")
    inertia_elem.set("ixz", f"{ixz:g}")
    inertia_elem.set("iyy", f"{iyy:g}")
    inertia_elem.set("iyz", f"{iyz:g}")
    inertia_elem.set("izz", f"{izz:g}")


def _add_visual(
    parent_elem: ET.Element,
    geom_type: str,
    size: tuple[float, ...],
    rgba: tuple[float, float, float, float],
    mat_name: str,
) -> None:
    visual = ET.SubElement(parent_elem, "visual")
    origin = ET.SubElement(visual, "origin")
    origin.set("xyz", "0.0 0.0 0.0")
    origin.set("rpy", "0.0 0.0 0.0")

    geom = ET.SubElement(visual, "geometry")
    if geom_type == "capsule":
        # URDF has no native capsule, cylinder is standard substitute
        cyl = ET.SubElement(geom, "cylinder")
        radius = size[0] if len(size) > 0 else 0.05
        length = size[1] if len(size) > 1 else 0.1
        cyl.set("radius", f"{radius:g}")
        cyl.set("length", f"{length:g}")
    elif geom_type == "cylinder":
        cyl = ET.SubElement(geom, "cylinder")
        radius = size[0] if len(size) > 0 else 0.05
        length = size[1] if len(size) > 1 else 0.1
        cyl.set("radius", f"{radius:g}")
        cyl.set("length", f"{length:g}")
    elif geom_type == "box":
        box = ET.SubElement(geom, "box")
        s = size if len(size) >= 3 else (0.1, 0.1, 0.1)
        box.set("size", f"{s[0]:g} {s[1]:g} {s[2]:g}")
    elif geom_type == "sphere":
        sph = ET.SubElement(geom, "sphere")
        sph.set("radius", f"{size[0]:g}" if len(size) > 0 else "0.1")
    else:
        cyl = ET.SubElement(geom, "cylinder")
        cyl.set("radius", "0.05")
        cyl.set("length", "0.1")

    mat = ET.SubElement(visual, "material")
    mat.set("name", mat_name)
    color = ET.SubElement(mat, "color")
    color.set("rgba", _format_floats(rgba))


def _add_joint_element(
    robot: ET.Element,
    *,
    name: str,
    joint_type: str,
    parent: str,
    child: str,
    origin_xyz: tuple[float, float, float] | str,
    origin_rpy: tuple[float, float, float] | str,
) -> ET.Element:
    """Create a base URDF joint element with parent, child, and origin configured.

    Preconditions:
        - robot is a valid ElementTree Element
        - name, joint_type, parent, and child are non-empty strings
    Postconditions:
        - Joint element appended to robot with parent, child, and origin tags
    """
    joint = ET.SubElement(robot, "joint")
    joint.set("name", name)
    joint.set("type", joint_type)

    p_elem = ET.SubElement(joint, "parent")
    p_elem.set("link", parent)

    c_elem = ET.SubElement(joint, "child")
    c_elem.set("link", child)

    orig = ET.SubElement(joint, "origin")
    orig.set(
        "xyz",
        origin_xyz if isinstance(origin_xyz, str) else _format_floats(origin_xyz),
    )
    orig.set(
        "rpy",
        origin_rpy if isinstance(origin_rpy, str) else _format_floats(origin_rpy),
    )
    return joint


@dataclass(frozen=True)
class RevoluteJointSpec:
    """Parameters of one URDF revolute joint (parameter object).

    Preconditions (checked on construction):
        - ``name``, ``parent`` and ``child`` are non-empty strings
        - ``damping`` >= 0.0
    """

    name: str
    parent: str
    child: str
    origin_xyz: tuple[float, float, float] | str
    origin_rpy: tuple[float, float, float] | str
    dof: JointDofSpec
    damping: float
    effort: float = 1000.0
    velocity: float = 10.0

    def __post_init__(self) -> None:
        for field_name in ("name", "parent", "child"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"RevoluteJointSpec.{field_name} must be a non-empty string"
                )
        if self.damping < 0.0:
            raise ValueError(
                f"RevoluteJointSpec.damping must be >= 0.0, got {self.damping}"
            )


def _format_limit(value: float) -> str:
    """Format an effort/velocity limit (one decimal when integral)."""
    return f"{value:.1f}" if value == int(value) else f"{value:g}"


def _add_revolute_joint(robot: ET.Element, spec: RevoluteJointSpec) -> ET.Element:
    """Add a revolute joint to the URDF robot element.

    Preconditions:
        - robot is a valid ElementTree Element
        - spec is a valid RevoluteJointSpec
    Postconditions:
        - Revolute joint element appended to robot with origin, axis, limit, and dynamics
    """
    joint = _add_joint_element(
        robot,
        name=spec.name,
        joint_type="revolute",
        parent=spec.parent,
        child=spec.child,
        origin_xyz=spec.origin_xyz,
        origin_rpy=spec.origin_rpy,
    )

    ax = ET.SubElement(joint, "axis")
    ax.set("xyz", _format_floats(spec.dof.axis))

    lim = ET.SubElement(joint, "limit")
    lim.set("lower", f"{spec.dof.limits[0]:g}")
    lim.set("upper", f"{spec.dof.limits[1]:g}")
    lim.set("effort", _format_limit(spec.effort))
    lim.set("velocity", _format_limit(spec.velocity))

    dyn = ET.SubElement(joint, "dynamics")
    dyn.set("damping", f"{spec.damping:g}")

    return joint


def _add_dummy_link(robot: ET.Element, name: str) -> ET.Element:
    """Add an intermediate dummy link with minimal mass/inertia to URDF robot.

    Preconditions:
        - robot is a valid ElementTree Element
        - name is a non-empty string
    Postconditions:
        - Dummy link element appended to robot with minimal mass and inertia
    """
    link = ET.SubElement(robot, "link")
    link.set("name", name)
    _add_inertial(link, 0.001, 0.0001, 0.0001, 0.0001)
    return link


def _add_body_link(robot: ET.Element, body: RootBody | Segment) -> ET.Element:
    """Add a link with inertial and visual elements to the URDF robot.

    Preconditions:
        - robot is a valid ElementTree Element
        - body has name, mass, inertia, and geometry specifications
    Postconditions:
        - Link element appended to robot with inertial and visual sub-elements
    """
    link = ET.SubElement(robot, "link")
    link.set("name", body.name)
    _add_inertial(
        link,
        body.mass,
        body.inertia.ixx,
        body.inertia.iyy,
        body.inertia.izz,
        body.inertia.ixy,
        body.inertia.ixz,
        body.inertia.iyz,
    )
    _add_visual(
        link,
        body.geometry.geom_type,
        body.geometry.size,
        body.geometry.visual_rgba,
        f"mat_{body.name}",
    )
    return link


def _add_fixed_joint(
    robot: ET.Element,
    *,
    name: str,
    parent: str,
    child: str,
    origin_xyz: tuple[float, float, float] | str,
    origin_rpy: tuple[float, float, float] | str,
) -> ET.Element:
    """Add a fixed joint to the URDF robot element.

    Preconditions:
        - robot is a valid ElementTree Element
        - name, parent, and child are non-empty strings
    Postconditions:
        - Fixed joint element appended to robot with origin
    """
    return _add_joint_element(
        robot,
        name=name,
        joint_type="fixed",
        parent=parent,
        child=child,
        origin_xyz=origin_xyz,
        origin_rpy=origin_rpy,
    )


def export_urdf(model: CanonicalModel, out_path: Path | None = None) -> str:
    """Generate URDF XML string from canonical model and optionally save to disk.

    Preconditions:
        - model is a valid CanonicalModel instance
    Postconditions:
        - Returns valid URDF XML string conforming to schema and physics contracts
    """
    robot = ET.Element("robot")
    robot.set("name", "golfer")

    # Pelvis root link
    _add_body_link(robot, model.root)

    # Process segments in order
    club_segments = {"club_shaft", "club_head"}

    for seg in model.segments:
        if seg.name in club_segments:
            continue  # Handled separately to weld to mid_hands

        _export_segment(robot, seg)

    # mid_hands virtual frame welded to thorax3
    mid_hands_link = ET.SubElement(robot, "link")
    mid_hands_link.set("name", "mid_hands")

    _add_fixed_joint(
        robot,
        name="thorax3_to_mid_hands",
        parent="thorax3",
        child="mid_hands",
        origin_xyz="0.0 0.0 -0.17",
        origin_rpy="0.0 0.0 0.0",
    )

    # club_shaft link + joint welded to mid_hands
    shaft_seg = model.get_segment("club_shaft")
    if shaft_seg:
        _add_body_link(robot, shaft_seg)
        _add_fixed_joint(
            robot,
            name="mid_hands_to_club_shaft",
            parent="mid_hands",
            child=shaft_seg.name,
            origin_xyz="0.0 0.0 -0.05",
            origin_rpy="0.0 0.0 0.0",
        )

    # club_head link + joint welded to club_shaft
    head_seg = model.get_segment("club_head")
    if head_seg:
        _add_body_link(robot, head_seg)
        _add_fixed_joint(
            robot,
            name="club_shaft_to_club_head",
            parent="club_shaft",
            child=head_seg.name,
            origin_xyz=head_seg.origin.xyz,
            origin_rpy=head_seg.origin.rpy,
        )

    # Convert to pretty XML string
    raw_xml = ET.tostring(robot, encoding="utf-8")
    dom = minidom.parseString(raw_xml)
    pretty_xml = dom.toprettyxml(indent="  ")

    # Inject header comment
    lines = pretty_xml.split("\n")
    # First line is <?xml ...?>
    final_lines = [lines[0], f"<!--{HEADER_COMMENT}-->"] + lines[1:]
    xml_content = "\n".join(final_lines)

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(xml_content, encoding="utf-8")
        logger.info("Wrote canonical URDF to %s", out_path)

    return xml_content


def _add_chain_joint(
    robot: ET.Element,
    seg: Segment,
    parent: str,
    child: str,
    dof_index: int,
) -> None:
    """Add one revolute joint of ``seg``'s (possibly decomposed) joint chain.

    The first joint of a chain (``parent == seg.parent``) carries the segment
    origin; intermediate joints sit at the zero offset.
    """
    first = parent == seg.parent
    _add_revolute_joint(
        robot,
        RevoluteJointSpec(
            name=f"{parent}_to_{child}",
            parent=parent,
            child=child,
            origin_xyz=seg.origin.xyz if first else "0.0 0.0 0.0",
            origin_rpy=seg.origin.rpy if first else "0.0 0.0 0.0",
            dof=seg.joint.dofs[dof_index],
            damping=seg.joint.damping,
        ),
    )


def _export_segment(robot: ET.Element, seg: Segment) -> None:
    """Export a standard segment and its joint(s) to URDF.

    Preconditions:
        - robot is a valid ElementTree Element
        - seg is a valid Segment specification
    Postconditions:
        - Joint(s), intermediate links (if multi-DOF), and segment link appended to robot
    """
    jt = seg.joint.joint_type

    if jt == "revolute":
        _add_chain_joint(robot, seg, seg.parent, seg.name, 0)
    elif jt == "universal":
        # Decompose into 2 revolute joints with 1 intermediate dummy link
        inter_name = f"{seg.name}_intermediate"
        _add_chain_joint(robot, seg, seg.parent, inter_name, 0)
        _add_dummy_link(robot, inter_name)
        _add_chain_joint(robot, seg, inter_name, seg.name, 1)
    elif jt == "gimbal":
        # Decompose into 3 revolute joints with 2 intermediate dummy links
        gz_name = f"{seg.name}_gimbal_z"
        gy_name = f"{seg.name}_gimbal_y"
        _add_chain_joint(robot, seg, seg.parent, gz_name, 0)
        _add_dummy_link(robot, gz_name)
        _add_chain_joint(robot, seg, gz_name, gy_name, 1)
        _add_dummy_link(robot, gy_name)
        _add_chain_joint(robot, seg, gy_name, seg.name, 2)
    elif jt == "fixed":
        _add_fixed_joint(
            robot,
            name=f"{seg.parent}_to_{seg.name}",
            parent=seg.parent,
            child=seg.name,
            origin_xyz=seg.origin.xyz,
            origin_rpy=seg.origin.rpy,
        )

    _add_body_link(robot, seg)
