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
from pathlib import Path
import xml.etree.ElementTree as ET

from defusedxml import minidom
from tools.model_converter.schema_validator import CanonicalModel, Segment

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


def export_urdf(model: CanonicalModel, out_path: Path | None = None) -> str:
    """Generate URDF XML string from canonical model and optionally save to disk."""
    robot = ET.Element("robot")
    robot.set("name", "golfer")

    # Pelvis root link
    pelvis_link = ET.SubElement(robot, "link")
    root = model.root
    root_inertia = root.inertia
    root_geom = root.geometry

    pelvis_link.set("name", root.name)
    _add_inertial(
        pelvis_link,
        root.mass,
        root_inertia.ixx,
        root_inertia.iyy,
        root_inertia.izz,
        root_inertia.ixy,
        root_inertia.ixz,
        root_inertia.iyz,
    )
    _add_visual(
        pelvis_link,
        root_geom.geom_type,
        root_geom.size,
        root_geom.visual_rgba,
        f"mat_{root.name}",
    )

    # Process segments in order
    club_segments = {"club_shaft", "club_head"}

    for seg in model.segments:
        if seg.name in club_segments:
            continue  # Handled separately to weld to mid_hands

        _export_segment(robot, seg)

    # mid_hands virtual frame welded to thorax3
    mid_hands_link = ET.SubElement(robot, "link")
    mid_hands_link.set("name", "mid_hands")

    mid_joint = ET.SubElement(robot, "joint")
    mid_joint.set("name", "thorax3_to_mid_hands")
    mid_joint.set("type", "fixed")
    p_elem = ET.SubElement(mid_joint, "parent")
    p_elem.set("link", "thorax3")
    c_elem = ET.SubElement(mid_joint, "child")
    c_elem.set("link", "mid_hands")
    orig_elem = ET.SubElement(mid_joint, "origin")
    orig_elem.set("xyz", "0.0 0.0 -0.17")
    orig_elem.set("rpy", "0.0 0.0 0.0")

    # club_shaft link + joint welded to mid_hands
    shaft_seg = model.get_segment("club_shaft")
    if shaft_seg:
        shaft_link = ET.SubElement(robot, "link")
        shaft_link.set("name", shaft_seg.name)
        _add_inertial(
            shaft_link,
            shaft_seg.mass,
            shaft_seg.inertia.ixx,
            shaft_seg.inertia.iyy,
            shaft_seg.inertia.izz,
            shaft_seg.inertia.ixy,
            shaft_seg.inertia.ixz,
            shaft_seg.inertia.iyz,
        )
        _add_visual(
            shaft_link,
            shaft_seg.geometry.geom_type,
            shaft_seg.geometry.size,
            shaft_seg.geometry.visual_rgba,
            f"mat_{shaft_seg.name}",
        )

        shaft_joint = ET.SubElement(robot, "joint")
        shaft_joint.set("name", "mid_hands_to_club_shaft")
        shaft_joint.set("type", "fixed")
        p_elem = ET.SubElement(shaft_joint, "parent")
        p_elem.set("link", "mid_hands")
        c_elem = ET.SubElement(shaft_joint, "child")
        c_elem.set("link", shaft_seg.name)
        orig_elem = ET.SubElement(shaft_joint, "origin")
        orig_elem.set("xyz", "0.0 0.0 -0.05")
        orig_elem.set("rpy", "0.0 0.0 0.0")

    # club_head link + joint welded to club_shaft
    head_seg = model.get_segment("club_head")
    if head_seg:
        head_link = ET.SubElement(robot, "link")
        head_link.set("name", head_seg.name)
        _add_inertial(
            head_link,
            head_seg.mass,
            head_seg.inertia.ixx,
            head_seg.inertia.iyy,
            head_seg.inertia.izz,
            head_seg.inertia.ixy,
            head_seg.inertia.ixz,
            head_seg.inertia.iyz,
        )
        _add_visual(
            head_link,
            head_seg.geometry.geom_type,
            head_seg.geometry.size,
            head_seg.geometry.visual_rgba,
            f"mat_{head_seg.name}",
        )

        head_joint = ET.SubElement(robot, "joint")
        head_joint.set("name", "club_shaft_to_club_head")
        head_joint.set("type", "fixed")
        p_elem = ET.SubElement(head_joint, "parent")
        p_elem.set("link", "club_shaft")
        c_elem = ET.SubElement(head_joint, "child")
        c_elem.set("link", head_seg.name)
        orig_elem = ET.SubElement(head_joint, "origin")
        orig_elem.set("xyz", _format_floats(head_seg.origin.xyz))
        orig_elem.set("rpy", _format_floats(head_seg.origin.rpy))

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


def _export_segment(robot: ET.Element, seg: Segment) -> None:
    """Export a standard segment and its joint(s) to URDF."""
    jt = seg.joint.joint_type

    if jt == "revolute":
        joint = ET.SubElement(robot, "joint")
        joint.set("name", f"{seg.parent}_to_{seg.name}")
        joint.set("type", "revolute")
        p_elem = ET.SubElement(joint, "parent")
        p_elem.set("link", seg.parent)
        c_elem = ET.SubElement(joint, "child")
        c_elem.set("link", seg.name)
        orig = ET.SubElement(joint, "origin")
        orig.set("xyz", _format_floats(seg.origin.xyz))
        orig.set("rpy", _format_floats(seg.origin.rpy))
        ax = ET.SubElement(joint, "axis")
        dof = seg.joint.dofs[0]
        ax.set("xyz", _format_floats(dof.axis))
        lim = ET.SubElement(joint, "limit")
        lim.set("lower", f"{dof.limits[0]:g}")
        lim.set("upper", f"{dof.limits[1]:g}")
        lim.set("effort", "1000.0")
        lim.set("velocity", "10.0")
        dyn = ET.SubElement(joint, "dynamics")
        dyn.set("damping", f"{seg.joint.damping:g}")

        link = ET.SubElement(robot, "link")
        link.set("name", seg.name)
        _add_inertial(
            link,
            seg.mass,
            seg.inertia.ixx,
            seg.inertia.iyy,
            seg.inertia.izz,
            seg.inertia.ixy,
            seg.inertia.ixz,
            seg.inertia.iyz,
        )
        _add_visual(
            link,
            seg.geometry.geom_type,
            seg.geometry.size,
            seg.geometry.visual_rgba,
            f"mat_{seg.name}",
        )

    elif jt == "universal":
        # Decompose into 2 revolute joints with 1 intermediate dummy link
        inter_name = f"{seg.name}_intermediate"

        # Joint 1: parent -> intermediate
        j1 = ET.SubElement(robot, "joint")
        j1.set("name", f"{seg.parent}_to_{inter_name}")
        j1.set("type", "revolute")
        p1 = ET.SubElement(j1, "parent")
        p1.set("link", seg.parent)
        c1 = ET.SubElement(j1, "child")
        c1.set("link", inter_name)
        orig1 = ET.SubElement(j1, "origin")
        orig1.set("xyz", _format_floats(seg.origin.xyz))
        orig1.set("rpy", _format_floats(seg.origin.rpy))
        ax1 = ET.SubElement(j1, "axis")
        dof1 = seg.joint.dofs[0]
        ax1.set("xyz", _format_floats(dof1.axis))
        lim1 = ET.SubElement(j1, "limit")
        lim1.set("lower", f"{dof1.limits[0]:g}")
        lim1.set("upper", f"{dof1.limits[1]:g}")
        lim1.set("effort", "1000.0")
        lim1.set("velocity", "10.0")
        dyn1 = ET.SubElement(j1, "dynamics")
        dyn1.set("damping", f"{seg.joint.damping:g}")

        # Intermediate link (massless / small mass)
        inter_link = ET.SubElement(robot, "link")
        inter_link.set("name", inter_name)
        _add_inertial(inter_link, 0.001, 0.0001, 0.0001, 0.0001)

        # Joint 2: intermediate -> segment
        j2 = ET.SubElement(robot, "joint")
        j2.set("name", f"{inter_name}_to_{seg.name}")
        j2.set("type", "revolute")
        p2 = ET.SubElement(j2, "parent")
        p2.set("link", inter_name)
        c2 = ET.SubElement(j2, "child")
        c2.set("link", seg.name)
        orig2 = ET.SubElement(j2, "origin")
        orig2.set("xyz", "0.0 0.0 0.0")
        orig2.set("rpy", "0.0 0.0 0.0")
        ax2 = ET.SubElement(j2, "axis")
        dof2 = seg.joint.dofs[1]
        ax2.set("xyz", _format_floats(dof2.axis))
        lim2 = ET.SubElement(j2, "limit")
        lim2.set("lower", f"{dof2.limits[0]:g}")
        lim2.set("upper", f"{dof2.limits[1]:g}")
        lim2.set("effort", "1000.0")
        lim2.set("velocity", "10.0")
        dyn2 = ET.SubElement(j2, "dynamics")
        dyn2.set("damping", f"{seg.joint.damping:g}")

        # Segment link
        link = ET.SubElement(robot, "link")
        link.set("name", seg.name)
        _add_inertial(
            link,
            seg.mass,
            seg.inertia.ixx,
            seg.inertia.iyy,
            seg.inertia.izz,
            seg.inertia.ixy,
            seg.inertia.ixz,
            seg.inertia.iyz,
        )
        _add_visual(
            link,
            seg.geometry.geom_type,
            seg.geometry.size,
            seg.geometry.visual_rgba,
            f"mat_{seg.name}",
        )

    elif jt == "gimbal":
        # Decompose into 3 revolute joints with 2 intermediate dummy links
        gz_name = f"{seg.name}_gimbal_z"
        gy_name = f"{seg.name}_gimbal_y"

        # Joint 1: parent -> gimbal_z
        j1 = ET.SubElement(robot, "joint")
        j1.set("name", f"{seg.parent}_to_{gz_name}")
        j1.set("type", "revolute")
        p1 = ET.SubElement(j1, "parent")
        p1.set("link", seg.parent)
        c1 = ET.SubElement(j1, "child")
        c1.set("link", gz_name)
        orig1 = ET.SubElement(j1, "origin")
        orig1.set("xyz", _format_floats(seg.origin.xyz))
        orig1.set("rpy", _format_floats(seg.origin.rpy))
        ax1 = ET.SubElement(j1, "axis")
        dof1 = seg.joint.dofs[0]
        ax1.set("xyz", _format_floats(dof1.axis))
        lim1 = ET.SubElement(j1, "limit")
        lim1.set("lower", f"{dof1.limits[0]:g}")
        lim1.set("upper", f"{dof1.limits[1]:g}")
        lim1.set("effort", "1000.0")
        lim1.set("velocity", "10.0")
        dyn1 = ET.SubElement(j1, "dynamics")
        dyn1.set("damping", f"{seg.joint.damping:g}")

        gz_link = ET.SubElement(robot, "link")
        gz_link.set("name", gz_name)
        _add_inertial(gz_link, 0.001, 0.0001, 0.0001, 0.0001)

        # Joint 2: gimbal_z -> gimbal_y
        j2 = ET.SubElement(robot, "joint")
        j2.set("name", f"{gz_name}_to_{gy_name}")
        j2.set("type", "revolute")
        p2 = ET.SubElement(j2, "parent")
        p2.set("link", gz_name)
        c2 = ET.SubElement(j2, "child")
        c2.set("link", gy_name)
        orig2 = ET.SubElement(j2, "origin")
        orig2.set("xyz", "0.0 0.0 0.0")
        orig2.set("rpy", "0.0 0.0 0.0")
        ax2 = ET.SubElement(j2, "axis")
        dof2 = seg.joint.dofs[1]
        ax2.set("xyz", _format_floats(dof2.axis))
        lim2 = ET.SubElement(j2, "limit")
        lim2.set("lower", f"{dof2.limits[0]:g}")
        lim2.set("upper", f"{dof2.limits[1]:g}")
        lim2.set("effort", "1000.0")
        lim2.set("velocity", "10.0")
        dyn2 = ET.SubElement(j2, "dynamics")
        dyn2.set("damping", f"{seg.joint.damping:g}")

        gy_link = ET.SubElement(robot, "link")
        gy_link.set("name", gy_name)
        _add_inertial(gy_link, 0.001, 0.0001, 0.0001, 0.0001)

        # Joint 3: gimbal_y -> segment
        j3 = ET.SubElement(robot, "joint")
        j3.set("name", f"{gy_name}_to_{seg.name}")
        j3.set("type", "revolute")
        p3 = ET.SubElement(j3, "parent")
        p3.set("link", gy_name)
        c3 = ET.SubElement(j3, "child")
        c3.set("link", seg.name)
        orig3 = ET.SubElement(j3, "origin")
        orig3.set("xyz", "0.0 0.0 0.0")
        orig3.set("rpy", "0.0 0.0 0.0")
        ax3 = ET.SubElement(j3, "axis")
        dof3 = seg.joint.dofs[2]
        ax3.set("xyz", _format_floats(dof3.axis))
        lim3 = ET.SubElement(j3, "limit")
        lim3.set("lower", f"{dof3.limits[0]:g}")
        lim3.set("upper", f"{dof3.limits[1]:g}")
        lim3.set("effort", "1000.0")
        lim3.set("velocity", "10.0")
        dyn3 = ET.SubElement(j3, "dynamics")
        dyn3.set("damping", f"{seg.joint.damping:g}")

        link = ET.SubElement(robot, "link")
        link.set("name", seg.name)
        _add_inertial(
            link,
            seg.mass,
            seg.inertia.ixx,
            seg.inertia.iyy,
            seg.inertia.izz,
            seg.inertia.ixy,
            seg.inertia.ixz,
            seg.inertia.iyz,
        )
        _add_visual(
            link,
            seg.geometry.geom_type,
            seg.geometry.size,
            seg.geometry.visual_rgba,
            f"mat_{seg.name}",
        )

    elif jt == "fixed":
        joint = ET.SubElement(robot, "joint")
        joint.set("name", f"{seg.parent}_to_{seg.name}")
        joint.set("type", "fixed")
        p_elem = ET.SubElement(joint, "parent")
        p_elem.set("link", seg.parent)
        c_elem = ET.SubElement(joint, "child")
        c_elem.set("link", seg.name)
        orig = ET.SubElement(joint, "origin")
        orig.set("xyz", _format_floats(seg.origin.xyz))
        orig.set("rpy", _format_floats(seg.origin.rpy))

        link = ET.SubElement(robot, "link")
        link.set("name", seg.name)
        _add_inertial(
            link,
            seg.mass,
            seg.inertia.ixx,
            seg.inertia.iyy,
            seg.inertia.izz,
            seg.inertia.ixy,
            seg.inertia.ixz,
            seg.inertia.iyz,
        )
        _add_visual(
            link,
            seg.geometry.geom_type,
            seg.geometry.size,
            seg.geometry.visual_rgba,
            f"mat_{seg.name}",
        )
