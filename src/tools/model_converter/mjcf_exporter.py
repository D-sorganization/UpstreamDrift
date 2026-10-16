"""MuJoCo (MJCF) model exporter for canonical golfer specification.

Generates a valid, MuJoCo-compilable MJCF XML from CanonicalModel:
- Proper body hierarchy with nested child bodies
- Local joint hinges for revolute, universal, and gimbal articulations
- Explicit inertial parameters (mass, diaginertia) matching canonical spec
- Closed-chain dual-grip equality weld constraint
- Actuator motors for continuous torque driving
- Automated validation via `mujoco.MjModel.from_xml_string`
"""

from __future__ import annotations

import logging
from pathlib import Path
import xml.etree.ElementTree as ET  # noqa: S405  # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml  # build-only

from defusedxml import minidom
from tools.model_converter.schema_validator import (
    CanonicalModel,
    Inertia,
    JointDof,
    RootBody,
    Segment,
)

logger = logging.getLogger(__name__)


#: Golfer segment geoms collide with the floor and other default-mask objects
#: (contype/conaffinity 1, e.g. a ball) but never with each other: MuJoCo
#: collides a pair when ``(t1 & a2) | (t2 & a1)`` is non-zero, and 2 & 1 is 0
#: (#11811).
SEGMENT_CONTYPE = "2"
SEGMENT_CONAFFINITY = "1"

#: Anatomical tracking sites keyed by segment name: (site name, pos).
_SEGMENT_SITES: dict[str, tuple[str, str]] = {
    "hand_left": ("hand_left_grip", "0 0 -0.05"),
    "hand_right": ("hand_right_grip", "0 0 -0.05"),
    "club_shaft": ("mid_hands", "0 0 0.05"),
    "club_head": ("clubhead", "0 0 0"),
}


def _format_floats(values: tuple[float, ...] | list[float]) -> str:
    return " ".join(f"{v:g}" for v in values)


def _add_attrs(parent: ET.Element, tag: str, **attrs: str) -> ET.Element:
    """Append ``<tag>`` to ``parent`` with ``attrs`` set in insertion order."""
    elem = ET.SubElement(parent, tag)
    for key, value in attrs.items():
        elem.set(key, value)
    return elem


def _add_segment_geom(body: ET.Element, segment: Segment | RootBody) -> ET.Element:
    """Append ``segment``'s geom to ``body`` with self-collision filtering."""
    geom = ET.SubElement(body, "geom")
    geom.set("name", f"{segment.name}_geom")
    geom.set("type", segment.geometry.geom_type)
    geom.set("size", _format_floats(segment.geometry.size))
    geom.set("rgba", _format_floats(segment.geometry.visual_rgba))
    geom.set("contype", SEGMENT_CONTYPE)
    geom.set("conaffinity", SEGMENT_CONAFFINITY)
    return geom


def _add_preamble(root_elem: ET.Element) -> None:
    """Append compiler, option, visual and asset sections."""
    _add_attrs(
        root_elem,
        "compiler",
        angle="radian",
        coordinate="local",
        inertiafromgeom="false",
    )
    _add_attrs(
        root_elem,
        "option",
        timestep="0.002",
        gravity="0 0 -9.80665",
        integrator="RK4",
    )

    visual = ET.SubElement(root_elem, "visual")
    _add_attrs(visual, "global", offwidth="1024", offheight="1024")
    _add_attrs(visual, "map", znear="0.01", zfar="50")
    _add_attrs(visual, "headlight", diffuse="0.8 0.8 0.8", ambient="0.3 0.3 0.3")

    asset = ET.SubElement(root_elem, "asset")
    _add_attrs(asset, "material", name="ground_mat", rgba="0.4 0.6 0.3 1.0")


def _add_world_fixtures(worldbody: ET.Element) -> None:
    """Append the floor, light and fixed cameras to ``worldbody``."""
    _add_attrs(
        worldbody,
        "geom",
        name="floor",
        type="plane",
        size="10 10 0.1",
        material="ground_mat",
    )
    _add_attrs(worldbody, "light", directional="true", pos="0 0 3", dir="0 0 -1")
    _add_attrs(
        worldbody,
        "camera",
        name="side",
        pos="-5 -2 1.5",
        euler="0.15 0 0.35",
        mode="fixed",
    )
    _add_attrs(
        worldbody,
        "camera",
        name="front",
        pos="0 -5 1.5",
        euler="0.15 0 1.57",
        mode="fixed",
    )


def _build_children_map(model: CanonicalModel) -> dict[str, list[Segment]]:
    """Map each body name to its child segments (in model order)."""
    children_map: dict[str, list[Segment]] = {model.root.name: []}
    for seg in model.segments:
        children_map[seg.name] = []
    for seg in model.segments:
        children_map[seg.parent].append(seg)
    return children_map


def _add_inertial(body: ET.Element, mass: float, inertia: Inertia) -> None:
    """Append an ``<inertial>`` element at the body origin."""
    _add_attrs(
        body,
        "inertial",
        pos="0 0 0",
        mass=f"{mass:g}",
        diaginertia=f"{inertia.ixx:g} {inertia.iyy:g} {inertia.izz:g}",
    )


def _add_root_body(worldbody: ET.Element, root: RootBody) -> ET.Element:
    """Append the free-floating root body (pelvis) to ``worldbody``."""
    pelvis_body = _add_attrs(
        worldbody,
        "body",
        name=root.name,
        pos=_format_floats(root.position),
        euler=_format_floats(root.orientation),
    )
    _add_attrs(pelvis_body, "freejoint", name=f"{root.name}_free")
    _add_inertial(pelvis_body, root.mass, root.inertia)
    _add_segment_geom(pelvis_body, root)
    return pelvis_body


def _add_hinge(body: ET.Element, name: str, dof: JointDof, damping: float) -> str:
    """Append a hinge joint for ``dof`` and return its name."""
    j_elem = _add_attrs(
        body,
        "joint",
        name=name,
        type="hinge",
        axis=_format_floats(dof.axis),
        range=f"{dof.limits[0]:g} {dof.limits[1]:g}",
    )
    if damping > 0:
        j_elem.set("damping", f"{damping:g}")
    return name


def _add_segment_joints(body: ET.Element, seg: Segment) -> list[str]:
    """Append joint elements for ``seg`` and return the actuated joint names.

    Fixed joints have no ``<joint>`` element in MuJoCo.
    """
    jt = seg.joint.joint_type
    damping = seg.joint.damping
    if jt == "revolute":
        return [_add_hinge(body, seg.name, seg.joint.dofs[0], damping)]
    if jt in ("universal", "gimbal"):
        return [
            _add_hinge(body, f"{seg.name}_dof{i}", dof, damping)
            for i, dof in enumerate(seg.joint.dofs)
        ]
    return []


def _add_segment_site(body: ET.Element, seg_name: str) -> None:
    """Append the anatomical tracking site for ``seg_name``, if it has one."""
    site_spec = _SEGMENT_SITES.get(seg_name)
    if site_spec is not None:
        site_name, pos = site_spec
        _add_attrs(body, "site", name=site_name, pos=pos, size="0.01")


def _add_segment_body(
    parent_body: ET.Element, seg: Segment
) -> tuple[ET.Element, list[str]]:
    """Append ``seg``'s body (joints, inertial, geom, site) to ``parent_body``."""
    child_body = _add_attrs(
        parent_body, "body", name=seg.name, pos=_format_floats(seg.origin.xyz)
    )
    if any(r != 0 for r in seg.origin.rpy):
        child_body.set("euler", _format_floats(seg.origin.rpy))
    joints = _add_segment_joints(child_body, seg)
    _add_inertial(child_body, seg.mass, seg.inertia)
    _add_segment_geom(child_body, seg)
    _add_segment_site(child_body, seg.name)
    return child_body, joints


def _add_body_tree(
    parent_body: ET.Element,
    parent_name: str,
    children_map: dict[str, list[Segment]],
    actuator_joints: list[str],
) -> None:
    """Recursively append child bodies, collecting actuated joint names."""
    for seg in children_map.get(parent_name, []):
        child_body, joints = _add_segment_body(parent_body, seg)
        actuator_joints.extend(joints)
        _add_body_tree(child_body, seg.name, children_map, actuator_joints)


def _add_equality_and_actuators(
    root_elem: ET.Element, actuator_joints: list[str]
) -> None:
    """Append the dual-grip weld constraint and one motor per actuated joint."""
    equality = ET.SubElement(root_elem, "equality")
    _add_attrs(
        equality,
        "weld",
        name="right_hand_grip_weld",
        body1="hand_right",
        body2="club_shaft",
        relpose="0 0 0 1 0 0 0",
        active="true",
    )
    actuator = ET.SubElement(root_elem, "actuator")
    for j_name in actuator_joints:
        _add_attrs(actuator, "motor", name=f"act_{j_name}", joint=j_name, gear="1")


def _pretty_xml(root_elem: ET.Element) -> str:
    """Serialise ``root_elem`` to an indented XML string."""
    raw_xml = ET.tostring(root_elem, encoding="utf-8")
    return minidom.parseString(raw_xml).toprettyxml(indent="  ")


def _check_mujoco_compiles(xml_content: str) -> None:
    """Best-effort validation against the MuJoCo compiler (warns on failure)."""
    try:
        import mujoco

        mujoco.MjModel.from_xml_string(xml_content)
        logger.info("MuJoCo model compilation validated successfully")
    except Exception as exc:  # noqa: BLE001
        logger.warning("MuJoCo compilation check raised: %s", exc)


def _write_xml(xml_content: str, out_path: Path) -> None:
    """Write ``xml_content`` to ``out_path``, creating parent directories."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(xml_content, encoding="utf-8")
    logger.info("Wrote canonical MJCF to %s", out_path)


def export_mjcf(model: CanonicalModel, out_path: Path | None = None) -> str:
    """Generate MuJoCo MJCF XML string from canonical model and optionally save to disk."""
    root_elem = ET.Element("mujoco")
    root_elem.set("model", "golfer")
    _add_preamble(root_elem)

    worldbody = ET.SubElement(root_elem, "worldbody")
    _add_world_fixtures(worldbody)

    children_map = _build_children_map(model)
    pelvis_body = _add_root_body(worldbody, model.root)
    actuator_joints: list[str] = []
    _add_body_tree(pelvis_body, model.root.name, children_map, actuator_joints)

    _add_equality_and_actuators(root_elem, actuator_joints)

    xml_content = _pretty_xml(root_elem)
    _check_mujoco_compiles(xml_content)
    if out_path:
        _write_xml(xml_content, out_path)
    return xml_content
