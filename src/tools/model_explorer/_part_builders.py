"""Generators for the bundled synthetic assembly parts (limbs, torso, ...).

Every generated link carries a physically valid inertial block so a freshly
dropped part never trips the composition validator's inertial checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.shared.python.model_generation.editor.attachment_ports import (
    PortPolarity,
    PortType,
)
from src.tools.model_explorer.attachment_manifest import (
    AttachmentInterfaceFrame,
    AttachmentPoint,
)

if TYPE_CHECKING:
    from src.tools.model_explorer.part_catalog import PartSpec

BUNDLED_CATEGORY_LABELS: dict[str, str] = {
    "base": "Bases",
    "torso": "Torso And Pelvis",
    "limb": "Limbs",
    "head": "Heads",
    "shoe": "Shoes",
    "club": "Golf Clubs",
    "robot_arm": "Robot Arms",
}


@dataclass(frozen=True)
class _LinkDef:
    name: str
    length: float
    radius: float
    mass: float
    joint: str = "revolute"


def _link_xml(link: _LinkDef) -> str:
    m, r, h = link.mass, link.radius, link.length
    ixx = m * (3 * r * r + h * h) / 12.0
    izz = m * r * r / 2.0
    return (
        f'  <link name="{link.name}">\n'
        f'    <inertial><origin xyz="0 0 {h / 2:.4f}"/><mass value="{m}"/>'
        f'<inertia ixx="{ixx:.6f}" iyy="{ixx:.6f}" izz="{izz:.6f}" '
        'ixy="0" ixz="0" iyz="0"/></inertial>\n'
        f'    <visual><origin xyz="0 0 {h / 2:.4f}"/><geometry>'
        f'<cylinder length="{h}" radius="{r}"/></geometry></visual>\n'
        "  </link>\n"
    )


def _joint_xml(parent: _LinkDef, child: _LinkDef) -> str:
    axis = '    <axis xyz="0 1 0"/>\n    <limit lower="-2.0" upper="2.0" effort="100" velocity="5"/>\n'
    body = axis if child.joint == "revolute" else ""
    return (
        f'  <joint name="{parent.name}_to_{child.name}" type="{child.joint}">\n'
        f'    <parent link="{parent.name}"/><child link="{child.name}"/>\n'
        f'    <origin xyz="0 0 {parent.length + 0.002:.4f}"/>\n{body}  </joint>\n'
    )


def chain_urdf(robot: str, links: list[_LinkDef]) -> str:
    """URDF for a serial chain of cylinders along +z."""
    if not links:
        raise ValueError("links must not be empty")
    xml = f'<?xml version="1.0"?>\n<robot name="{robot}">\n'
    xml += "".join(_link_xml(link) for link in links)
    xml += "".join(_joint_xml(a, b) for a, b in zip(links, links[1:], strict=False))
    return xml + "</robot>\n"


def _port(
    name: str,
    link: str,
    port_type: PortType,
    polarity: PortPolarity,
    *,
    side: str | None = None,
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
    payload: float | None = None,
) -> AttachmentPoint:
    return AttachmentPoint(
        name=name,
        link_name=link,
        role=f"{port_type.value}-{polarity.value}",
        interface_frame=AttachmentInterfaceFrame(
            xyz=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
            rpy=(float(rpy[0]), float(rpy[1]), float(rpy[2])),
        ),
        max_payload_kg=payload,
        tags=(side,) if side else (),
        port_type=port_type,
        polarity=polarity,
    )


def _part(
    part_id: str,
    name: str,
    category: str,
    description: str,
    links: list[_LinkDef],
    ports: tuple[AttachmentPoint, ...],
) -> PartSpec:
    from src.tools.model_explorer.part_catalog import PartSpec

    return PartSpec(
        part_id=part_id,
        name=name,
        category=category,
        description=description,
        urdf_xml=chain_urdf(part_id, links),
        ports=ports,
    )


_S, _P = PortPolarity.SOCKET, PortPolarity.PLUG
_DOWN = (3.14159265, 0.0, 0.0)  # flip +z so limbs hang below their socket


def _torso() -> PartSpec:
    links = [
        _LinkDef("pelvis", 0.18, 0.12, 16.0),
        _LinkDef("chest", 0.45, 0.14, 22.0, joint="fixed"),
    ]
    ports = (
        _port(
            "hip_left",
            "pelvis",
            PortType.HIP,
            _S,
            side="left",
            xyz=(0, 0.09, 0),
            rpy=_DOWN,
            payload=20.0,
        ),
        _port(
            "hip_right",
            "pelvis",
            PortType.HIP,
            _S,
            side="right",
            xyz=(0, -0.09, 0),
            rpy=_DOWN,
            payload=20.0,
        ),
        _port(
            "shoulder_left",
            "chest",
            PortType.SHOULDER,
            _S,
            side="left",
            xyz=(0, 0.2, 0.4),
            rpy=_DOWN,
            payload=8.0,
        ),
        _port(
            "shoulder_right",
            "chest",
            PortType.SHOULDER,
            _S,
            side="right",
            xyz=(0, -0.2, 0.4),
            rpy=_DOWN,
            payload=8.0,
        ),
        _port("neck", "chest", PortType.NECK, _S, xyz=(0, 0, 0.45), payload=8.0),
    )
    return _part(
        "humanoid_torso",
        "Torso And Pelvis",
        "torso",
        "Pelvis and chest with hip, shoulder and neck sockets.",
        links,
        ports,
    )


def _arm(side: str) -> PartSpec:
    pid = f"arm_{side}"
    links = [
        _LinkDef(f"upper_arm_{side}", 0.30, 0.045, 2.0),
        _LinkDef(f"forearm_{side}", 0.27, 0.035, 1.5),
        _LinkDef(f"hand_{side}", 0.09, 0.04, 0.5),
    ]
    ports = (
        _port("shoulder_plug", links[0].name, PortType.SHOULDER, _P, side=side),
        _port(
            "hand_grip",
            links[2].name,
            PortType.GRIP,
            _S,
            side=side,
            xyz=(0, 0, 0.05),
            payload=2.0,
        ),
    )
    return _part(
        pid,
        f"Arm ({side.title()})",
        "limb",
        f"{side.title()} arm with a hand grip socket.",
        links,
        ports,
    )


def _leg(side: str) -> PartSpec:
    links = [
        _LinkDef(f"thigh_{side}", 0.42, 0.07, 8.0),
        _LinkDef(f"shank_{side}", 0.40, 0.05, 3.5),
        _LinkDef(f"foot_{side}", 0.08, 0.05, 1.2),
    ]
    ports = (
        _port("hip_plug", links[0].name, PortType.HIP, _P, side=side),
        _port("ankle", links[2].name, PortType.ANKLE, _S, side=side, payload=2.5),
    )
    return _part(
        f"leg_{side}",
        f"Leg ({side.title()})",
        "limb",
        f"{side.title()} leg with an ankle socket.",
        links,
        ports,
    )


def _head() -> PartSpec:
    links = [_LinkDef("head", 0.22, 0.09, 4.5)]
    ports = (_port("neck_plug", "head", PortType.NECK, _P),)
    return _part("head", "Head", "head", "Head with a neck plug.", links, ports)


def _shoe(side: str) -> PartSpec:
    links = [_LinkDef(f"shoe_{side}", 0.10, 0.06, 0.6)]
    ports = (_port("ankle_plug", links[0].name, PortType.ANKLE, _P, side=side),)
    return _part(
        f"shoe_{side}",
        f"Shoe ({side.title()})",
        "shoe",
        f"{side.title()} golf shoe.",
        links,
        ports,
    )


def _iron() -> PartSpec:
    links = [
        _LinkDef("iron_grip", 0.25, 0.013, 0.05, joint="fixed"),
        _LinkDef("iron_shaft", 0.70, 0.006, 0.07, joint="fixed"),
        _LinkDef("iron_head", 0.08, 0.03, 0.25, joint="fixed"),
    ]
    ports = (_port("grip_plug", "iron_grip", PortType.GRIP, _P),)
    return _part("club_iron", "7 Iron", "club", "Seven iron.", links, ports)


def _pedestal() -> PartSpec:
    links = [_LinkDef("pedestal", 0.80, 0.20, 40.0)]
    ports = (
        _port(
            "top_mount", "pedestal", PortType.MOUNT, _S, xyz=(0, 0, 0.8), payload=60.0
        ),
    )
    return _part("pedestal", "Pedestal", "base", "Fixed floor pedestal.", links, ports)


def _robot_arm() -> PartSpec:
    links = [
        _LinkDef("ra_base", 0.10, 0.10, 6.0),
        _LinkDef("ra_link1", 0.40, 0.05, 4.0),
        _LinkDef("ra_link2", 0.35, 0.04, 3.0),
        _LinkDef("ra_tool", 0.06, 0.04, 0.8),
    ]
    ports = (
        _port("base_plug", "ra_base", PortType.MOUNT, _P),
        _port("tool_grip", "ra_tool", PortType.GRIP, _S, xyz=(0, 0, 0.06), payload=3.0),
    )
    return _part(
        "robot_arm",
        "Robot Arm",
        "robot_arm",
        "Three-joint robot arm with a tool grip socket.",
        links,
        ports,
    )


def bundled_parts() -> tuple[PartSpec, ...]:
    """All bundled synthetic parts."""
    return (
        _torso(),
        _arm("left"),
        _arm("right"),
        _leg("left"),
        _leg("right"),
        _head(),
        _shoe("left"),
        _shoe("right"),
        _iron(),
        _pedestal(),
        _robot_arm(),
    )
