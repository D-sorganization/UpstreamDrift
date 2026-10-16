"""MuJoCo model of a grip-clamped chain shaft plus head (R6, #12686).

Emits :func:`club_shaft_chain.clamped_shaft_document` with the same joint and
inertia emitter as the full-body export (``native_mjcf._attach_joint_element``),
so the static and modal reference checks exercise exactly the MJCF the swing
model uses. A ``shaft_tip`` site marks the head centre on the tip link.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET  # nosec B405 # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml - construction only; parsing is defused
from collections.abc import Mapping
from typing import Any

import numpy as np

from src.engines.physics_engines.mujoco.python.native_mjcf import (
    _attach_frame_site,
    _attach_joint_element,
    _numbers,
)
from src.shared.python.motion_matching.full_body_spec import order_directed_tree

TIP_SITE = "shaft_tip"


def export_clamped_shaft_mjcf(document: Mapping[str, Any]) -> str:
    """MJCF of a clamped-shaft document; the coordinates keep spec order.

    Raises ``ValueError`` when the joint tree does not reach every body or the
    emitted coordinates differ from ``coordinate_order``.
    """
    root = ET.Element("mujoco", model="clamped_shaft")
    ET.SubElement(
        root, "compiler", angle="radian", autolimits="false", inertiafromgeom="false"
    )
    ET.SubElement(
        root,
        "option",
        gravity=_numbers(np.asarray(document["gravity_m_s2"], dtype=float)),
        jacobian="dense",
    )
    bodies = {body["name"]: body for body in document["bodies"]}
    elements: dict[str, Any] = {"world": ET.SubElement(root, "worldbody")}
    offsets: dict[str, np.ndarray] = {"world": np.eye(4)}
    coordinates: list[str] = []
    for joint in order_directed_tree(document["joints"]):
        element, offset = _attach_joint_element(
            elements[joint["parent"]],
            bodies[joint["child"]],
            joint,
            offsets[joint["parent"]],
            coordinates,
        )
        elements[joint["child"]], offsets[joint["child"]] = element, offset
    if set(elements) != set(bodies) or coordinates != list(
        document["coordinate_order"]
    ):
        raise ValueError("Clamped shaft body or coordinate inventory not preserved")
    tip = document["tip_body"]
    placement = np.eye(4)
    placement[:3, 3] = document["tip_point_m"]
    _attach_frame_site(elements[tip], TIP_SITE, offsets[tip], placement)
    return ET.tostring(root, encoding="unicode")
