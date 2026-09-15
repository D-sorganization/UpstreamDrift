"""Export a full-body model specification to an MJCF document with shared contact.

Embeds the qualified upper-body rigid body tree byte-identically and appends the
lower limbs, contact sphere geoms and sites, and the closed-loop dual-grip weld.
Stock MuJoCo contact is disabled so that NativeMujocoFullBodyModel applies the
FB-2 shared compliant Hunt-Crossley and regularized Coulomb contact law.
"""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET  # nosec B405 # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml - construction only; parsing is defused
from collections.abc import Mapping
from typing import Any

import numpy as np

from src.engines.physics_engines.mujoco.python.native_mjcf import (
    _add_weld_equality,
    _attach_frame_site,
    _attach_joint_element,
    _numbers,
)
from src.shared.python.motion_matching.full_body_spec import order_full_body_joints


def _build_full_body_kinematics(
    root: ET.Element,
    spec: Mapping[str, Any],
) -> tuple[dict[str, ET.Element], dict[str, np.ndarray]]:
    """Construct worldbody, rigid body elements, joints, and inertias."""
    world = ET.SubElement(root, "worldbody")
    bodies = {body["name"]: body for body in spec["bodies"]}
    if len(bodies) != len(spec["bodies"]) or "world" not in bodies:
        raise ValueError("Invalid full-body body inventory")

    elements: dict[str, ET.Element] = {"world": world}
    offsets: dict[str, np.ndarray] = {"world": np.eye(4)}
    coordinates: list[str] = []

    for joint in order_full_body_joints(spec):
        parent, child = joint["parent"], joint["child"]
        if parent not in elements:
            raise ValueError(f"Parent body {parent} not yet constructed in tree")
        element, offset = _attach_joint_element(
            elements[parent], bodies[child], joint, offsets[parent], coordinates
        )
        elements[child], offsets[child] = element, offset

    if (
        set(elements) != set(bodies)
        or len(coordinates) != len(spec["coordinate_order"])
        or set(coordinates) != set(spec["coordinate_order"])
    ):
        raise ValueError("Body or coordinate inventory was not preserved")
    return elements, offsets


def _attach_full_body_sites(
    root: ET.Element,
    elements: Mapping[str, ET.Element],
    offsets: Mapping[str, np.ndarray],
    spec: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, str]]:
    """Attach frame sites, contact sphere geoms/sites, and closure weld sites."""
    frame_sites: dict[str, str] = {}
    for i, frame in enumerate(spec["frames"]):
        if frame["name"] in frame_sites:
            raise ValueError("Duplicate full-body frame")
        site = f"native_frame_{i}"
        frame_sites[frame["name"]] = site
        _attach_frame_site(
            elements[frame["body"]], site, offsets[frame["body"]], frame["placement"]
        )

    contact_sites: dict[str, str] = {}
    for sphere in spec["contact"]["spheres"]:
        s_name = sphere["name"]
        b_name = sphere["body"]
        radius = float(sphere["radius_m"])
        p_body = np.asarray(sphere["position_m"], dtype=float)
        p_mjcf = offsets[b_name][:3, :3] @ p_body + offsets[b_name][:3, 3]

        geom_name = f"contact_{s_name}"
        ET.SubElement(
            elements[b_name],
            "geom",
            name=geom_name,
            type="sphere",
            size=format(radius, ".17g"),
            pos=_numbers(p_mjcf),
            attrib={"class": "contact"},
        )

        site_name = f"contact_site_{s_name}"
        contact_sites[s_name] = site_name
        ET.SubElement(
            elements[b_name],
            "site",
            name=site_name,
            pos=_numbers(p_mjcf),
            size=".005",
        )

    for suffix in ("a", "b"):
        closure_body = spec["closure"][f"body_{suffix}"]
        _attach_frame_site(
            elements[closure_body],
            f"native_closure_{suffix}",
            offsets[closure_body],
            spec["closure"][f"placement_{suffix}"],
        )
    _add_weld_equality(root)
    return frame_sites, contact_sites


def export_full_body_mjcf(
    model_bytes: bytes, *, visual: bool = False
) -> tuple[str, dict[str, Any]]:
    """Export the full-body specification to MJCF with 41 scalar joints and contact sites.

    ``visual=True`` additionally attaches the shared visual skeleton (capsules,
    COM/frame spheres, floor, lights, camera) as massless non-colliding geoms;
    the physics content is identical to the plain export.
    """
    spec = json.loads(model_bytes)
    if spec.get("schema_version") != "full-body-v1":
        raise ValueError("Unsupported schema version: expected full-body-v1")
    if not isinstance(spec.get("closure"), dict):
        raise ValueError("Missing closure specification")
    if not isinstance(spec.get("contact"), dict):
        raise ValueError("Missing contact specification")

    root = ET.Element("mujoco", model="full_body_golf")
    ET.SubElement(
        root, "compiler", angle="radian", autolimits="false", inertiafromgeom="false"
    )
    gravity = np.asarray(spec["gravity_m_s2"], dtype=float)
    if gravity.shape != (3,) or not np.isfinite(gravity).all():
        raise ValueError("Invalid full-body gravity")
    option = ET.SubElement(root, "option", gravity=_numbers(gravity), jacobian="dense")
    ET.SubElement(option, "flag", contact="disable")

    default_root = ET.SubElement(root, "default")
    default_contact = ET.SubElement(
        default_root, "default", attrib={"class": "contact"}
    )
    ET.SubElement(default_contact, "geom", contype="0", conaffinity="0")

    elements, offsets = _build_full_body_kinematics(root, spec)
    frame_sites, contact_sites = _attach_full_body_sites(root, elements, offsets, spec)
    visual_meta = None
    if visual:
        from src.engines.physics_engines.mujoco.python.visual_layer import (
            attach_visual_layer,
        )

        visual_meta = attach_visual_layer(root, elements, offsets, spec)

    xml = ET.tostring(root, encoding="unicode")
    extra = {} if visual_meta is None else {"visual_layer": visual_meta}
    return xml, {
        **extra,
        "representation": "native-full-body-mjcf-v1",
        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "mjcf_sha256": hashlib.sha256(xml.encode("utf-8")).hexdigest(),
        "coordinate_order": spec["coordinate_order"],
        "frame_sites": frame_sites,
        "contact_sites": contact_sites,
        "execution": "explicit-rigid-closure-and-contact; stock mj_step unqualified",
    }
