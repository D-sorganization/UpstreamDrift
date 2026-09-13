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
from typing import Any

import numpy as np

from src.engines.physics_engines.mujoco.python.native_mjcf import (
    _inertia,
    _numbers,
    _pose,
    transform,
)
from src.shared.python.motion_matching.full_body_spec import (
    order_directed_tree,
    upper_body_slice,
)


def export_full_body_mjcf(model_bytes: bytes) -> tuple[str, dict[str, Any]]:
    """Export the full-body specification to MJCF with 41 scalar joints and contact sites."""
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

    world = ET.SubElement(root, "worldbody")
    bodies = {body["name"]: body for body in spec["bodies"]}
    if len(bodies) != len(spec["bodies"]) or "world" not in bodies:
        raise ValueError("Invalid full-body body inventory")

    elements: dict[str, ET.Element] = {"world": world}
    offsets: dict[str, np.ndarray] = {"world": np.eye(4)}
    coordinates: list[str] = []

    # Separate upper-body joints and lower-limb joints
    upper_spec = upper_body_slice(spec)
    upper_joint_names = {j["name"] for j in upper_spec["joints"]}
    upper_ordered = order_directed_tree(upper_spec["joints"])

    # Lower limb chains: right leg then left leg
    lower_joints_by_name = {
        j["name"]: j for j in spec["joints"] if j["name"] not in upper_joint_names
    }
    leg_chain = [
        "hip_r",
        "knee_r",
        "ankle_r",
        "subtalar_r",
        "mtp_r",
        "hip_l",
        "knee_l",
        "ankle_l",
        "subtalar_l",
        "mtp_l",
    ]
    lower_ordered = [
        lower_joints_by_name[name] for name in leg_chain if name in lower_joints_by_name
    ]

    all_ordered_joints = upper_ordered + lower_ordered

    for joint in all_ordered_joints:
        parent, child = joint["parent"], joint["child"]
        if parent not in elements:
            raise ValueError(f"Parent body {parent} not yet constructed in tree")
        offset = np.linalg.inv(transform(joint["child_to_follower"]))
        element = ET.SubElement(
            elements[parent],
            "body",
            {
                "name": child,
                **_pose(offsets[parent] @ transform(joint["parent_to_base"])),
            },
        )
        for primitive in joint["primitives"]:
            kind, name = primitive["primitive"], primitive["coordinate"]
            if kind not in ("Px", "Py", "Pz", "Rx", "Ry", "Rz") or name in coordinates:
                raise ValueError(f"Duplicate or unsupported coordinate: {name}")
            coordinates.append(name)
            ET.SubElement(
                element,
                "joint",
                name=name,
                type="slide" if kind[0] == "P" else "hinge",
                axis=_numbers(np.eye(3)["xyz".index(kind[1])]),
                limited="false",
                damping="0",
                armature="0",
                frictionloss="0",
                stiffness="0",
            )
        ET.SubElement(element, "inertial", _inertia(bodies[child], offset))
        elements[child], offsets[child] = element, offset

    if (
        set(elements) != set(bodies)
        or len(coordinates) != len(spec["coordinate_order"])
        or set(coordinates) != set(spec["coordinate_order"])
    ):
        raise ValueError("Body or coordinate inventory was not preserved")

    # Frame sites
    frame_sites: dict[str, str] = {}
    for i, frame in enumerate(spec["frames"]):
        if frame["name"] in frame_sites:
            raise ValueError(f"Duplicate frame name {frame['name']}")
        site = f"native_frame_{i}"
        frame_sites[frame["name"]] = site
        body_name = frame["body"]
        ET.SubElement(
            elements[body_name],
            "site",
            {
                "name": site,
                "size": ".001",
                **_pose(offsets[body_name] @ transform(frame["placement"])),
            },
        )

    # Contact geoms and sites
    contact_sites: dict[str, str] = {}
    for sphere in spec["contact"]["spheres"]:
        s_name = sphere["name"]
        b_name = sphere["body"]
        radius = float(sphere["radius_m"])
        p_body = np.asarray(sphere["position_m"], dtype=float)
        # Position in MuJoCo body frame
        p_mjcf = offsets[b_name][:3, :3] @ p_body + offsets[b_name][:3, 3]

        geom_name = f"contact_{s_name}"
        site_name = f"contact_site_{s_name}"
        contact_sites[s_name] = site_name

        ET.SubElement(
            elements[b_name],
            "geom",
            attrib={"class": "contact"},
            name=geom_name,
            type="sphere",
            size=_numbers([radius]),
            pos=_numbers(p_mjcf),
        )
        ET.SubElement(
            elements[b_name],
            "site",
            name=site_name,
            pos=_numbers(p_mjcf),
            size=".005",
        )

    # Dual-grip weld closure sites
    for suffix in ("a", "b"):
        closure_body = spec["closure"][f"body_{suffix}"]
        ET.SubElement(
            elements[closure_body],
            "site",
            {
                "name": f"native_closure_{suffix}",
                "size": ".001",
                **_pose(
                    offsets[closure_body]
                    @ transform(spec["closure"][f"placement_{suffix}"])
                ),
            },
        )
    ET.SubElement(
        ET.SubElement(root, "equality"),
        "weld",
        name="native_grip",
        site1="native_closure_a",
        site2="native_closure_b",
    )

    xml = ET.tostring(root, encoding="unicode")
    return xml, {
        "representation": "native-full-body-mjcf-v1",
        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "mjcf_sha256": hashlib.sha256(xml.encode("utf-8")).hexdigest(),
        "coordinate_order": spec["coordinate_order"],
        "frame_sites": frame_sites,
        "contact_sites": contact_sites,
        "execution": "explicit-rigid-closure-and-contact; stock mj_step unqualified",
    }
