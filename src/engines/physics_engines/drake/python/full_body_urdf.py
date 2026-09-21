"""Export a full-body model specification to a URDF bundle with shared contact.

Embeds the physical solids and scalar 1-DOF joint chain of the full-body model
specification (41 coordinates total: 27 upper-body joints + 14 lower-limb joints).
Adds four foot contact sphere links on the calcaneus bodies (calcn_r, calcn_l)
and generates sidecar metadata preserving closure, coordinate order, and frames.
"""

from __future__ import annotations

import hashlib
import json
import sys
import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from src.shared.python.model_generation.builders.urdf_writer import URDFWriter
from src.shared.python.model_generation.core.types import (
    Inertia,
    Joint,
    JointDynamics,
    JointLimits,
    JointType,
    Link,
    Origin,
)
from src.shared.python.motion_matching.full_body_spec import (
    order_directed_tree,
    upper_body_slice,
)


def _origin(value: Any) -> Origin:
    matrix = np.asarray(value, dtype=float)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-12, rtol=0)
        or not np.allclose(
            matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-10, rtol=0
        )
        or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-10, rtol=0)
    ):
        raise ValueError("Invalid rigid transform for URDF origin")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Gimbal lock detected.*", category=UserWarning
        )
        rpy = Rotation.from_matrix(matrix[:3, :3]).as_euler("xyz")
    if not np.allclose(
        Rotation.from_euler("xyz", rpy).as_matrix(), matrix[:3, :3], atol=1e-12, rtol=0
    ):
        raise ValueError("Rigid transform cannot be preserved in URDF RPY")
    xyz = (float(matrix[0, 3]), float(matrix[1, 3]), float(matrix[2, 3]))
    rpy_tuple = (float(rpy[0]), float(rpy[1]), float(rpy[2]))
    return Origin(xyz=xyz, rpy=rpy_tuple)


def _order_full_body_joints(spec: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Sequence joints with upper-body tree first, then lower limb chains."""
    if "upper_body_counts" in spec:
        upper_spec = upper_body_slice(spec)
        upper_joint_names = {j["name"] for j in upper_spec["joints"]}
        upper_ordered = order_directed_tree(upper_spec["joints"])
    else:
        upper_joint_names = {j["name"] for j in spec["joints"]}
        upper_ordered = order_directed_tree(spec["joints"])

    lower_ordered = [j for j in spec["joints"] if j["name"] not in upper_joint_names]
    return list(upper_ordered) + lower_ordered


def _build_joint_primitives(
    joint: Mapping[str, Any],
    parent_link: str,
    target_child_link: str,
    coordinates: list[str],
    links: list[Link],
    joints: list[Joint],
) -> None:
    """Decompose joint into scalar 1-DOF joints and intermediate massless links."""
    parent = parent_link
    placement = joint["parent_to_base"]
    bound = sys.float_info.max

    for primitive in joint["primitives"]:
        kind, name = primitive["primitive"], primitive["coordinate"]
        if kind not in ("Px", "Py", "Pz", "Rx", "Ry", "Rz") or name in coordinates:
            raise ValueError(f"Duplicate or unsupported coordinate {name}")
        child = f"primitive_{len(coordinates)}"
        coordinates.append(name)
        links.append(Link(name=child, inertia=Inertia(0.0, 0.0, 0.0, mass=0.0)))
        axis_index = "xyz".index(kind[1].lower())
        axis = (
            float(axis_index == 0),
            float(axis_index == 1),
            float(axis_index == 2),
        )
        joints.append(
            Joint(
                name=name,
                parent=parent,
                child=child,
                joint_type=(
                    JointType.PRISMATIC if kind[0] == "P" else JointType.REVOLUTE
                ),
                origin=_origin(placement),
                axis=axis,
                limits=JointLimits(-bound, bound, bound, bound),
                dynamics=JointDynamics(damping=0.0, friction=0.0),
            )
        )
        parent, placement = child, np.eye(4)

    joints.append(
        Joint(
            name=f"fixed_{target_child_link}",
            joint_type=JointType.FIXED,
            parent=parent,
            child=target_child_link,
            origin=_origin(np.linalg.inv(joint["child_to_follower"])),
            dynamics=JointDynamics(damping=0.0, friction=0.0),
        )
    )


def _attach_solids_and_frames(
    spec: Mapping[str, Any],
    body_links: Mapping[str, str],
    links: list[Link],
    joints: list[Joint],
) -> tuple[dict[str, str], dict[str, str]]:
    """Attach physical solid bodies and frame links to their parent body links."""
    solid_links: dict[str, str] = {}
    for body in spec["bodies"]:
        for solid in body["solids"]:
            name = f"solid_{len(solid_links)}"
            if solid["name"] in solid_links:
                raise ValueError("Duplicate solid in specification")
            solid_links[solid["name"]] = name
            links.append(
                Link(
                    name=name,
                    inertia=Inertia.from_matrix(
                        np.asarray(solid["inertia_com_kg_m2"], dtype=float),
                        mass=float(solid["mass_kg"]),
                        center_of_mass=(
                            float(solid["com_m"][0]),
                            float(solid["com_m"][1]),
                            float(solid["com_m"][2]),
                        ),
                    ),
                )
            )
            joints.append(
                Joint(
                    name=f"fixed_{name}",
                    joint_type=JointType.FIXED,
                    parent=body_links[body["name"]],
                    child=name,
                    origin=_origin(solid["placement"]),
                    dynamics=JointDynamics(damping=0.0, friction=0.0),
                )
            )

    frame_links: dict[str, str] = {}
    for frame in spec["frames"]:
        if frame["name"] in frame_links:
            raise ValueError("Duplicate frame in specification")
        name = f"frame_{len(frame_links)}"
        frame_links[frame["name"]] = name
        links.append(Link(name=name, inertia=Inertia(0.0, 0.0, 0.0, mass=0.0)))
        joints.append(
            Joint(
                name=f"fixed_{name}",
                joint_type=JointType.FIXED,
                parent=body_links[frame["body"]],
                child=name,
                origin=_origin(frame["placement"]),
                dynamics=JointDynamics(damping=0.0, friction=0.0),
            )
        )
    return solid_links, frame_links


def _attach_contact_spheres(
    spec: Mapping[str, Any],
    body_links: Mapping[str, str],
    links: list[Link],
    joints: list[Joint],
) -> dict[str, dict[str, Any]]:
    """Attach foot contact sphere links to calcaneus body links."""
    contact_spheres: dict[str, dict[str, Any]] = {}
    for sphere in spec["contact"]["spheres"]:
        s_name = sphere["name"]
        link_name = f"contact_{s_name}"
        b_name = sphere["body"]
        if b_name not in body_links:
            raise ValueError(
                f"Contact sphere {s_name} references unknown body {b_name}"
            )
        radius = float(sphere["radius_m"])
        p_body = (
            float(sphere["position_m"][0]),
            float(sphere["position_m"][1]),
            float(sphere["position_m"][2]),
        )

        links.append(Link(name=link_name, inertia=Inertia(0.0, 0.0, 0.0, mass=0.0)))
        placement = np.eye(4)
        placement[:3, 3] = p_body
        joints.append(
            Joint(
                name=f"fixed_{link_name}",
                joint_type=JointType.FIXED,
                parent=body_links[b_name],
                child=link_name,
                origin=_origin(placement),
                dynamics=JointDynamics(damping=0.0, friction=0.0),
            )
        )
        contact_spheres[s_name] = {
            "link": link_name,
            "body": b_name,
            "radius_m": radius,
            "position_m": p_body,
        }
    return contact_spheres


def export_full_body_urdf(model_bytes: bytes) -> tuple[str, dict[str, Any]]:
    """Export the full-body specification to URDF with 41 scalar joints and contact links."""
    spec = json.loads(model_bytes)
    if (
        spec.get("schema_version") != "full-body-v1"
        or not isinstance(spec.get("closure"), dict)
        or not isinstance(spec.get("contact"), dict)
    ):
        raise ValueError("Invalid full-body specification for Drake URDF export")

    body_links = {body["name"]: f"body_{i}" for i, body in enumerate(spec["bodies"])}
    if len(body_links) != len(spec["bodies"]) or "world" not in body_links:
        raise ValueError("Invalid full-body body inventory")

    links: list[Link] = []
    joints: list[Joint] = []

    for link_name in body_links.values():
        links.append(Link(name=link_name, inertia=Inertia(0.0, 0.0, 0.0, mass=0.0)))

    coordinates: list[str] = []
    for joint in _order_full_body_joints(spec):
        _build_joint_primitives(
            joint,
            body_links[joint["parent"]],
            body_links[joint["child"]],
            coordinates,
            links,
            joints,
        )

    if len(coordinates) != len(spec["coordinate_order"]) or set(coordinates) != set(
        spec["coordinate_order"]
    ):
        raise ValueError("Full-body coordinate inventory was not preserved")

    solid_links, frame_links = _attach_solids_and_frames(
        spec, body_links, links, joints
    )
    contact_spheres = _attach_contact_spheres(spec, body_links, links, joints)

    closure = spec["closure"]
    for suffix in ("a", "b"):
        if closure[f"body_{suffix}"] not in body_links:
            raise ValueError(f"Unknown closure body_{suffix}")
        _origin(closure[f"placement_{suffix}"])

    xml = URDFWriter(expand_composite_joints=False).write(
        "full_body_golf", links, joints
    )

    sidecar = {
        "schema_version": 1,
        "requires_sidecar": True,
        "representation": "native-full-body-urdf-v1",
        "qualification": "full-body URDF export; dynamics via explicit continuous KKT adapter",
        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "urdf_sha256": hashlib.sha256(xml.encode("utf-8")).hexdigest(),
        "coordinate_order": spec["coordinate_order"],
        "body_links": body_links,
        "solid_links": solid_links,
        "frame_links": frame_links,
        "contact_spheres": contact_spheres,
        "closure": closure,
        "gravity_m_s2": spec["gravity_m_s2"],
        "limit_semantics": "restore-unbounded-before-dynamics",
    }
    return xml, sidecar
