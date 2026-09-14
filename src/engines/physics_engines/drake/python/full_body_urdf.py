"""Export a full-body model specification to a URDF document with shared contact.

Embeds the upper-body rigid body tree and appends lower limbs, contact sphere links/joints,
and sidecar metadata required for explicit KKT dynamics execution in Drake.
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
    Geometry,
    Inertia,
    Joint,
    JointDynamics,
    JointLimits,
    JointType,
    Link,
    Origin,
)
from src.shared.python.motion_matching.full_body_spec import (
    order_full_body_joints,
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
        raise ValueError("Invalid native rigid transform")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Gimbal lock detected.*", category=UserWarning
        )
        rpy = Rotation.from_matrix(matrix[:3, :3]).as_euler("xyz")
    if not np.allclose(
        Rotation.from_euler("xyz", rpy).as_matrix(), matrix[:3, :3], atol=1e-12, rtol=0
    ):
        raise ValueError("Native transform cannot be preserved in URDF RPY")
    return Origin(xyz=tuple(matrix[:3, 3]), rpy=tuple(rpy))


def _build_joint_primitives(
    joint: Mapping[str, Any],
    parent_link: str,
    coordinates: list[str],
    links: list[Link],
    joints: list[Joint],
) -> str:
    """Add intermediate massless links and 1-DOF joints for each primitive."""
    current_parent = parent_link
    placement = joint["parent_to_base"]
    for primitive in joint["primitives"]:
        kind, name = primitive["primitive"], primitive["coordinate"]
        if kind not in ("Px", "Py", "Pz", "Rx", "Ry", "Rz") or name in coordinates:
            raise ValueError("Duplicate or unsupported native coordinate")
        child_link = f"primitive_{len(coordinates)}"
        coordinates.append(name)
        links.append(Link(name=child_link, inertia=Inertia(0.0, 0.0, 0.0, mass=0.0)))
        axis_index = "xyz".index(kind[1].lower())
        axis = (
            float(axis_index == 0),
            float(axis_index == 1),
            float(axis_index == 2),
        )
        bound = sys.float_info.max
        joints.append(
            Joint(
                name=name,
                parent=current_parent,
                child=child_link,
                joint_type=JointType.PRISMATIC
                if kind[0] == "P"
                else JointType.REVOLUTE,
                origin=_origin(placement),
                axis=axis,
                limits=JointLimits(-bound, bound, bound, bound),
                dynamics=JointDynamics(damping=0.0, friction=0.0),
            )
        )
        current_parent = child_link
        placement = np.eye(4)
    return current_parent


def _build_kinematic_tree(
    spec: Mapping[str, Any],
    body_links: Mapping[str, str],
    links: list[Link],
    joints: list[Joint],
) -> None:
    """Construct tree kinematics and verify coordinate inventory."""
    coordinates: list[str] = []
    for joint in order_full_body_joints(spec):
        parent = body_links[joint["parent"]]
        last_primitive = _build_joint_primitives(
            joint, parent, coordinates, links, joints
        )
        _origin(joint["child_to_follower"])
        joints.append(
            Joint(
                name=f"fixed_{body_links[joint['child']]}",
                joint_type=JointType.FIXED,
                parent=last_primitive,
                child=body_links[joint["child"]],
                origin=_origin(np.linalg.inv(joint["child_to_follower"])),
                dynamics=JointDynamics(damping=0.0, friction=0.0),
            )
        )
    if len(coordinates) != len(spec["coordinate_order"]) or set(coordinates) != set(
        spec["coordinate_order"]
    ):
        raise ValueError("Coordinate inventory not preserved")


def _attach_solids_and_frames(
    spec: Mapping[str, Any],
    body_links: Mapping[str, str],
    links: list[Link],
    joints: list[Joint],
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Attach physical solids, marker-tracking frames, and foot contact spheres."""
    solid_links: dict[str, str] = {}
    for body in spec["bodies"]:
        for solid in body["solids"]:
            s_name = f"solid_{len(solid_links)}"
            if solid["name"] in solid_links:
                raise ValueError("Duplicate full-body solid")
            solid_links[solid["name"]] = s_name
            links.append(
                Link(
                    name=s_name,
                    inertia=Inertia.from_matrix(
                        np.asarray(solid["inertia_com_kg_m2"]),
                        mass=solid["mass_kg"],
                        center_of_mass=tuple(solid["com_m"]),
                    ),
                )
            )
            joints.append(
                Joint(
                    name=f"fixed_{s_name}",
                    joint_type=JointType.FIXED,
                    parent=body_links[body["name"]],
                    child=s_name,
                    origin=_origin(solid["placement"]),
                    dynamics=JointDynamics(damping=0.0, friction=0.0),
                )
            )

    frame_links: dict[str, str] = {}
    for frame in spec["frames"]:
        if frame["name"] in frame_links:
            raise ValueError("Duplicate full-body frame")
        f_name = f"frame_{len(frame_links)}"
        frame_links[frame["name"]] = f_name
        links.append(Link(name=f_name, inertia=Inertia(0.0, 0.0, 0.0, mass=0.0)))
        joints.append(
            Joint(
                name=f"fixed_{f_name}",
                joint_type=JointType.FIXED,
                parent=body_links[frame["body"]],
                child=f_name,
                origin=_origin(frame["placement"]),
                dynamics=JointDynamics(damping=0.0, friction=0.0),
            )
        )

    contact_links: dict[str, str] = {}
    for sphere in spec["contact"]["spheres"]:
        s_name, b_name = sphere["name"], sphere["body"]
        c_link = f"contact_{len(contact_links)}_{s_name}"
        contact_links[s_name] = c_link
        t_sphere = np.eye(4)
        t_sphere[:3, 3] = sphere["position_m"]
        links.append(
            Link(
                name=c_link,
                inertia=Inertia(0.0, 0.0, 0.0, mass=0.0),
                collision_geometry=Geometry.sphere(float(sphere["radius_m"])),
            )
        )
        joints.append(
            Joint(
                name=f"contact_joint_{c_link}",
                joint_type=JointType.FIXED,
                parent=body_links[b_name],
                child=c_link,
                origin=_origin(t_sphere),
                dynamics=JointDynamics(damping=0.0, friction=0.0),
            )
        )

    return solid_links, frame_links, contact_links


def export_full_body_urdf(model_bytes: bytes) -> tuple[str, dict[str, Any]]:
    """Export full-body physical solids and foot contact spheres into URDF XML and sidecar."""
    spec = json.loads(model_bytes)
    if spec.get("schema_version") != "full-body-v1":
        raise ValueError("Unsupported full-body schema")
    if not isinstance(spec.get("closure"), dict):
        raise ValueError("Native closure must be explicit")

    body_links = {body["name"]: f"body_{i}" for i, body in enumerate(spec["bodies"])}
    if len(body_links) != len(spec["bodies"]) or "world" not in body_links:
        raise ValueError("Invalid full-body body inventory")

    links: list[Link] = [
        Link(name=link_name, inertia=Inertia(0.0, 0.0, 0.0, mass=0.0))
        for link_name in body_links.values()
    ]
    joints: list[Joint] = []

    _build_kinematic_tree(spec, body_links, links, joints)
    solid_links, frame_links, contact_links = _attach_solids_and_frames(
        spec, body_links, links, joints
    )

    closure = spec["closure"]
    for suffix in ("a", "b"):
        if closure[f"body_{suffix}"] not in body_links:
            raise ValueError("Unknown closure body")
        _origin(closure[f"placement_{suffix}"])

    xml = URDFWriter(expand_composite_joints=False).write(
        "full_body_drake", links, joints
    )

    sidecar = {
        "schema_version": 1,
        "requires_sidecar": True,
        "qualification": "full-body extension of native-derived geometry; dynamics unqualified",
        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "urdf_sha256": hashlib.sha256(xml.encode("utf-8")).hexdigest(),
        "coordinate_order": spec["coordinate_order"],
        "body_links": body_links,
        "solid_links": solid_links,
        "frame_links": frame_links,
        "contact_links": contact_links,
        "contact": spec["contact"],
        "closure": closure,
        "gravity_m_s2": spec["gravity_m_s2"],
        "limit_semantics": "restore-unbounded-before-dynamics",
        "effort_convention": "native primitive-conjugate",
        "native_joints": spec["joints"],
    }
    return xml, sidecar
