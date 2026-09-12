"""Native-derived URDF tree with mandatory closure and execution metadata."""

import hashlib
import json
import sys
from typing import Any
import warnings

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
from src.shared.python.motion_matching.native_spec import order_native_tree


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


def export_native_urdf(model_bytes: bytes) -> tuple[str, dict[str, Any]]:
    """Export physical solids without synthetic mass; sidecar is mandatory.

    URDF finite-max limit placeholders accommodate parsers rejecting infinity.
    Qualified loaders must restore unbounded limits before dynamics, attach
    the weld, and apply gravity and native effort conventions from the sidecar.
    """
    spec = json.loads(model_bytes)
    if spec.get("schema_version") != 1:
        raise ValueError("Unsupported native schema")
    if not isinstance(spec.get("closure"), dict):
        raise ValueError("Native closure must be explicit")
    body_links = {body["name"]: f"body_{i}" for i, body in enumerate(spec["bodies"])}
    if len(body_links) != len(spec["bodies"]) or "world" not in body_links:
        raise ValueError("Invalid native body inventory")
    links, joints = [], []

    def massless(name: str) -> None:
        links.append(Link(name=name, inertia=Inertia(0.0, 0.0, 0.0, mass=0.0)))

    def fixed(parent: str, child: str, placement: Any) -> None:
        joints.append(
            Joint(
                name=f"fixed_{child}",
                joint_type=JointType.FIXED,
                parent=parent,
                child=child,
                origin=_origin(placement),
                dynamics=JointDynamics(damping=0.0, friction=0.0),
            )
        )

    for link in body_links.values():
        massless(link)
    coordinates = []
    for joint in order_native_tree(spec["joints"]):
        parent = body_links[joint["parent"]]
        placement = joint["parent_to_base"]
        for primitive in joint["primitives"]:
            kind, name = primitive["primitive"], primitive["coordinate"]
            if kind not in ("Px", "Py", "Pz", "Rx", "Ry", "Rz") or name in coordinates:
                raise ValueError("Duplicate or unsupported native coordinate")
            child = f"primitive_{len(coordinates)}"
            coordinates.append(name)
            massless(child)
            axis = tuple(float(i == "xyz".index(kind[1].lower())) for i in range(3))
            bound = sys.float_info.max
            joints.append(
                Joint(
                    name=name,
                    parent=parent,
                    child=child,
                    joint_type=JointType.PRISMATIC
                    if kind[0] == "P"
                    else JointType.REVOLUTE,
                    origin=_origin(placement),
                    axis=axis,
                    limits=JointLimits(-bound, bound, bound, bound),
                    dynamics=JointDynamics(damping=0.0, friction=0.0),
                )
            )
            parent, placement = child, np.eye(4)
        _origin(joint["child_to_follower"])
        fixed(
            parent,
            body_links[joint["child"]],
            np.linalg.inv(joint["child_to_follower"]),
        )
    if len(coordinates) != len(spec["coordinate_order"]) or set(coordinates) != set(
        spec["coordinate_order"]
    ):
        raise ValueError("Native coordinate inventory not preserved")
    solid_links = {}
    for body in spec["bodies"]:
        for solid in body["solids"]:
            name = f"solid_{len(solid_links)}"
            if solid["name"] in solid_links:
                raise ValueError("Duplicate native solid")
            solid_links[solid["name"]] = name
            links.append(
                Link(
                    name=name,
                    inertia=Inertia.from_matrix(
                        np.asarray(solid["inertia_com_kg_m2"]),
                        mass=solid["mass_kg"],
                        center_of_mass=tuple(solid["com_m"]),
                    ),
                )
            )
            fixed(body_links[body["name"]], name, solid["placement"])
    frame_links = {}
    for frame in spec["frames"]:
        if frame["name"] in frame_links:
            raise ValueError("Duplicate native frame")
        name = f"frame_{len(frame_links)}"
        frame_links[frame["name"]] = name
        massless(name)
        fixed(body_links[frame["body"]], name, frame["placement"])
    closure = spec["closure"]
    for suffix in ("a", "b"):
        if closure[f"body_{suffix}"] not in body_links:
            raise ValueError("Unknown closure body")
        _origin(closure[f"placement_{suffix}"])
    xml = URDFWriter(numeric_precision=17, expand_composite_joints=False).write(
        "native_golf", links, joints
    )
    sidecar = {
        "schema_version": 1,
        "requires_sidecar": True,
        "qualification": "native tree export only; reload and dynamics unqualified",
        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "urdf_sha256": hashlib.sha256(xml.encode()).hexdigest(),
        "coordinate_order": spec["coordinate_order"],
        "body_links": body_links,
        "solid_links": solid_links,
        "frame_links": frame_links,
        "closure": closure,
        "gravity_m_s2": spec["gravity_m_s2"],
        "limit_semantics": "restore-unbounded-before-dynamics",
        "effort_convention": "native primitive-conjugate; polynomial world forces require hip-base rotation",
        "native_joints": spec["joints"],
    }
    return xml, sidecar
