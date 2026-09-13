"""Opt-in native spherical MJCF representation; kinematics only qualified.

Transform the canonical export's colocated same-body XYZ hinge triples. No
inertia, attachment, site, translation or weld is rebuilt. A ball contributes
four wxyz positions and three tangent velocities. Those velocities/efforts
are NOT native Euler rates/torques; a dynamics adapter is still required.
MuJoCo ball/slide and scalar-first quaternion references:
https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-joint
https://mujoco.readthedocs.io/en/stable/modeling.html#frame-orientations
"""

import hashlib
import json
import xml.etree.ElementTree as ET
from typing import Any

from src.engines.physics_engines.mujoco.python.native_mjcf import export_native_mjcf
from src.shared.python.pose_interchange.native_joint_state import (
    NativeJointStateAdapter,
)


def export_native_spherical_mjcf(model_bytes: bytes) -> tuple[str, dict[str, Any]]:
    """Replace exactly three native XYZ groups, retaining eighteen scalars.

    Model SHA256 identifies unchanged source physics; representation SHA256
    identifies different MJCF coordinates. Neither digest qualifies dynamics.
    Use NativeJointStateAdapter.export for quaternion conversion, then the
    compiled jnt_qposadr/jnt_dofadr for layouts; never assume native array order.
    """
    adapter = NativeJointStateAdapter(json.loads(model_bytes))
    if (
        len(adapter.coordinate_order) != 27
        or len(adapter.scalar_coordinates) != 18
        or len(adapter.groups) != 3
        or any(group.axes != "XYZ" for group in adapter.groups)
    ):
        raise ValueError(
            "Expected native 27-coordinate inventory with three XYZ groups"
        )
    canonical, metadata = export_native_mjcf(model_bytes)
    tree = ET.fromstring(canonical)
    bodies = {body.get("name"): body for body in tree.iter("body")}
    existing_names = {joint.get("name") for joint in tree.iter("joint")}
    mapping = {}
    for index, group in enumerate(adapter.groups):
        body = bodies[group.child_body]
        joints = body.findall("joint")
        rotations = [joint for joint in joints if joint.get("type") == "hinge"]
        if tuple(joint.get("name") for joint in rotations) != group.coordinates:
            raise ValueError("Native group must contain all rotations of one body")
        name = f"native_ball_{index}"
        if name in existing_names:
            raise ValueError("Spherical joint name collides with native inventory")
        # Canonical exporter gives these hinges identical zero positions and
        # passive parameters. Preserve those attributes, dropping hinge axis.
        attributes = dict(rotations[0].attrib)
        attributes.update(name=name, type="ball")
        attributes.pop("axis")
        position = list(body).index(rotations[0])
        for joint in rotations:
            body.remove(joint)
        body.insert(position, ET.Element("joint", attributes))
        mapping[group.name] = {
            "joint_name": name,
            "native_coordinates": list(group.coordinates),
            "axes": group.axes,
            "body": group.child_body,
            "qpos_components": ["w", "x", "y", "z"],
            "qpos_size": 4,
            "tangent_size": 3,
        }
    xml = ET.tostring(tree, encoding="unicode")
    digest = hashlib.sha256(xml.encode()).hexdigest()
    return xml, {
        **metadata,
        "canonical_mjcf_sha256": metadata["mjcf_sha256"],
        "mjcf_sha256": digest,
        "specification_sha256": adapter.specification_sha256,
        "representation": "native-spherical-mjcf-v1",
        "representation_sha256": digest,
        "ball_joints": mapping,
        "scalar_coordinates": list(adapter.scalar_coordinates),
        "expected_nq": 30,
        "expected_nv": 27,
        "execution": "kinematics-only; dynamics and stock mj_step unqualified",
    }
