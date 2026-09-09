"""Compile MJCF once and retain its hinge-tree geometry for reference fitting."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from src.motion_capture.reconstruct.model import Joint, ModelSpec
from src.motion_capture.reconstruct.model.registry import RegisteredModel
from src.motion_capture.reconstruct.model.session import LandmarkMap
from src.shared.python.model_generation.core.types import Joint as SourceJoint
from src.shared.python.model_generation.core.types import JointLimits, JointType, Origin

from .urdf_models import _joint


def load_mjcf_model(
    path: Path,
    *,
    name: str,
    landmark_map: Mapping[str, str | tuple[str, ...]],
) -> RegisteredModel:
    """Compile a MuJoCo hinge tree and preserve local origins and pivot offsets.

    Preconditions: MuJoCo installed, no equality/slide/ball constraints, one
    body tree. Postcondition: native body origins are fit landmarks; intermediate
    pivots remain in the output tree. Root q is a scene-placement Euler pose;
    each hinge angle is displacement from native qpos0, not native qpos itself.
    """
    import mujoco

    native = mujoco.MjModel.from_xml_path(str(path))
    if native.neq:
        raise ValueError(
            "MJCF equality constraints require a coupled-coordinate adapter"
        )
    body_names = {
        i: mujoco.mj_id2name(native, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
        for i in range(1, native.nbody)
    }
    if not landmark_map or set(landmark_map) - set(body_names.values()):
        raise ValueError("MJCF landmark mapping must name existing bodies")
    if sum(int(native.body_parentid[i]) == 0 for i in body_names) != 1:
        raise ValueError("MJCF must contain one body tree")
    world_name = "__reference_world__"
    if world_name in body_names.values():
        raise ValueError("MJCF body uses a reserved reference name")
    joints = [Joint(world_name, None, axes="xyz", landmark=False)]
    lengths: dict[str, float] = {}

    def append(source: SourceJoint, observed: bool = False) -> None:
        converted = _joint(source, observed)
        joints.append(converted)
        if converted.length:
            lengths[converted.length] = float(np.linalg.norm(source.origin.xyz))

    def fixed(
        child: str, parent: str, offset: np.ndarray, angles: np.ndarray | None = None
    ) -> None:
        append(
            SourceJoint(
                child,
                JointType.FIXED,
                parent,
                child,
                Origin(
                    tuple(offset),
                    tuple(angles) if angles is not None else (0.0, 0.0, 0.0),
                ),
            ),
            child in landmark_map,
        )

    for body_id, body_name in body_names.items():
        parent = body_names.get(int(native.body_parentid[body_id]), world_name)
        quat = native.body_quat[body_id]
        angles = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_euler("xyz")
        previous = f"{body_name}::origin"
        fixed(previous, parent, native.body_pos[body_id], angles)
        start = int(native.body_jntadr[body_id])
        for joint_id in range(start, start + int(native.body_jntnum[body_id])):
            kind = int(native.jnt_type[joint_id])
            if kind == int(mujoco.mjtJoint.mjJNT_FREE) and parent == world_name:
                continue
            if kind != int(mujoco.mjtJoint.mjJNT_HINGE):
                raise ValueError(
                    "MJCF fitting supports hinges and a root free joint only"
                )
            pivot = f"{body_name}::pivot_{joint_id}"
            origin = Origin(tuple(native.jnt_pos[joint_id]))
            ref = float(native.qpos0[native.jnt_qposadr[joint_id]])
            bounds = native.jnt_range[joint_id] - ref
            limits = (
                JointLimits(float(bounds[0]), float(bounds[1]))
                if native.jnt_limited[joint_id]
                else None
            )
            joint_kind = JointType.REVOLUTE if limits else JointType.CONTINUOUS
            append(
                SourceJoint(
                    pivot,
                    joint_kind,
                    previous,
                    pivot,
                    origin,
                    tuple(native.jnt_axis[joint_id]),
                    limits,
                )
            )
            previous = f"{body_name}::return_{joint_id}"
            fixed(previous, pivot, -native.jnt_pos[joint_id])
        fixed(body_name, previous, np.zeros(3))
    spec = ModelSpec(name, tuple(joints), lengths)
    return RegisteredModel(
        name,
        spec,
        LandmarkMap(dict(landmark_map)),
        (),
        f"Compiled MuJoCo kinematic tree from {path.name}; no dynamics claim.",
    )
