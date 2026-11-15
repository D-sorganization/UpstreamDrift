"""Hand skeleton poses for skinned grip visuals."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from src.shared.python.model_appearance.hand_backends.joint_angles import (
    HandJointAngles,
)

Array = NDArray[np.float64]

_FINGERS = ("index", "middle", "ring", "little")
_SPREAD_SIGN = {"index": 0.6, "middle": 0.2, "ring": -0.2, "little": -0.6}


def _rot_y(deg: float) -> Array:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array(
        [
            [c, 0.0, s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-s, 0.0, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def _rot_z(deg: float) -> Array:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array(
        [
            [c, -s, 0.0, 0.0],
            [s, c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def _trans(x: float, y: float, z: float) -> Array:
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def _about_joint(rest: Array, rot: Array) -> Array:
    p = rest[:3, 3]
    t = _trans(p[0], p[1], p[2]) @ rot @ _trans(-p[0], -p[1], -p[2])
    return t @ rest


def bone_poses_from_joint_angles(
    angles: HandJointAngles,
    *,
    bone_names: tuple[str, ...],
    rest_poses: Array,
) -> dict[str, Array]:
    """Map anatomical flexion onto the asset's rest skeleton."""
    poses: dict[str, Array] = {}
    spread_deg = math.degrees(angles.metacarpal_spread_rad)
    for i, name in enumerate(bone_names):
        poses[name] = rest_poses[i].copy()

    if "wrist" in poses:
        poses["wrist"] = _about_joint(poses["wrist"], _rot_z(spread_deg * 0.15))

    for finger in _FINGERS:
        flex = angles.fingers[finger]
        meta = f"meta_{finger}"
        prox = f"prox_{finger}"
        mid = f"mid_{finger}"
        dist = f"dist_{finger}"
        if meta in poses:
            poses[meta] = _about_joint(
                poses[meta],
                _rot_z(spread_deg * _SPREAD_SIGN[finger])
                @ _rot_y(-math.degrees(flex.mcp_rad) * 0.35),
            )
        if prox in poses:
            poses[prox] = _about_joint(poses[prox], _rot_y(-math.degrees(flex.mcp_rad)))
        if mid in poses:
            poses[mid] = _about_joint(poses[mid], _rot_y(-math.degrees(flex.pip_rad)))
        if dist in poses:
            poses[dist] = _about_joint(poses[dist], _rot_y(-math.degrees(flex.dip_rad)))

    if "thumb_meta" in poses:
        poses["thumb_meta"] = _about_joint(
            poses["thumb_meta"], _rot_y(-math.degrees(angles.thumb_mcp_rad))
        )
    if "thumb_dist" in poses:
        poses["thumb_dist"] = _about_joint(
            poses["thumb_dist"], _rot_y(-math.degrees(angles.thumb_ip_rad))
        )
    if "palm_cup" in poses:
        poses["palm_cup"] = _about_joint(
            poses["palm_cup"], _rot_y(-math.degrees(angles.palm_arch_rad) * 0.8)
        )
    return poses
