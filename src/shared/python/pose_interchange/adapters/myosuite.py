"""MyoSuite :class:`PoseConventionAdapter` implementation.

MyoSuite is a muscle-driven reinforcement-learning environment built on
top of MuJoCo.  Its models are MJCF files, so the underlying ``qpos``
layout is **identical** to MuJoCo's:

- Free-joint prefix: ``[x, y, z, qw, qx, qy, qz]`` (7 scalars, w-first).
- Per-joint slots: radians for hinge joints, placed after the prefix.

This adapter therefore shares ``_PELVIS_PREFIX = 7`` with
:class:`~src.shared.python.pose_interchange.adapters.mujoco.MujocoAdapter`
and delegates all quaternion/angle helpers to the same ``_base`` module.

The adapter operates in mock mode by default; if a real MyoSuite
``MjModel`` is supplied as ``model``, the per-joint layout must be
passed alongside as a ``Mapping[str, JointSlot]`` (we do not introspect
MyoSuite's qpos table here — the same restriction applies to the MuJoCo
adapter).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from src.shared.python.pose_interchange.adapters._base import (
    build_default_joint_layout,
    decode_joint_angles,
    encode_joint_angles,
    euler_xyz_deg_to_quat_wxyz,
    quat_wxyz_to_euler_xyz_deg,
)
from src.shared.python.pose_interchange.canonical import (
    CONVENTION_TAG,
    CanonicalPose,
)
from src.shared.python.pose_interchange.protocol import JointSlot

# MyoSuite uses MuJoCo's coordinate convention: free-joint qpos prefix is
# [x, y, z, qw, qx, qy, qz] — 7 scalars, quaternion w-first.
_PELVIS_PREFIX = 7  # identical to MuJoCo free-joint prefix


def _layout_from_model(model: Any | None) -> Mapping[str, JointSlot]:
    if model is None:
        return build_default_joint_layout(
            base_offset=_PELVIS_PREFIX, units="rad", sign=1, name_prefix="myo_"
        )
    if hasattr(model, "joint_layout") and isinstance(model.joint_layout, Mapping):
        return model.joint_layout
    if isinstance(model, Mapping) and "joint_layout" in model:
        return model["joint_layout"]
    raise TypeError(
        "MyosuiteAdapter: 'model' must be None, a Mapping with 'joint_layout', "
        "or an object exposing a 'joint_layout' Mapping attribute"
    )


class MyosuiteAdapter:
    """Adapter for MyoSuite MJCF (free-joint qpos is ``[xyz, quat_wxyz]``).

    MyoSuite is backed by MuJoCo, so the qpos coordinate convention is
    identical: position prefix ``[x, y, z]`` followed by a w-first unit
    quaternion ``[qw, qx, qy, qz]``, then per-joint hinge angles in
    radians.
    """

    engine_name: str = "myosuite"

    def joint_layout(self, model: Any | None = None) -> Mapping[str, JointSlot]:
        return _layout_from_model(model)

    def from_canonical(
        self,
        pose: CanonicalPose,
        *,
        model: Any | None = None,
    ) -> npt.NDArray[np.float64]:
        if pose.convention_tag != CONVENTION_TAG:
            raise ValueError(
                f"MyosuiteAdapter.from_canonical: unsupported convention "
                f"{pose.convention_tag!r}"
            )
        layout = _layout_from_model(model)
        max_idx = max(
            (slot.start_index + slot.length for slot in layout.values()),
            default=_PELVIS_PREFIX,
        )
        size = max(_PELVIS_PREFIX, max_idx)
        q = np.zeros(size, dtype=float)
        q[0:3] = pose.pelvis_translation_m
        q[3:7] = euler_xyz_deg_to_quat_wxyz(pose.pelvis_rotation_xyz_deg)
        encode_joint_angles(pose.joint_angles_deg, layout, q)
        return q

    def to_canonical(
        self,
        engine_q: npt.ArrayLike,
        *,
        model: Any | None = None,
    ) -> CanonicalPose:
        q = np.asarray(engine_q, dtype=float)
        if q.ndim != 1 or q.shape[0] < _PELVIS_PREFIX:
            raise ValueError(
                f"MyosuiteAdapter.to_canonical: expected 1-D q with at least "
                f"{_PELVIS_PREFIX} entries, got shape {q.shape}"
            )
        layout = _layout_from_model(model)
        translation = q[0:3].copy()
        rotation_deg = quat_wxyz_to_euler_xyz_deg(q[3:7])
        joint_angles = decode_joint_angles(q, layout)
        return CanonicalPose(
            pelvis_translation_m=translation,
            pelvis_rotation_xyz_deg=rotation_deg,
            joint_angles_deg=joint_angles,
        )


# Backward-compatible spelling for callers that preserve the MyoSuite brand
# casing in class names.
MyoSuiteAdapter = MyosuiteAdapter


__all__ = ["MyoSuiteAdapter", "MyosuiteAdapter"]
