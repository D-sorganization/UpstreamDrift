"""MuJoCo :class:`PoseConventionAdapter` implementation.

MyoSuite's MJCF free-joint ``qpos`` prefix is ``[x, y, z, qw, qx, qy, qz]``
— quaternion **w-first**. Per-joint slots that follow are radians for
hinges. This adapter mirrors that convention exactly: the canonical
``pelvis_rotation_xyz_deg`` is converted to a unit quaternion in
``[w, x, y, z]`` order before being placed into ``qpos``.

The adapter operates in mock mode by default; if a real MuJoCo
``mjModel`` is supplied as ``model``, the per-joint layout must be
passed alongside as a ``Mapping[str, JointSlot]`` (we do not attempt to
introspect MuJoCo's qpos table here).
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

_PELVIS_PREFIX = 7  # [x, y, z, qw, qx, qy, qz]


def _layout_from_model(model: Any | None) -> Mapping[str, JointSlot]:
    if model is None:
        return build_default_joint_layout(
            base_offset=_PELVIS_PREFIX, units="rad", sign=1, name_prefix="mj_"
        )
    if hasattr(model, "joint_layout") and isinstance(model.joint_layout, Mapping):
        return model.joint_layout
    if isinstance(model, Mapping) and "joint_layout" in model:
        return model["joint_layout"]
    raise TypeError(
        "MyoSuiteAdapter: 'model' must be None, a Mapping with 'joint_layout', "
        "or an object exposing a 'joint_layout' Mapping attribute"
    )


class MyoSuiteAdapter:
    """Adapter for MuJoCo MJCF (free-joint qpos is ``[xyz, quat_wxyz]``)."""

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
                f"MyoSuiteAdapter.from_canonical: unsupported convention "
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
                f"MyoSuiteAdapter.to_canonical: expected 1-D q with at least "
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


# trigger ci
