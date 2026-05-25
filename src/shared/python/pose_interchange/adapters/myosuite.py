"""MyoSuite :class:`PoseConventionAdapter` implementation.

MyoSuite (https://sites.google.com/view/myosuite) is built on top of
MuJoCo and consumes MJCF (``.xml``) models, so its native ``q`` vector
follows the MuJoCo free-joint convention:
``[x, y, z, qw, qx, qy, qz, joint_0, joint_1, ...]`` with the quaternion
in **w-first** order and per-joint slots in radians for hinges.

This adapter therefore mirrors :class:`MujocoAdapter` almost exactly. The
only intentional difference is the engine identifier (``"myosuite"``) so
the registry can dispatch on it, plus the engine-side joint-name prefix
(``"myo_"``) used by the default mock layout. Real MyoSuite environments
ship per-environment joint-name conventions; when an actual model is
supplied callers must pass an explicit ``joint_layout`` via the
``model`` argument (mirroring the MuJoCo adapter's contract).

Per issue #6091. Coordinate-convention choices marked
``_TODO_review_convention`` below are guesses copied verbatim from the
MuJoCo sibling — they should hold because MyoSuite uses MuJoCo's free
joint, but the maintainer should confirm before any production use.
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

# _TODO_review_convention: MyoSuite uses MuJoCo's free-joint layout
# (``[x, y, z, qw, qx, qy, qz]``) so the 7-slot prefix matches the
# MuJoCo adapter exactly. Verified against MyoSuite docs that the
# underlying MJCF free joint is unchanged; confirm before production.
_PELVIS_PREFIX = 7  # [x, y, z, qw, qx, qy, qz]


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
        "MyoSuiteAdapter: 'model' must be None, a Mapping with 'joint_layout', "
        "or an object exposing a 'joint_layout' Mapping attribute"
    )


class MyoSuiteAdapter:
    """Adapter for MyoSuite MJCF (free-joint qpos is ``[xyz, quat_wxyz]``).

    MyoSuite wraps MuJoCo, so the byte-level layout is identical to the
    MuJoCo adapter. We keep a separate class only so the registry can
    dispatch on ``engine_name == "myosuite"`` and so future
    MyoSuite-specific quirks (muscle DoFs, joint-name conventions) can
    be handled here without polluting the MuJoCo path.
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
        # _TODO_review_convention: MuJoCo (and therefore MyoSuite)
        # stores the free-joint quaternion as w-first ``[qw, qx, qy, qz]``.
        # If an environment ever re-orders this, override via ``model``.
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
