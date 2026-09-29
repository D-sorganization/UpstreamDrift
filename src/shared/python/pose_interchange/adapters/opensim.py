"""OpenSim :class:`PoseConventionAdapter` implementation.

OpenSim ``.osim`` BodyKinematics reports the pelvis as
``[x, y, z, rotX, rotY, rotZ]`` in **degrees** with intrinsic XYZ
Euler — the same Euler convention as the canonical pose, so the
pelvis pose is a direct copy. Joint angles are exposed as
``CoordinateSet`` entries; canonical coordinate values are returned in
degrees by ``BodyKinematics`` so this adapter keeps the per-joint slots
in degrees as well.

Several ``.osim`` models invert the Y axis for shoulder external
rotation; that flip is captured per-joint via :class:`JointSlot.sign`
rather than scattered through the function bodies.

Mock-mode by default — pass a ``Mapping`` (or any object exposing a
``joint_layout`` attribute) as ``model`` to override the default mock
layout.
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
)
from src.shared.python.pose_interchange.canonical import (
    CONVENTION_TAG,
    CanonicalPose,
)
from src.shared.python.pose_interchange.protocol import JointSlot

_PELVIS_PREFIX = 6  # [x, y, z, rotX_deg, rotY_deg, rotZ_deg]


def _layout_from_model(model: Any | None) -> Mapping[str, JointSlot]:
    if model is None:
        # OpenSim coordinates are reported in degrees.
        return build_default_joint_layout(
            base_offset=_PELVIS_PREFIX, units="deg", sign=1, name_prefix="osim_"
        )
    if hasattr(model, "joint_layout") and isinstance(model.joint_layout, Mapping):
        return model.joint_layout
    if isinstance(model, Mapping) and "joint_layout" in model:
        return model["joint_layout"]
    raise TypeError(
        "OpenSimAdapter: 'model' must be None, a Mapping with 'joint_layout', "
        "or an object exposing a 'joint_layout' Mapping attribute"
    )


class OpenSimAdapter:
    """Adapter for OpenSim ``.osim`` BodyKinematics (XYZ Euler in deg)."""

    engine_name: str = "opensim"

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
                f"OpenSimAdapter.from_canonical: unsupported convention "
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
        # OpenSim BodyKinematics is XYZ Euler in degrees — direct copy.
        q[3:6] = pose.pelvis_rotation_xyz_deg
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
                f"OpenSimAdapter.to_canonical: expected 1-D q with at least "
                f"{_PELVIS_PREFIX} entries, got shape {q.shape}"
            )
        layout = _layout_from_model(model)
        translation = q[0:3].copy()
        rotation_deg = q[3:6].copy()
        joint_angles = decode_joint_angles(q, layout)
        return CanonicalPose(
            pelvis_translation_m=translation,
            pelvis_rotation_xyz_deg=rotation_deg,
            joint_angles_deg=joint_angles,
        )
