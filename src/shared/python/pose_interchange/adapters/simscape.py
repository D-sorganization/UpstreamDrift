"""Simscape :class:`PoseConventionAdapter` implementation.

Simscape Multibody parameter-bus joints carry Simulink.Parameter names
that already match :data:`REFERENCE_GOLFER_FIELDS` 1:1 (e.g.
``HipStartPositionX``, ``LSStartPositionZ``). Values are in **degrees**.

This adapter is therefore the closest thing to an identity adapter: the
canonical joint-angle dict is laid out directly into the engine ``q``
vector with no name remapping and no degree/radian conversion. The
pelvis pose is appended as a 6-DOF prefix in ``[xyz, rot_xyz_deg]``
order (the parameter bus also exposes ``HipStartPositionX/Y/Z`` for the
pelvis joint axes; we keep the SE(3) world-frame translation as the
prefix and let the joint-angle slots carry the in-pelvis rotations).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from src.shared.python.motion_matching.diagnostics.reference_pose import (
    REFERENCE_GOLFER_FIELDS,
)
from src.shared.python.pose_interchange.adapters._base import (
    decode_joint_angles,
    encode_joint_angles,
)
from src.shared.python.pose_interchange.canonical import (
    CONVENTION_TAG,
    CanonicalPose,
)
from src.shared.python.pose_interchange.protocol import JointSlot

_PELVIS_PREFIX = 6  # [x_m, y_m, z_m, rot_x_deg, rot_y_deg, rot_z_deg]


def _default_layout() -> dict[str, JointSlot]:
    """Identity layout: engine names match canonical names exactly."""
    layout: dict[str, JointSlot] = {}
    for index, canonical in enumerate(REFERENCE_GOLFER_FIELDS):
        layout[canonical] = JointSlot(
            canonical_name=canonical,
            engine_name=canonical,
            start_index=_PELVIS_PREFIX + index,
            length=1,
            units="deg",
            sign=1,
        )
    return layout


def _layout_from_model(model: Any | None) -> Mapping[str, JointSlot]:
    if model is None:
        return _default_layout()
    if hasattr(model, "joint_layout") and isinstance(model.joint_layout, Mapping):
        return model.joint_layout
    if isinstance(model, Mapping) and "joint_layout" in model:
        return model["joint_layout"]
    raise TypeError(
        "SimscapeAdapter: 'model' must be None, a Mapping with 'joint_layout', "
        "or an object exposing a 'joint_layout' Mapping attribute"
    )


class SimscapeAdapter:
    """Adapter for Simscape Multibody (degrees, names match canonical)."""

    engine_name: str = "simscape"

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
                f"SimscapeAdapter.from_canonical: unsupported convention "
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
                f"SimscapeAdapter.to_canonical: expected 1-D q with at least "
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
