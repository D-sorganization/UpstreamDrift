"""Drake :class:`PoseConventionAdapter` implementation.

Drake reads URDF and represents a free-flyer's ``q`` vector as
``[x, y, z, roll, pitch, yaw]`` followed by per-joint scalar positions
in radians. This adapter assumes the canonical golfer layout: a 6-DOF
pelvis prefix in ``[xyz, rpy]`` order, then one revolute slot per
canonical joint.

The implementation is mock-mode by design: the per-joint layout comes
from a hardcoded fixture so the adapter works in CI without a Drake
wheel installed. If a real Drake ``MultibodyPlant`` is supplied as
``model``, the layout mapping must be passed explicitly via the
``model.joint_layout`` attribute (a dict-of-:class:`JointSlot`); we do
not try to introspect Drake's plant from this adapter.

Drake's RPY convention here is intrinsic XYZ in degrees-on-the-canonical
side and radians-on-the-engine side; we do the deg <-> rad conversion at
the boundary so the canonical-pose contract stays in degrees.
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

_PELVIS_PREFIX = 6  # [x, y, z, roll, pitch, yaw]


def _layout_from_model(model: Any | None) -> Mapping[str, JointSlot]:
    """Return the joint layout, falling back to the mock fixture.

    If *model* is None, returns the hardcoded mock layout.
    If *model* exposes a ``joint_layout`` mapping attribute, returns
    that.  Anything else is rejected with :class:`TypeError`.
    """
    if model is None:
        return build_default_joint_layout(
            base_offset=_PELVIS_PREFIX, units="rad", sign=1, name_prefix="drake_"
        )
    if hasattr(model, "joint_layout") and isinstance(model.joint_layout, Mapping):
        return model.joint_layout
    if isinstance(model, Mapping) and "joint_layout" in model:
        return model["joint_layout"]
    raise TypeError(
        "DrakeAdapter: 'model' must be None, a Mapping with 'joint_layout', "
        "or an object exposing a 'joint_layout' Mapping attribute"
    )


class DrakeAdapter:
    """Adapter for Drake URDF/SDF (free-flyer ``q`` is ``[xyz, rpy]``)."""

    engine_name: str = "drake"

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
                f"DrakeAdapter.from_canonical: unsupported convention "
                f"{pose.convention_tag!r}"
            )
        layout = _layout_from_model(model)
        size = _PELVIS_PREFIX + max(
            (slot.start_index + slot.length for slot in layout.values()),
            default=_PELVIS_PREFIX,
        )
        # The layout's start_index is already absolute (>= _PELVIS_PREFIX), so
        # we don't need to subtract the prefix.
        size = max(size, _PELVIS_PREFIX)
        max_idx = max(
            (slot.start_index + slot.length for slot in layout.values()),
            default=_PELVIS_PREFIX,
        )
        size = max(size, max_idx)
        q = np.zeros(size, dtype=float)
        q[0:3] = pose.pelvis_translation_m
        q[3:6] = np.radians(pose.pelvis_rotation_xyz_deg)
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
                f"DrakeAdapter.to_canonical: expected 1-D q with at least "
                f"{_PELVIS_PREFIX} entries, got shape {q.shape}"
            )
        layout = _layout_from_model(model)
        translation = q[0:3].copy()
        rotation_deg = np.degrees(q[3:6])
        joint_angles = decode_joint_angles(q, layout)
        return CanonicalPose(
            pelvis_translation_m=translation,
            pelvis_rotation_xyz_deg=rotation_deg,
            joint_angles_deg=joint_angles,
        )
