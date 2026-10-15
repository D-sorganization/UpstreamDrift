"""Pinocchio :class:`PoseConventionAdapter` implementation.

Pinocchio's free-flyer ``q`` prefix is ``[x, y, z, qx, qy, qz, qw]`` —
quaternion **w-last**, the opposite of MuJoCo. Per-joint slots are
radians.

Mock-mode by default; if a real Pinocchio ``Model`` is supplied as
``model``, pass the per-joint :class:`JointSlot` map alongside (we do
not introspect Pinocchio's joint table from this adapter).

"Local stub" status (Issue #4095)
---------------------------------

This adapter is **intentionally** a pure-Python convention adapter: it
performs canonical <-> engine ``q`` conversions (w-first <-> w-last
quaternion reordering, joint-angle encode/decode) without depending on
the Pinocchio wheel. That keeps the adapter layer importable in every
environment, mock or real. The *live* engine kinematics (forward
kinematics, ``aba`` dynamics step, frame placements) live in
:class:`~src.shared.python.pose_interchange.services.pinocchio.PinocchioKinematicsService`,
which lazily imports the heavy Stack-of-Tasks ``pinocchio`` wheel and
falls back to a mock when it is absent. So the "local stub" label
applies only to this conversion layer by design -- it is not an
incomplete bridge. The real service path is exercised by
:mod:`tests.integration.pose_interchange.services.test_pinocchio_real`.
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
    quat_wxyz_to_xyzw,
    quat_xyzw_to_wxyz,
)
from src.shared.python.pose_interchange.canonical import (
    CONVENTION_TAG,
    CanonicalPose,
)
from src.shared.python.pose_interchange.dynamic_transforms import (
    acceleration_from_canonical,
    acceleration_to_canonical,
    generalized_force_from_canonical,
    generalized_force_to_canonical,
    velocity_from_canonical,
    velocity_to_canonical,
)
from src.shared.python.pose_interchange.protocol import JointSlot

_PELVIS_PREFIX = 7  # [x, y, z, qx, qy, qz, qw]


def _layout_from_model(model: Any | None) -> Mapping[str, JointSlot]:
    if model is None:
        return build_default_joint_layout(
            base_offset=_PELVIS_PREFIX, units="rad", sign=1, name_prefix="pin_"
        )
    if hasattr(model, "joint_layout") and isinstance(model.joint_layout, Mapping):
        return model.joint_layout
    if isinstance(model, Mapping) and "joint_layout" in model:
        return model["joint_layout"]
    raise TypeError(
        "PinocchioAdapter: 'model' must be None, a Mapping with 'joint_layout', "
        "or an object exposing a 'joint_layout' Mapping attribute. A real "
        "Pinocchio 'Model' is not introspected here; supply the per-joint "
        "layout explicitly, e.g. model={'joint_layout': "
        "build_default_joint_layout(...)} or attach a 'joint_layout' Mapping "
        f"attribute to your model object (got {type(model).__name__})."
    )


class PinocchioAdapter:
    """Adapter for Pinocchio (free-flyer ``q`` is ``[xyz, quat_xyzw]``)."""

    engine_name: str = "pinocchio"

    def joint_layout(self, model: Any | None = None) -> Mapping[str, JointSlot]:
        """Return the per-joint ``q``-vector layout for ``model``.

        ``model`` must be one of:

        * ``None`` — use the default golfer layout (mock mode);
        * a ``Mapping`` containing a ``"joint_layout"`` key; or
        * an object exposing a ``joint_layout`` ``Mapping`` attribute.

        A **raw Pinocchio ``Model`` is not introspected** by this adapter: pass
        the per-joint :class:`JointSlot` map alongside it (wrap it as
        ``{"joint_layout": ...}`` or attach a ``joint_layout`` attribute).
        Anything else raises :class:`TypeError` with the remediation.
        """

        return _layout_from_model(model)

    def from_canonical(
        self,
        pose: CanonicalPose,
        *,
        model: Any | None = None,
    ) -> npt.NDArray[np.float64]:
        if pose.convention_tag != CONVENTION_TAG:
            raise ValueError(
                f"PinocchioAdapter.from_canonical: unsupported convention "
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
        wxyz = euler_xyz_deg_to_quat_wxyz(pose.pelvis_rotation_xyz_deg)
        q[3:7] = quat_wxyz_to_xyzw(wxyz)
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
                f"PinocchioAdapter.to_canonical: expected 1-D q with at least "
                f"{_PELVIS_PREFIX} entries, got shape {q.shape}"
            )
        layout = _layout_from_model(model)
        translation = q[0:3].copy()
        wxyz = quat_xyzw_to_wxyz(q[3:7])
        rotation_deg = quat_wxyz_to_euler_xyz_deg(wxyz)
        joint_angles = decode_joint_angles(q, layout)
        return CanonicalPose(
            pelvis_translation_m=translation,
            pelvis_rotation_xyz_deg=rotation_deg,
            joint_angles_deg=joint_angles,
        )

    def velocity_map(
        self,
        q: npt.ArrayLike,
        v: npt.ArrayLike,
        *,
        to_canonical: bool = False,
        model: Any | None = None,
    ) -> npt.NDArray[np.float64]:
        """Map velocity between canonical-v2 and Pinocchio native layouts (NSX-50)."""
        layout = _layout_from_model(model)
        if to_canonical:
            return velocity_to_canonical(
                q,
                v,
                base_convention="free_flyer_xyzw",
                q_is_canonical=False,
                layout=layout,
                base_offset=_PELVIS_PREFIX,
            )
        return velocity_from_canonical(
            q,
            v,
            base_convention="free_flyer_xyzw",
            q_is_canonical=True,
            layout=layout,
            base_offset=_PELVIS_PREFIX,
        )

    def velocity_to_canonical(
        self,
        q: npt.ArrayLike,
        v_engine: npt.ArrayLike,
        *,
        model: Any | None = None,
    ) -> npt.NDArray[np.float64]:
        """Decode Pinocchio native velocity into canonical-v2 velocity."""
        return self.velocity_map(q, v_engine, to_canonical=True, model=model)

    def acceleration_map(
        self,
        q: npt.ArrayLike,
        v: npt.ArrayLike,
        a: npt.ArrayLike,
        *,
        to_canonical: bool = False,
        model: Any | None = None,
    ) -> npt.NDArray[np.float64]:
        """Map acceleration between canonical-v2 and Pinocchio native layouts (NSX-50)."""
        layout = _layout_from_model(model)
        if to_canonical:
            return acceleration_to_canonical(
                q,
                v,
                a,
                base_convention="free_flyer_xyzw",
                q_is_canonical=False,
                layout=layout,
                base_offset=_PELVIS_PREFIX,
            )
        return acceleration_from_canonical(
            q,
            v,
            a,
            base_convention="free_flyer_xyzw",
            q_is_canonical=True,
            layout=layout,
            base_offset=_PELVIS_PREFIX,
        )

    def acceleration_to_canonical(
        self,
        q: npt.ArrayLike,
        v_engine: npt.ArrayLike,
        a_engine: npt.ArrayLike,
        *,
        model: Any | None = None,
    ) -> npt.NDArray[np.float64]:
        """Decode Pinocchio native acceleration into canonical-v2 acceleration."""
        return self.acceleration_map(
            q, v_engine, a_engine, to_canonical=True, model=model
        )

    def generalized_force_map(
        self,
        q: npt.ArrayLike,
        tau: npt.ArrayLike,
        *,
        to_canonical: bool = False,
        model: Any | None = None,
    ) -> npt.NDArray[np.float64]:
        """Dual generalized-force transform preserving virtual power tau . v (NSX-50)."""
        layout = _layout_from_model(model)
        if to_canonical:
            return generalized_force_to_canonical(
                q,
                tau,
                base_convention="free_flyer_xyzw",
                q_is_canonical=False,
                layout=layout,
                base_offset=_PELVIS_PREFIX,
            )
        return generalized_force_from_canonical(
            q,
            tau,
            base_convention="free_flyer_xyzw",
            q_is_canonical=True,
            layout=layout,
            base_offset=_PELVIS_PREFIX,
        )

    def generalized_force_to_canonical(
        self,
        q: npt.ArrayLike,
        tau_engine: npt.ArrayLike,
        *,
        model: Any | None = None,
    ) -> npt.NDArray[np.float64]:
        """Decode Pinocchio native generalized force into canonical-v2 via T^T."""
        return self.generalized_force_map(q, tau_engine, to_canonical=True, model=model)


# Re-export for backwards-compat / convenience.
__all__ = [
    "PinocchioAdapter",
    "euler_xyz_deg_to_quat_wxyz",
    "quat_wxyz_to_euler_xyz_deg",
]
