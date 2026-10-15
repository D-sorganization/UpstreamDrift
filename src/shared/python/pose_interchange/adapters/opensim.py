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
from src.shared.python.pose_interchange.dynamic_transforms import (
    BaseConvention,
    acceleration_from_canonical,
    acceleration_to_canonical,
    generalized_force_from_canonical,
    generalized_force_to_canonical,
    velocity_from_canonical,
    velocity_to_canonical,
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

    def velocity_map(
        self,
        q: npt.ArrayLike,
        v: npt.ArrayLike,
        *,
        to_canonical: bool = False,
        base_units: str = "deg",
        model: Any | None = None,
    ) -> npt.NDArray[np.float64]:
        """Map velocity between canonical-v2 and OpenSim native layouts (NSX-50).

        Handles the nonlinear intrinsic-XYZ Euler kinematic map ``omega_body = E(rpy) rpy_dot``
        as well as per-joint unit/sign conventions from ``joint_layout(model)``.
        """
        conv, q_is_canon = _opensim_base_convention(q, v, base_units)
        layout = _layout_from_model(model)
        if to_canonical:
            return velocity_to_canonical(
                q,
                v,
                base_convention=conv,
                q_is_canonical=q_is_canon,
                layout=layout,
                base_offset=_PELVIS_PREFIX,
            )
        return velocity_from_canonical(
            q,
            v,
            base_convention=conv,
            q_is_canonical=q_is_canon,
            layout=layout,
            base_offset=_PELVIS_PREFIX,
        )

    def velocity_to_canonical(
        self,
        q: npt.ArrayLike,
        v_engine: npt.ArrayLike,
        *,
        base_units: str = "deg",
        model: Any | None = None,
    ) -> npt.NDArray[np.float64]:
        """Decode OpenSim native velocity into canonical-v2 velocity."""
        return self.velocity_map(
            q, v_engine, to_canonical=True, base_units=base_units, model=model
        )

    def acceleration_map(
        self,
        q: npt.ArrayLike,
        v: npt.ArrayLike,
        a: npt.ArrayLike,
        *,
        to_canonical: bool = False,
        base_units: str = "deg",
        model: Any | None = None,
    ) -> npt.NDArray[np.float64]:
        """Map acceleration between canonical-v2 and OpenSim native layouts (NSX-50).

        Includes the non-linear ``T_dot(q, v) v`` Euler-rate derivative term.
        """
        conv, q_is_canon = _opensim_base_convention(q, v, base_units)
        layout = _layout_from_model(model)
        if to_canonical:
            return acceleration_to_canonical(
                q,
                v,
                a,
                base_convention=conv,
                q_is_canonical=q_is_canon,
                layout=layout,
                base_offset=_PELVIS_PREFIX,
            )
        return acceleration_from_canonical(
            q,
            v,
            a,
            base_convention=conv,
            q_is_canonical=q_is_canon,
            layout=layout,
            base_offset=_PELVIS_PREFIX,
        )

    def acceleration_to_canonical(
        self,
        q: npt.ArrayLike,
        v_engine: npt.ArrayLike,
        a_engine: npt.ArrayLike,
        *,
        base_units: str = "deg",
        model: Any | None = None,
    ) -> npt.NDArray[np.float64]:
        """Decode OpenSim native acceleration into canonical-v2 acceleration."""
        return self.acceleration_map(
            q,
            v_engine,
            a_engine,
            to_canonical=True,
            base_units=base_units,
            model=model,
        )

    def generalized_force_map(
        self,
        q: npt.ArrayLike,
        tau: npt.ArrayLike,
        *,
        to_canonical: bool = False,
        base_units: str = "deg",
        model: Any | None = None,
    ) -> npt.NDArray[np.float64]:
        """Dual generalized-force transform preserving virtual power tau . v (NSX-50)."""
        conv, q_is_canon = _opensim_base_convention(q, tau, base_units)
        layout = _layout_from_model(model)
        if to_canonical:
            return generalized_force_to_canonical(
                q,
                tau,
                base_convention=conv,
                q_is_canonical=q_is_canon,
                layout=layout,
                base_offset=_PELVIS_PREFIX,
            )
        return generalized_force_from_canonical(
            q,
            tau,
            base_convention=conv,
            q_is_canonical=q_is_canon,
            layout=layout,
            base_offset=_PELVIS_PREFIX,
        )

    def generalized_force_to_canonical(
        self,
        q: npt.ArrayLike,
        tau_engine: npt.ArrayLike,
        *,
        base_units: str = "deg",
        model: Any | None = None,
    ) -> npt.NDArray[np.float64]:
        """Decode OpenSim native generalized force into canonical-v2 via T^T."""
        return self.generalized_force_map(
            q, tau_engine, to_canonical=True, base_units=base_units, model=model
        )


def _opensim_base_convention(
    q: npt.ArrayLike,
    tangent: npt.ArrayLike,
    base_units: str,
) -> tuple[BaseConvention, bool]:
    if base_units not in ("deg", "rad"):
        raise ValueError(f"base_units must be 'deg' or 'rad', got {base_units!r}")
    q_arr = np.asarray(q, dtype=float)
    t_arr = np.asarray(tangent, dtype=float)
    conv: BaseConvention = "euler_xyz_deg" if base_units == "deg" else "euler_xyz_rad"
    q_is_canon = (
        q_arr.ndim == 1 and t_arr.ndim == 1 and q_arr.shape[0] == t_arr.shape[0] + 1
    )
    return conv, q_is_canon
