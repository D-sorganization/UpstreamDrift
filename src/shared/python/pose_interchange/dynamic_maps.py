"""Dual velocity, acceleration and generalized-force maps between engines (NSX-50).

A canonical-v2 state ``(q, v, a)`` (``docs/conventions/canonical-v2.md``) and an
engine's native state describe the same motion with different tangent-space
coordinates: base twist ordering and frame, Euler-rate versus body angular
velocity, joint units and sign.  For a configuration ``q`` the native tangent
vector is a linear image of the canonical one::

    v_n = T(q) v_c                       (velocity_map)
    a_n = T a_c + Tdot v_c               (acceleration_map; Tdot = dT/dt)
    tau_c = T^T tau_n                    (generalized_force_map)

``Tdot`` is the ``Tdot v`` term required wherever the map depends on ``q``
(body-frame linear velocity, Euler rates).  Because ``tau_c . v_c = tau_n . v_n``
the force map is the exact dual, so power is invariant by construction; every
conversion helper re-checks it (design by contract).

Conventions (rows of ``T`` are native components, columns canonical):

* base orientation is ``R`` (world <- body), taken from the native ``q``;
* canonical base twist is ``[linear (world); angular (body)]``;
* a native base block may be ordered angular-first, may express linear or
  angular velocity in the world or body frame, or may use XYZ Euler rates in
  degrees per second (OpenSim; singular at ``cos(beta) = 0``);
* joint speeds scale by ``sign`` and by ``180/pi`` for ``deg`` slots, exactly
  the encoding of :func:`adapters._base.encode_joint_angles`.

Pure numpy; no engine imports.  The closure-row (twist/wrench at the grip)
duality lives in :mod:`pose_interchange.wrench_transform`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from src.shared.python.core.contracts import ensure, require
from src.shared.python.motion_matching.diagnostics.reference_pose import (
    REFERENCE_GOLFER_FIELDS,
)
from src.shared.python.pose_interchange import se3
from src.shared.python.pose_interchange.protocol import JointSlot

Array = npt.NDArray[np.float64]
Orientation = Literal["quat_wxyz", "quat_xyzw", "euler_xyz_deg"]
Frame = Literal["world", "body"]

_BASE_NV = 6
_GIMBAL_TOL = 1e-6
_POWER_RTOL = 1e-9
_ORIENTATION_WIDTH = {"quat_wxyz": 4, "quat_xyzw": 4, "euler_xyz_deg": 3}


@dataclass(frozen=True)
class NativeDynamicLayout:
    """Tangent-space layout of one engine's floating base.

    Attributes:
        engine: Engine identifier (documentation).
        orientation: How the native ``q`` encodes the base orientation.
        orientation_start: Index of the first orientation entry in native ``q``.
        pose_prefix: Length of the pose-layout prefix that precedes the joint
            slots of ``joint_layout``; the native joint velocity index is
            ``slot.start_index - pose_prefix + 6``.
        angular_first: ``True`` when the native base block is ``[ang; lin]``.
        linear_frame: Frame of the native base linear velocity.
        angular_frame: ``"world"``, ``"body"`` or ``"euler_rate"``.
    """

    engine: str
    orientation: Orientation
    orientation_start: int
    pose_prefix: int
    angular_first: bool
    linear_frame: Frame
    angular_frame: Literal["world", "body", "euler_rate"]

    def __post_init__(self) -> None:
        if self.orientation not in _ORIENTATION_WIDTH:
            raise ValueError(f"unknown orientation encoding: {self.orientation!r}")
        if self.linear_frame not in ("world", "body"):
            raise ValueError(
                f"linear_frame must be world or body: {self.linear_frame!r}"
            )
        if self.angular_frame not in ("world", "body", "euler_rate"):
            raise ValueError(f"unknown angular_frame: {self.angular_frame!r}")
        if (self.orientation == "euler_xyz_deg") != (
            self.angular_frame == "euler_rate"
        ):
            raise ValueError(
                "euler_rate angular_frame requires euler_xyz_deg orientation"
            )
        if self.orientation_start < 0 or self.pose_prefix < _BASE_NV:
            raise ValueError("orientation_start >= 0 and pose_prefix >= 6 are required")


def _skew(w: Array) -> Array:
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def base_rotation(layout: NativeDynamicLayout, q: npt.ArrayLike) -> Array:
    """Return ``R`` (world <- body) decoded from the native configuration ``q``."""
    vec = np.asarray(q, dtype=float)
    require(vec.ndim == 1 and bool(np.isfinite(vec).all()), "q must be a finite vector")
    start = layout.orientation_start
    width = _ORIENTATION_WIDTH[layout.orientation]
    require(
        vec.shape[0] >= start + width, f"q too short for {layout.engine} orientation"
    )
    block = vec[start : start + width]
    if layout.orientation == "euler_xyz_deg":
        return se3.euler_xyz_deg_to_matrix(block)
    wxyz = block if layout.orientation == "quat_wxyz" else block[[3, 0, 1, 2]]
    require(
        abs(float(np.linalg.norm(wxyz)) - 1.0) < 1e-6, "base quaternion must be unit"
    )
    return se3.quat_to_matrix(wxyz)


def _euler_rate_matrix(euler_rad: Array) -> tuple[Array, Array, Array]:
    """Return ``(S, dS/dbeta, dS/dgamma)`` with ``omega_body = S @ euler_rates``."""
    _, beta, gamma = euler_rad
    cb, sb, cg, sg = np.cos(beta), np.sin(beta), np.cos(gamma), np.sin(gamma)
    require(abs(cb) > _GIMBAL_TOL, "Euler XYZ rates are singular at cos(beta) = 0")
    s_mat = np.array([[cb * cg, sg, 0.0], [-cb * sg, cg, 0.0], [sb, 0.0, 1.0]])
    d_beta = np.array([[-sb * cg, 0.0, 0.0], [sb * sg, 0.0, 0.0], [cb, 0.0, 0.0]])
    d_gamma = np.array([[-cb * sg, cg, 0.0], [-cb * cg, -sg, 0.0], [0.0, 0.0, 0.0]])
    return s_mat, d_beta, d_gamma


def _base_blocks(
    layout: NativeDynamicLayout, q: Array, omega_body: Array
) -> tuple[Array, Array, Array, Array]:
    """Return ``(A_lin, A_ang, dA_lin, dA_ang)``: native = A canonical, rate dA."""
    rot = base_rotation(layout, q)
    spin = _skew(omega_body)
    lin: Array
    dlin: Array
    ang: Array
    dang: Array
    if layout.linear_frame == "body":
        lin, dlin = rot.T, -spin @ rot.T
    else:
        lin, dlin = np.eye(3), np.zeros((3, 3))
    if layout.angular_frame == "body":
        ang, dang = np.eye(3), np.zeros((3, 3))
    elif layout.angular_frame == "world":
        ang, dang = rot, rot @ spin
    else:
        start = layout.orientation_start
        euler_rad = np.radians(q[start : start + 3])
        s_mat, d_beta, d_gamma = _euler_rate_matrix(euler_rad)
        s_inv = np.linalg.inv(s_mat)
        euler_rate = s_inv @ omega_body
        s_dot = d_beta * euler_rate[1] + d_gamma * euler_rate[2]
        scale = 180.0 / np.pi
        ang, dang = scale * s_inv, -scale * s_inv @ s_dot @ s_inv
    return lin, ang, dlin, dang


def canonical_joint_names(joint_layout: Mapping[str, JointSlot]) -> list[str]:
    """Canonical joint order (``REFERENCE_GOLFER_FIELDS``) restricted to the layout."""
    unknown = sorted(set(joint_layout) - set(REFERENCE_GOLFER_FIELDS))
    require(not unknown, f"joint_layout has non-canonical joints: {unknown}")
    return [name for name in REFERENCE_GOLFER_FIELDS if name in joint_layout]


def _joint_scale(slot: JointSlot) -> float:
    if slot.length != 1:
        raise ValueError(
            f"joint slot {slot.canonical_name!r} is not 1-DOF; dynamic maps "
            "support scalar joint slots only"
        )
    return float(slot.sign) * (180.0 / np.pi if slot.units == "deg" else 1.0)


def velocity_map_matrices(
    layout: NativeDynamicLayout,
    q: npt.ArrayLike,
    joint_layout: Mapping[str, JointSlot],
    v_canonical: npt.ArrayLike | None = None,
) -> tuple[Array, Array]:
    """Return ``(T, Tdot)`` with ``v_native = T v_canonical``.

    ``Tdot`` is ``dT/dt`` along the canonical velocity ``v_canonical`` (zeros
    when it is omitted), so ``a_native = T a_canonical + Tdot v_canonical``.

    Preconditions: ``q`` finite with a valid base orientation; ``v_canonical``
    (if given) has length ``6 + n_joints``.  Postconditions: both matrices are
    ``nv_native x nv_canonical``, finite, and ``T`` is invertible.
    """
    qvec = np.asarray(q, dtype=float)
    names = canonical_joint_names(joint_layout)
    n_c = _BASE_NV + len(names)
    vc: Array
    if v_canonical is None:
        vc = np.zeros(n_c)
    else:
        vc = np.asarray(v_canonical, dtype=float)
        require(vc.shape == (n_c,), f"v_canonical must have length {n_c}")
        require(bool(np.isfinite(vc).all()), "v_canonical must be finite")
    slots = [joint_layout[name] for name in names]
    native_idx = [s.start_index - layout.pose_prefix + _BASE_NV for s in slots]
    require(
        min(native_idx, default=_BASE_NV) >= _BASE_NV, "joint slot inside base prefix"
    )
    n_n = max(native_idx, default=_BASE_NV - 1) + 1
    require(len(set(native_idx)) == len(native_idx), "duplicate native joint indices")
    t_mat, t_dot = np.zeros((n_n, n_c)), np.zeros((n_n, n_c))
    lin, ang, dlin, dang = _base_blocks(layout, qvec, vc[3:6])
    rows = (
        (slice(3, 6), slice(0, 3))
        if layout.angular_first
        else (slice(0, 3), slice(3, 6))
    )
    lin_rows, ang_rows = rows
    t_mat[lin_rows, 0:3], t_mat[ang_rows, 3:6] = lin, ang
    t_dot[lin_rows, 0:3], t_dot[ang_rows, 3:6] = dlin, dang
    for k, (slot, row) in enumerate(zip(slots, native_idx, strict=True)):
        t_mat[row, _BASE_NV + k] = _joint_scale(slot)
    ensure(bool(np.isfinite(t_mat).all() and np.isfinite(t_dot).all()), "maps finite")
    ensure(n_n == n_c and np.linalg.matrix_rank(t_mat) == n_c, "T must be invertible")
    return t_mat, t_dot


# ---- conversions with power-invariance contract ---------------------------------


def check_power_invariance(
    velocity_map: npt.ArrayLike,
    force_map: npt.ArrayLike,
    v_canonical: npt.ArrayLike,
    tau_native: npt.ArrayLike,
    *,
    rtol: float = _POWER_RTOL,
) -> float:
    """Raise ``ValueError`` unless ``tau_n . (T v_c) == (F tau_n) . v_c``.

    ``velocity_map`` is ``T`` and ``force_map`` the claimed dual ``F`` (normally
    ``T^T``), so a force map that is not the dual of the velocity map is caught.
    Returns the native power ``tau_n . v_n``; the tolerance is relative to
    ``|tau_n| |v_n|`` (floor ``1e-12``).
    """
    t_arr = np.asarray(velocity_map, dtype=float)
    f_arr = np.asarray(force_map, dtype=float)
    vc = np.asarray(v_canonical, dtype=float)
    tau = np.asarray(tau_native, dtype=float)
    native_power = float(tau @ (t_arr @ vc))
    canonical_power = float((f_arr @ tau) @ vc)
    scale = max(float(np.linalg.norm(tau) * np.linalg.norm(t_arr @ vc)), 1e-12)
    if abs(native_power - canonical_power) > rtol * scale:
        raise ValueError(
            f"power not preserved: native {native_power!r} vs canonical "
            f"{canonical_power!r}"
        )
    return native_power


def native_velocity(t_mat: npt.ArrayLike, v_canonical: npt.ArrayLike) -> Array:
    """``v_n = T v_c``."""
    return np.asarray(t_mat, dtype=float) @ np.asarray(v_canonical, dtype=float)


def canonical_velocity(t_mat: npt.ArrayLike, v_native: npt.ArrayLike) -> Array:
    """``v_c = T^{-1} v_n`` (inverse of :func:`native_velocity`)."""
    return np.linalg.solve(np.asarray(t_mat, float), np.asarray(v_native, float))


def native_acceleration(
    t_mat: npt.ArrayLike,
    t_dot: npt.ArrayLike,
    a_canonical: npt.ArrayLike,
    v_canonical: npt.ArrayLike,
) -> Array:
    """``a_n = T a_c + Tdot v_c``."""
    t_arr = np.asarray(t_mat, dtype=float)
    return t_arr @ np.asarray(a_canonical, float) + np.asarray(
        t_dot, float
    ) @ np.asarray(v_canonical, float)


def canonical_acceleration(
    t_mat: npt.ArrayLike,
    t_dot: npt.ArrayLike,
    a_native: npt.ArrayLike,
    v_canonical: npt.ArrayLike,
) -> Array:
    """``a_c = T^{-1} (a_n - Tdot v_c)`` (inverse of :func:`native_acceleration`)."""
    drift = np.asarray(t_dot, float) @ np.asarray(v_canonical, float)
    return np.linalg.solve(
        np.asarray(t_mat, float), np.asarray(a_native, float) - drift
    )


def canonical_force(
    t_mat: npt.ArrayLike,
    tau_native: npt.ArrayLike,
    *,
    v_canonical: npt.ArrayLike | None = None,
) -> Array:
    """``tau_c = T^T tau_n``; with ``v_canonical`` the power is re-checked."""
    t_arr = np.asarray(t_mat, dtype=float)
    tau_c = t_arr.T @ np.asarray(tau_native, dtype=float)
    if v_canonical is not None:
        check_power_invariance(t_mat, t_arr.T, v_canonical, tau_native)
    return tau_c


def native_force(t_mat: npt.ArrayLike, tau_canonical: npt.ArrayLike) -> Array:
    """``tau_n = T^{-T} tau_c`` (inverse of :func:`canonical_force`)."""
    return np.linalg.solve(
        np.asarray(t_mat, dtype=float).T, np.asarray(tau_canonical, dtype=float)
    )


# ---- adapter surface -------------------------------------------------------------


@runtime_checkable
class DynamicMapAdapter(Protocol):
    """Adapter extension: dual velocity, acceleration and force maps."""

    def velocity_map(
        self, engine_q: npt.ArrayLike, *, model: Any | None = None
    ) -> Array:
        """``T`` with ``v_native = T v_canonical``."""

    def acceleration_map(
        self,
        engine_q: npt.ArrayLike,
        v_canonical: npt.ArrayLike,
        *,
        model: Any | None = None,
    ) -> tuple[Array, Array]:
        """``(T, Tdot)`` with ``a_native = T a_c + Tdot v_c``."""

    def generalized_force_map(
        self, engine_q: npt.ArrayLike, *, model: Any | None = None
    ) -> Array:
        """``T^T`` with ``tau_canonical = T^T tau_native``."""


class DynamicMapsMixin:
    """Implements :class:`DynamicMapAdapter` for a pose adapter.

    Subclasses set ``dynamic_layout`` and provide ``joint_layout(model)``.
    """

    dynamic_layout: NativeDynamicLayout

    def joint_layout(self, model: Any | None = None) -> Mapping[str, JointSlot]:
        """Native joint layout; every concrete adapter overrides this."""
        raise TypeError(f"{type(self).__name__} must define joint_layout()")

    def velocity_map(
        self, engine_q: npt.ArrayLike, *, model: Any | None = None
    ) -> Array:
        """Return ``T(q)`` (``v_native = T v_canonical``); see module docstring."""
        t_mat, _ = velocity_map_matrices(
            self.dynamic_layout, engine_q, self.joint_layout(model)
        )
        return t_mat

    def acceleration_map(
        self,
        engine_q: npt.ArrayLike,
        v_canonical: npt.ArrayLike,
        *,
        model: Any | None = None,
    ) -> tuple[Array, Array]:
        """Return ``(T, Tdot)``: ``a_native = T a_canonical + Tdot v_canonical``."""
        return velocity_map_matrices(
            self.dynamic_layout, engine_q, self.joint_layout(model), v_canonical
        )

    def generalized_force_map(
        self, engine_q: npt.ArrayLike, *, model: Any | None = None
    ) -> Array:
        """Return ``T^T`` (``tau_canonical = T^T tau_native``); power-dual of T."""
        return self.velocity_map(engine_q, model=model).T


__all__ = [
    "DynamicMapAdapter",
    "DynamicMapsMixin",
    "NativeDynamicLayout",
    "base_rotation",
    "canonical_acceleration",
    "canonical_force",
    "canonical_joint_names",
    "canonical_velocity",
    "check_power_invariance",
    "native_acceleration",
    "native_force",
    "native_velocity",
    "velocity_map_matrices",
]
