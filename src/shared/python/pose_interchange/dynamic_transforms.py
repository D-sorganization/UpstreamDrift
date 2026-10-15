"""Dynamic state and dual (force/wrench) transforms across engine coordinate conventions (NSX-50).

Canonical-v2 dynamic state uses:
- ``q`` (``nq = 7 + n_j``): ``[base_xyz(3), base_quat_wxyz(4), joints_rad(n_j)]``
- ``v`` (``nv = 6 + n_j``): ``[base_lin_world(3), base_ang_body(3), joint_rates_rad_s(n_j)]``
- ``a`` (``nv``): time derivative of ``v``, ``[base_acc_world(3), base_ang_acc_body(3), joint_acc_rad_s2(n_j)]``
- ``tau`` (``nv``): dual generalized forces conjugate to ``v`` so virtual power is ``tau . v``.

For any engine whose tangent space is related to canonical-v2 by a configuration-dependent
invertible linear map ``v_engine = T(q) @ v_canonical``:
- **Velocity**: ``v_engine = T(q) @ v_canonical``
- **Acceleration**: ``a_engine = T(q) @ a_canonical + T_dot(q, v) @ v_canonical``
- **Generalized force (dual)**: ``tau_canonical = T(q).T @ tau_engine`` and
  ``tau_engine = T(q)^{-T} @ tau_canonical``, preserving virtual power
  ``tau_engine . v_engine == tau_canonical . v_canonical`` identically.

Engine tangent conventions handled here:
1. **Pinocchio / MuJoCo / MyoSuite** (free-flyer ``[lin_world(3), ang_body(3), joints]``):
   ``T_base = I_6``, ``T_dot_base = 0``, plus per-joint unit/sign scaling from ``JointSlot``.
2. **Drake**:
   - Quaternion-floating mode (``len(q_engine) == nv + 1``):
     ``v_base_drake = [ang_world(3), lin_world(3)] = [[0, R], [I_3, 0]] @ [lin_world, ang_body]``.
     Since ``R_dot = R [omega_body]_x``, ``d/dt (R omega_body) = R alpha_body + R (omega_body x omega_body) = R alpha_body``,
     so ``T_dot = 0`` even at nonzero angular velocity.
   - RPY-Euler floating-base mode (``len(q_engine) == nv``, ``q_base = [xyz, rpy_rad]``):
     For intrinsic XYZ Euler angles ``(phi, theta, psi)``, body angular velocity is
     ``omega_body = E(phi, theta, psi) @ rpy_dot``, so ``rpy_dot = E^{-1} @ omega_body`` and
     ``rpy_ddot = E^{-1} @ (alpha_body - E_dot(rpy, rpy_dot) @ rpy_dot)``.
3. **OpenSim / Simscape** (intrinsic XYZ Euler in degrees or radians, ``len(q_engine) == nv``):
   Base rotational coordinates are intrinsic XYZ Euler angles (in degrees for OpenSim/Simscape
   pose conventions, or radians when ``base_angle_units="rad"``), with exact ``E(rpy)`` and
   ``E_dot(rpy, rpy_dot)`` kinematic coupling in ``acceleration_map`` and ``E^{-T}`` in
   ``generalized_force_map``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt

from src.shared.python.contracts import require
from src.shared.python.pose_interchange.protocol import JointSlot
from src.shared.python.pose_interchange.se3 import (
    euler_xyz_deg_to_matrix,
    matrix_to_euler_xyz_deg,
    quat_to_matrix,
)

Array: TypeAlias = npt.NDArray[np.float64]
BaseConvention: TypeAlias = Literal[
    "free_flyer_wxyz",
    "free_flyer_xyzw",
    "drake_quat_floating",
    "euler_xyz_rad",
    "euler_xyz_deg",
]

_BASE_NV = 6
_SINGULAR_COS_TOL = 1e-7

__all__ = [
    "BaseConvention",
    "acceleration_from_canonical",
    "acceleration_to_canonical",
    "euler_xyz_rate_matrix",
    "euler_xyz_rate_matrix_dot",
    "generalized_force_from_canonical",
    "generalized_force_to_canonical",
    "tangent_jacobian_from_canonical",
    "velocity_from_canonical",
    "velocity_to_canonical",
]


def _require_1d_finite(
    vec: npt.ArrayLike, name: str, expected_len: int | None = None
) -> Array:
    if not isinstance(vec, (np.ndarray, list, tuple)):
        raise TypeError(f"{name} must be a 1-D array-like")
    arr = np.asarray(vec, dtype=np.float64)
    require(arr.ndim == 1, f"{name} must be 1-D, got shape {arr.shape}")
    if expected_len is not None:
        require(
            arr.shape[0] == expected_len,
            f"{name} must have length {expected_len}, got {arr.shape[0]}",
        )
    require(bool(np.isfinite(arr).all()), f"{name} must contain only finite values")
    return arr


def euler_xyz_rate_matrix(rpy_rad: npt.ArrayLike) -> Array:
    """Return 3x3 matrix ``E(phi, theta, psi)`` mapping ``rpy_dot -> omega_body``.

    For intrinsic XYZ rotation ``R = Rx(phi) Ry(theta) Rz(psi)``, the body-frame
    angular velocity is::

        omega_body = E(phi, theta, psi) @ [phi_dot, theta_dot, psi_dot]^T

    where::

        E = [[ cos(theta)*cos(psi),  sin(psi), 0 ],
             [-cos(theta)*sin(psi),  cos(psi), 0 ],
             [ sin(theta),           0,        1 ]]
    """
    angles = _require_1d_finite(rpy_rad, "rpy_rad", 3)
    _, theta, psi = float(angles[0]), float(angles[1]), float(angles[2])
    ct, st = float(np.cos(theta)), float(np.sin(theta))
    require(
        abs(ct) > _SINGULAR_COS_TOL,
        f"Intrinsic XYZ Euler parameterization is singular near pitch=+-pi/2 (cos(theta)={ct:.3e})",
    )
    cp, sp = float(np.cos(psi)), float(np.sin(psi))
    return np.array(
        [
            [ct * cp, sp, 0.0],
            [-ct * sp, cp, 0.0],
            [st, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def euler_xyz_rate_matrix_dot(
    rpy_rad: npt.ArrayLike, rpy_dot_rad_s: npt.ArrayLike
) -> Array:
    """Return time derivative ``E_dot(rpy, rpy_dot)`` of :func:`euler_xyz_rate_matrix`."""
    angles = _require_1d_finite(rpy_rad, "rpy_rad", 3)
    rates = _require_1d_finite(rpy_dot_rad_s, "rpy_dot_rad_s", 3)
    _, theta, psi = float(angles[0]), float(angles[1]), float(angles[2])
    _, theta_dot, psi_dot = float(rates[0]), float(rates[1]), float(rates[2])
    ct, st = float(np.cos(theta)), float(np.sin(theta))
    cp, sp = float(np.cos(psi)), float(np.sin(psi))
    return np.array(
        [
            [-st * theta_dot * cp - ct * sp * psi_dot, cp * psi_dot, 0.0],
            [st * theta_dot * sp - ct * cp * psi_dot, -sp * psi_dot, 0.0],
            [ct * theta_dot, 0.0, 0.0],
        ],
        dtype=np.float64,
    )


def _joint_scales(
    n_joints: int,
    layout: Mapping[str, JointSlot] | None,
    base_offset: int,
) -> Array:
    """Return per-joint scale factors ``s_i`` such that ``v_eng_joint[i] = s_i * v_canon_joint[i]``."""
    scales = np.ones(n_joints, dtype=np.float64)
    if layout is None:
        return scales
    for slot in layout.values():
        idx = slot.start_index - base_offset
        if 0 <= idx < n_joints:
            unit_scale = float(np.degrees(1.0)) if slot.units == "deg" else 1.0
            scales[idx] = float(slot.sign) * unit_scale
    return scales


def _extract_canonical_rot_and_rpy(
    q: Array,
    base_convention: BaseConvention,
    q_is_canonical: bool,
) -> tuple[Array, Array, int]:
    """Return ``(R_wb, rpy_rad, n_joints)`` from either canonical or engine ``q``."""
    if q_is_canonical:
        require(q.shape[0] >= 7, f"canonical q must have length >= 7, got {q.shape[0]}")
        rot = quat_to_matrix(q[3:7])
        rpy_rad = np.radians(matrix_to_euler_xyz_deg(rot))
        return rot, rpy_rad, int(q.shape[0] - 7)

    if base_convention == "free_flyer_wxyz":
        require(q.shape[0] >= 7, f"engine q must have length >= 7, got {q.shape[0]}")
        rot = quat_to_matrix(q[3:7])
        return rot, np.radians(matrix_to_euler_xyz_deg(rot)), int(q.shape[0] - 7)
    if base_convention == "free_flyer_xyzw":
        require(q.shape[0] >= 7, f"engine q must have length >= 7, got {q.shape[0]}")
        wxyz = np.array([q[6], q[3], q[4], q[5]], dtype=np.float64)
        rot = quat_to_matrix(wxyz)
        return rot, np.radians(matrix_to_euler_xyz_deg(rot)), int(q.shape[0] - 7)
    if base_convention == "drake_quat_floating":
        require(q.shape[0] >= 7, f"engine q must have length >= 7, got {q.shape[0]}")
        rot = quat_to_matrix(q[0:4])
        return rot, np.radians(matrix_to_euler_xyz_deg(rot)), int(q.shape[0] - 7)
    if base_convention == "euler_xyz_rad":
        require(q.shape[0] >= 6, f"engine q must have length >= 6, got {q.shape[0]}")
        rpy_rad = q[3:6].copy()
        rot = euler_xyz_deg_to_matrix(np.degrees(rpy_rad))
        return rot, rpy_rad, int(q.shape[0] - 6)
    if base_convention == "euler_xyz_deg":
        require(q.shape[0] >= 6, f"engine q must have length >= 6, got {q.shape[0]}")
        rpy_deg = q[3:6].copy()
        rot = euler_xyz_deg_to_matrix(rpy_deg)
        return rot, np.radians(rpy_deg), int(q.shape[0] - 6)
    raise ValueError(f"Unsupported base_convention {base_convention!r}")


def tangent_jacobian_from_canonical(
    q: npt.ArrayLike,
    *,
    base_convention: BaseConvention,
    q_is_canonical: bool = True,
    layout: Mapping[str, JointSlot] | None = None,
    base_offset: int = 6,
) -> Array:
    """Return ``(nv, nv)`` tangent map ``T(q)`` such that ``v_engine = T(q) @ v_canonical``."""
    q_arr = _require_1d_finite(q, "q")
    rot, rpy_rad, n_joints = _extract_canonical_rot_and_rpy(
        q_arr, base_convention, q_is_canonical
    )
    nv = _BASE_NV + n_joints
    t_mat = np.zeros((nv, nv), dtype=np.float64)
    eye3 = np.eye(3, dtype=np.float64)

    if base_convention in ("free_flyer_wxyz", "free_flyer_xyzw"):
        t_mat[0:3, 0:3] = eye3
        t_mat[3:6, 3:6] = eye3
    elif base_convention == "drake_quat_floating":
        # v_drake_base = [omega_world(3), v_lin_world(3)]
        t_mat[0:3, 3:6] = rot
        t_mat[3:6, 0:3] = eye3
    elif base_convention == "euler_xyz_rad":
        # v_base = [v_lin_world(3), rpy_dot_rad(3)]
        e_inv = np.linalg.inv(euler_xyz_rate_matrix(rpy_rad))
        t_mat[0:3, 0:3] = eye3
        t_mat[3:6, 3:6] = e_inv
    elif base_convention == "euler_xyz_deg":
        e_inv = np.linalg.inv(euler_xyz_rate_matrix(rpy_rad))
        t_mat[0:3, 0:3] = eye3
        t_mat[3:6, 3:6] = float(np.degrees(1.0)) * e_inv
    else:
        raise ValueError(f"Unsupported base_convention {base_convention!r}")

    scales = _joint_scales(n_joints, layout, base_offset)
    if n_joints > 0:
        t_mat[_BASE_NV:, _BASE_NV:] = np.diag(scales)
    return t_mat


def _tangent_jacobian_dot_from_canonical(
    q: npt.ArrayLike,
    v_canonical: Array,
    *,
    base_convention: BaseConvention,
    q_is_canonical: bool = True,
) -> Array:
    """Return ``(nv, nv)`` matrix ``T_dot(q, v)`` for ``a_engine = T a_c + T_dot v_c``."""
    q_arr = _require_1d_finite(q, "q")
    _, rpy_rad, n_joints = _extract_canonical_rot_and_rpy(
        q_arr, base_convention, q_is_canonical
    )
    nv = _BASE_NV + n_joints
    t_dot = np.zeros((nv, nv), dtype=np.float64)

    if base_convention in ("euler_xyz_rad", "euler_xyz_deg"):
        e_mat = euler_xyz_rate_matrix(rpy_rad)
        e_inv = np.linalg.inv(e_mat)
        omega_body = v_canonical[3:6]
        rpy_dot_rad = e_inv @ omega_body
        e_dot = euler_xyz_rate_matrix_dot(rpy_rad, rpy_dot_rad)
        # d/dt (E^{-1}) = -E^{-1} E_dot E^{-1}
        e_inv_dot = -e_inv @ e_dot @ e_inv
        scale = float(np.degrees(1.0)) if base_convention == "euler_xyz_deg" else 1.0
        t_dot[3:6, 3:6] = scale * e_inv_dot
    return t_dot


def velocity_from_canonical(
    q: npt.ArrayLike,
    v_canonical: npt.ArrayLike,
    *,
    base_convention: BaseConvention,
    q_is_canonical: bool = True,
    layout: Mapping[str, JointSlot] | None = None,
    base_offset: int = 6,
) -> Array:
    """Map canonical-v2 velocity ``v_canonical`` into engine velocity ``v_engine = T(q) v_c``."""
    t_mat = tangent_jacobian_from_canonical(
        q,
        base_convention=base_convention,
        q_is_canonical=q_is_canonical,
        layout=layout,
        base_offset=base_offset,
    )
    v_arr = _require_1d_finite(v_canonical, "v", t_mat.shape[0])
    return np.ascontiguousarray(t_mat @ v_arr, dtype=np.float64)


def velocity_to_canonical(
    q: npt.ArrayLike,
    v_engine: npt.ArrayLike,
    *,
    base_convention: BaseConvention,
    q_is_canonical: bool = False,
    layout: Mapping[str, JointSlot] | None = None,
    base_offset: int = 6,
) -> Array:
    """Map engine velocity ``v_engine`` into canonical-v2 velocity ``v_c = T(q)^{-1} v_eng``."""
    t_mat = tangent_jacobian_from_canonical(
        q,
        base_convention=base_convention,
        q_is_canonical=q_is_canonical,
        layout=layout,
        base_offset=base_offset,
    )
    v_arr = _require_1d_finite(v_engine, "v", t_mat.shape[0])
    return np.ascontiguousarray(np.linalg.solve(t_mat, v_arr), dtype=np.float64)


def acceleration_from_canonical(
    q: npt.ArrayLike,
    v_canonical: npt.ArrayLike,
    a_canonical: npt.ArrayLike,
    *,
    base_convention: BaseConvention,
    q_is_canonical: bool = True,
    layout: Mapping[str, JointSlot] | None = None,
    base_offset: int = 6,
) -> Array:
    """Map canonical-v2 ``(v_c, a_c)`` to engine acceleration ``a_eng = T a_c + T_dot v_c``."""
    t_mat = tangent_jacobian_from_canonical(
        q,
        base_convention=base_convention,
        q_is_canonical=q_is_canonical,
        layout=layout,
        base_offset=base_offset,
    )
    nv = t_mat.shape[0]
    v_arr = _require_1d_finite(v_canonical, "v", nv)
    a_arr = _require_1d_finite(a_canonical, "a", nv)
    t_dot = _tangent_jacobian_dot_from_canonical(
        q, v_arr, base_convention=base_convention, q_is_canonical=q_is_canonical
    )
    return np.ascontiguousarray(t_mat @ a_arr + t_dot @ v_arr, dtype=np.float64)


def acceleration_to_canonical(
    q: npt.ArrayLike,
    v_engine: npt.ArrayLike,
    a_engine: npt.ArrayLike,
    *,
    base_convention: BaseConvention,
    q_is_canonical: bool = False,
    layout: Mapping[str, JointSlot] | None = None,
    base_offset: int = 6,
) -> Array:
    """Map engine ``(v_eng, a_eng)`` to canonical-v2 acceleration ``a_c = T^{-1}(a_eng - T_dot v_c)``."""
    t_mat = tangent_jacobian_from_canonical(
        q,
        base_convention=base_convention,
        q_is_canonical=q_is_canonical,
        layout=layout,
        base_offset=base_offset,
    )
    nv = t_mat.shape[0]
    v_eng = _require_1d_finite(v_engine, "v", nv)
    a_eng = _require_1d_finite(a_engine, "a", nv)
    v_canon = np.ascontiguousarray(np.linalg.solve(t_mat, v_eng), dtype=np.float64)
    t_dot = _tangent_jacobian_dot_from_canonical(
        q, v_canon, base_convention=base_convention, q_is_canonical=q_is_canonical
    )
    return np.ascontiguousarray(
        np.linalg.solve(t_mat, a_eng - t_dot @ v_canon), dtype=np.float64
    )


def generalized_force_from_canonical(
    q: npt.ArrayLike,
    tau_canonical: npt.ArrayLike,
    *,
    base_convention: BaseConvention,
    q_is_canonical: bool = True,
    layout: Mapping[str, JointSlot] | None = None,
    base_offset: int = 6,
) -> Array:
    """Map canonical-v2 generalized force ``tau_c`` to engine ``tau_eng = T(q)^{-T} tau_c``.

    Preserves virtual power ``tau_eng . v_eng == tau_c . v_c`` for all ``v_c``.
    """
    t_mat = tangent_jacobian_from_canonical(
        q,
        base_convention=base_convention,
        q_is_canonical=q_is_canonical,
        layout=layout,
        base_offset=base_offset,
    )
    tau_arr = _require_1d_finite(tau_canonical, "tau", t_mat.shape[0])
    return np.ascontiguousarray(np.linalg.solve(t_mat.T, tau_arr), dtype=np.float64)


def generalized_force_to_canonical(
    q: npt.ArrayLike,
    tau_engine: npt.ArrayLike,
    *,
    base_convention: BaseConvention,
    q_is_canonical: bool = False,
    layout: Mapping[str, JointSlot] | None = None,
    base_offset: int = 6,
) -> Array:
    """Map engine generalized force ``tau_eng`` to canonical-v2 ``tau_c = T(q)^T tau_eng``.

    Preserves virtual power ``tau_c . v_c == tau_eng . v_eng`` for all ``v_eng``.
    """
    t_mat = tangent_jacobian_from_canonical(
        q,
        base_convention=base_convention,
        q_is_canonical=q_is_canonical,
        layout=layout,
        base_offset=base_offset,
    )
    tau_arr = _require_1d_finite(tau_engine, "tau", t_mat.shape[0])
    return np.ascontiguousarray(t_mat.T @ tau_arr, dtype=np.float64)
