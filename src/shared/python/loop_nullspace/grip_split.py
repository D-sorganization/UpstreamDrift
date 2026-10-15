"""Grip-frame hand-split coordinates, 6x6 grasp-map kernel, and bare-club net wrench (#12561, NSX-16).

Implements Rule R4 of Epic #12593:
- Canonical grip-frame grasp map G (6x12) mapping (W_L, W_R) to net wrench W_net about grip_frame.origin.
- Exact 12x6 internal-wrench kernel K (G @ K == 0, rank(K) == 6) with six physical split coordinates.
- Bare-club net wrench from Newton-Euler dynamics tagged AllocationDependence.INVARIANT (C-FRAME-GRIP).
- Exact invertible synthesis hand_wrenches and analysis split_coordinates mappings.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.loop_nullspace.contracts import (
    AllocationDependence,
    ClosedLoopSnapshot,
    FloatArray,
    GripFrame,
    QuantityRecord,
    _readonly_float_array,
)

SPLIT_COORDINATE_NAMES: tuple[str, ...] = (
    "axial_pull_squeeze_n",
    "shear_v_with_moment_n",
    "shear_w_with_moment_n",
    "differential_twist_nm",
    "differential_bend_v_nm",
    "differential_bend_w_nm",
)

GRIP_FRAME_CONVENTION_ID = "C-FRAME-GRIP"


def _skew(vec: FloatArray) -> FloatArray:
    """Return the 3x3 skew-symmetric cross-product matrix [vec]_x."""
    x, y, z = float(vec[0]), float(vec[1]), float(vec[2])
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )


def _grip_lever_arms(
    grip_frame: GripFrame,
    r_L_world_m: ArrayLike,
    r_R_world_m: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """Validate grip frame and hand positions and return lever arms (d_L, d_R) in grip frame."""
    if not isinstance(grip_frame, GripFrame):
        raise TypeError("grip_frame must be a GripFrame instance")
    r_l = _readonly_float_array(r_L_world_m, "r_L_world_m", shape=(3,))
    r_r = _readonly_float_array(r_R_world_m, "r_R_world_m", shape=(3,))
    sep = float(np.linalg.norm(r_r - r_l))
    if sep <= 1e-9:
        raise ValueError(
            f"Hand grip points r_L_world_m and r_R_world_m must be distinct (||r_R - r_L||={sep:.3e})"
        )
    rot_t = grip_frame.rotation.T
    d_l = rot_t @ (r_l - grip_frame.origin)
    d_r = rot_t @ (r_r - grip_frame.origin)
    return d_l, d_r


def grasp_map(
    grip_frame: GripFrame,
    r_L_world_m: ArrayLike,
    r_R_world_m: ArrayLike,
) -> FloatArray:
    """Return the 6x12 grasp map G = [G_L, G_R] in the canonical grip frame.

    Maps stacked hand wrenches [f_L; tau_L; f_R; tau_R] expressed in grip_frame
    to the net wrench W_net = [f_net; tau_net] about grip_frame.origin.
    """
    d_l, d_r = _grip_lever_arms(grip_frame, r_L_world_m, r_R_world_m)
    eye3 = np.eye(3, dtype=np.float64)
    zero3 = np.zeros((3, 3), dtype=np.float64)
    top = np.hstack([eye3, zero3, eye3, zero3])
    bottom = np.hstack([_skew(d_l), eye3, _skew(d_r), eye3])
    mat = np.vstack([top, bottom])
    mat.setflags(write=False)
    return mat


def _interhand_basis_in_grip(d_l: FloatArray, d_r: FloatArray) -> FloatArray:
    """Construct right-handed orthonormal basis [u_hat, v_hat, w_hat] in grip frame."""
    delta = d_l - d_r
    norm_delta = float(np.linalg.norm(delta))
    u_hat = delta / norm_delta
    # If u_hat is parallel to +/- e_1 in grip_frame, preserve canonical [e_1, e_2, e_3]
    if abs(abs(float(u_hat[0])) - 1.0) <= 1e-9:
        return np.eye(3, dtype=np.float64)
    ref = (
        np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(float(u_hat[1])) < 0.9
        else np.array([0.0, 0.0, 1.0], dtype=np.float64)
    )
    v_raw = ref - float(np.dot(ref, u_hat)) * u_hat
    v_hat = v_raw / float(np.linalg.norm(v_raw))
    w_hat = np.cross(u_hat, v_hat)
    return np.column_stack([u_hat, v_hat, w_hat])


def split_basis(
    grip_frame: GripFrame,
    r_L_world_m: ArrayLike,
    r_R_world_m: ArrayLike,
) -> FloatArray:
    """Return the exact 12x6 kernel matrix K (G @ K == 0, rank(K) == 6).

    Columns of K correspond to SPLIT_COORDINATE_NAMES:
    - s_1..s_3: internal force components (+1/2 b_i on L, -1/2 b_i on R) paired with
      compensating free moments -1/2 (d_L x f_L + d_R x f_R) so net force and net
      moment about grip_frame.origin vanish identically.
    - s_4..s_6: differential free moments (+1/2 e_i on L, -1/2 e_i on R, zero force).
    """
    d_l, d_r = _grip_lever_arms(grip_frame, r_L_world_m, r_R_world_m)
    basis_f = _interhand_basis_in_grip(d_l, d_r)
    eye3 = np.eye(3, dtype=np.float64)
    zero3 = np.zeros((3, 3), dtype=np.float64)
    moment_comp = -0.25 * _skew(d_l - d_r) @ basis_f

    kernel = np.vstack(
        [
            np.hstack([0.5 * basis_f, zero3]),
            np.hstack([moment_comp, 0.5 * eye3]),
            np.hstack([-0.5 * basis_f, zero3]),
            np.hstack([moment_comp, -0.5 * eye3]),
        ]
    )
    kernel.setflags(write=False)
    return kernel


def _particular_inverse(d_l: FloatArray, d_r: FloatArray) -> FloatArray:
    """Return the 12x6 right-inverse G_p of G such that G @ G_p == I_6."""
    eye3 = np.eye(3, dtype=np.float64)
    zero3 = np.zeros((3, 3), dtype=np.float64)
    mean_skew = -0.25 * _skew(d_l + d_r)
    return np.vstack(
        [
            np.hstack([0.5 * eye3, zero3]),
            np.hstack([mean_skew, 0.5 * eye3]),
            np.hstack([0.5 * eye3, zero3]),
            np.hstack([mean_skew, 0.5 * eye3]),
        ]
    )


def hand_wrenches(
    w_net_grip: ArrayLike,
    s_coords: ArrayLike,
    *,
    grip_frame: GripFrame,
    r_L_world_m: ArrayLike,
    r_R_world_m: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """Synthesize left and right hand wrenches (W_L, W_R) in grip_frame from (W_net, s)."""
    w_net = _readonly_float_array(w_net_grip, "w_net_grip", shape=(6,))
    s_vec = _readonly_float_array(s_coords, "s_coords", shape=(6,))
    d_l, d_r = _grip_lever_arms(grip_frame, r_L_world_m, r_R_world_m)
    g_p = _particular_inverse(d_l, d_r)
    k_mat = split_basis(grip_frame, r_L_world_m, r_R_world_m)
    stacked = g_p @ w_net + k_mat @ s_vec
    w_l = _readonly_float_array(stacked[:6], "w_L_grip", shape=(6,))
    w_r = _readonly_float_array(stacked[6:], "w_R_grip", shape=(6,))
    return w_l, w_r


def split_coordinates(
    w_L_grip: ArrayLike,
    w_R_grip: ArrayLike,
    *,
    grip_frame: GripFrame,
    r_L_world_m: ArrayLike,
    r_R_world_m: ArrayLike,
) -> FloatArray:
    """Extract the 6 hand-split coordinates s from (W_L, W_R) in grip_frame."""
    w_l = _readonly_float_array(w_L_grip, "w_L_grip", shape=(6,))
    w_r = _readonly_float_array(w_R_grip, "w_R_grip", shape=(6,))
    d_l, d_r = _grip_lever_arms(grip_frame, r_L_world_m, r_R_world_m)
    basis_f = _interhand_basis_in_grip(d_l, d_r)
    s_f = basis_f.T @ (w_l[:3] - w_r[:3])
    s_m = w_l[3:] - w_r[3:]
    return _readonly_float_array(np.concatenate([s_f, s_m]), "s_coords", shape=(6,))


def _infer_club_kinematics(
    snapshot: ClosedLoopSnapshot,
    club_com_accel_world_m_s2: ArrayLike | None,
    club_ang_vel_world_rad_s: ArrayLike | None,
    club_ang_accel_world_rad_s2: ArrayLike | None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Resolve bare-club CoM linear acceleration, angular velocity, and angular acceleration."""
    if (
        club_com_accel_world_m_s2 is not None
        and club_ang_vel_world_rad_s is not None
        and club_ang_accel_world_rad_s2 is not None
    ):
        return (
            _readonly_float_array(
                club_com_accel_world_m_s2, "club_com_accel_world_m_s2", shape=(3,)
            ),
            _readonly_float_array(
                club_ang_vel_world_rad_s, "club_ang_vel_world_rad_s", shape=(3,)
            ),
            _readonly_float_array(
                club_ang_accel_world_rad_s2,
                "club_ang_accel_world_rad_s2",
                shape=(3,),
            ),
        )

    coords = snapshot.topology.coordinate_order
    r_rel = snapshot.bare_club.com_m - snapshot.grip_origin
    omega_default = np.zeros(3, dtype=np.float64)
    alpha_default = np.zeros(3, dtype=np.float64)
    if "q_club" in coords:
        idx = coords.index("q_club")
        omega_default[2] = float(snapshot.v[idx])
        alpha_default[2] = float(snapshot.a[idx])
    elif snapshot.nv() >= 3:
        omega_default = np.asarray(snapshot.v[-3:], dtype=np.float64)
        alpha_default = np.asarray(snapshot.a[-3:], dtype=np.float64)

    omega = (
        omega_default
        if club_ang_vel_world_rad_s is None
        else _readonly_float_array(
            club_ang_vel_world_rad_s, "club_ang_vel_world_rad_s", shape=(3,)
        )
    )
    alpha = (
        alpha_default
        if club_ang_accel_world_rad_s2 is None
        else _readonly_float_array(
            club_ang_accel_world_rad_s2, "club_ang_accel_world_rad_s2", shape=(3,)
        )
    )
    a_com_default = np.cross(alpha, r_rel) + np.cross(omega, np.cross(omega, r_rel))
    a_com = (
        a_com_default
        if club_com_accel_world_m_s2 is None
        else _readonly_float_array(
            club_com_accel_world_m_s2, "club_com_accel_world_m_s2", shape=(3,)
        )
    )
    return a_com, omega, alpha


def net_club_wrench(
    snapshot: ClosedLoopSnapshot,
    *,
    club_com_accel_world_m_s2: ArrayLike | None = None,
    club_ang_vel_world_rad_s: ArrayLike | None = None,
    club_ang_accel_world_rad_s2: ArrayLike | None = None,
    gravity_world_m_s2: ArrayLike = (0.0, 0.0, -9.81),
) -> QuantityRecord:
    """Compute the bare-club net wrench (INVARIANT, C-FRAME-GRIP) from Newton-Euler dynamics."""
    if not isinstance(snapshot, ClosedLoopSnapshot):
        raise TypeError("snapshot must be a ClosedLoopSnapshot instance")
    g_world = _readonly_float_array(
        gravity_world_m_s2, "gravity_world_m_s2", shape=(3,)
    )
    a_com, omega, alpha = _infer_club_kinematics(
        snapshot,
        club_com_accel_world_m_s2,
        club_ang_vel_world_rad_s,
        club_ang_accel_world_rad_s2,
    )

    club = snapshot.bare_club
    f_world = club.mass_kg * (a_com - g_world)
    i_world = club.inertia_com_kg_m2
    m_world = (
        i_world @ alpha
        + np.cross(omega, i_world @ omega)
        + np.cross(club.com_m - snapshot.grip_origin, f_world)
    )

    rot_t = snapshot.grip_frame_rotation.T
    w_grip = np.concatenate([rot_t @ f_world, rot_t @ m_world])
    return QuantityRecord(
        values=w_grip,
        units="N,N*m",
        allocation_dependence=AllocationDependence.INVARIANT,
        conventions=(GRIP_FRAME_CONVENTION_ID,),
        hypotheses=(),
        model_provenance=snapshot.model_provenance,
        input_provenance=snapshot.input_provenance,
    )


__all__ = [
    "GRIP_FRAME_CONVENTION_ID",
    "SPLIT_COORDINATE_NAMES",
    "grasp_map",
    "hand_wrenches",
    "net_club_wrench",
    "split_basis",
    "split_coordinates",
]
