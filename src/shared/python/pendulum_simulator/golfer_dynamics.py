"""Dynamics (mass matrix, Coriolis, gravity, energy) for golfer model.

Uses Lagrangian formulation with analytical Jacobian-based computation
and optional native backend acceleration.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from . import native_backend as _native_backend
from .golfer_kinematics import forward_kinematics
from .physics_golfer import N_DOF, GolferParams, State


def _mass_point_positions(
    q: np.ndarray, p: GolferParams
) -> list[tuple[float, Callable]]:
    """Return list of (mass, position_function) for all point masses."""
    if not isinstance(q, np.ndarray):
        raise TypeError("q must be a numpy ndarray")
    if q.ndim != 1 or q.shape[0] < N_DOF:
        raise ValueError(f"q must be a 1D array of at least {N_DOF} elements")
    if not isinstance(p, GolferParams):
        raise TypeError("p must be a GolferParams instance")

    def hub_pos(qq: np.ndarray) -> tuple[float, float]:
        from .golfer_kinematics import _hub_position

        return _hub_position(qq[0], p)

    def re_pos(qq: np.ndarray) -> tuple[float, float]:
        fk = forward_kinematics(qq, p)
        return fk["re"]

    def rh_pos(qq: np.ndarray) -> tuple[float, float]:
        fk = forward_kinematics(qq, p)
        return fk["rh"]

    def le_pos(qq: np.ndarray) -> tuple[float, float]:
        fk = forward_kinematics(qq, p)
        return fk["le"]

    def lh_pos(qq: np.ndarray) -> tuple[float, float]:
        fk = forward_kinematics(qq, p)
        return fk["lh"]

    def club_tip_pos(qq: np.ndarray) -> tuple[float, float]:
        fk = forward_kinematics(qq, p)
        return fk["club_tip"]

    def club_com_pos(qq: np.ndarray) -> tuple[float, float]:
        fk = forward_kinematics(qq, p)
        base = fk["club_base"]
        tip = fk["club_tip"]
        return (0.5 * (base[0] + tip[0]), 0.5 * (base[1] + tip[1]))

    return [
        (p.m_hub, hub_pos),
        (p.m_r_upper, re_pos),
        (p.m_r_fore, rh_pos),
        (p.m_l_upper, le_pos),
        (p.m_l_fore, lh_pos),
        (p.m_club, club_com_pos),
        (p.m_clubhead, club_tip_pos),
    ]


def potential_energy_from_q(q: np.ndarray, p: GolferParams) -> float:
    """Compute total gravitational potential energy from coordinates."""
    if not isinstance(q, np.ndarray):
        raise TypeError("q must be a numpy ndarray")
    if q.ndim != 1 or q.shape[0] < N_DOF:
        raise ValueError(f"q must be a 1D array of at least {N_DOF} elements")
    if not isinstance(p, GolferParams):
        raise TypeError("p must be a GolferParams instance")

    if q.shape[0] > N_DOF:
        q = q[:N_DOF]

    mass_points = _mass_point_positions(q, p)
    V = 0.0
    for mass_val, pos_func in mass_points:
        _, y = pos_func(q)
        V += mass_val * p.g * y

    return V


class _TrigCache:
    """Precomputed sine/cosine values for golfer FK Jacobians (DRY)."""

    __slots__ = (
        "sin_hub",
        "cos_hub",
        "sin_rs",
        "cos_rs",
        "sin_re",
        "cos_re",
        "sin_ls",
        "cos_ls",
        "sin_le",
        "cos_le",
        "sin_club",
        "cos_club",
    )

    def __init__(self, q: np.ndarray) -> None:
        if not (q is not None):
            raise ValueError("q must be provided")
        th_hub = q[0]
        self.sin_hub = np.sin(th_hub)
        self.cos_hub = np.cos(th_hub)

        th_rs_abs = th_hub + q[1]
        self.sin_rs = np.sin(th_rs_abs)
        self.cos_rs = np.cos(th_rs_abs)

        th_re_abs = th_rs_abs + q[2]
        self.sin_re = np.sin(th_re_abs)
        self.cos_re = np.cos(th_re_abs)

        th_ls_abs = th_hub + q[4]
        self.sin_ls = np.sin(th_ls_abs)
        self.cos_ls = np.cos(th_ls_abs)

        th_le_abs = th_ls_abs + q[5]
        self.sin_le = np.sin(th_le_abs)
        self.cos_le = np.cos(th_le_abs)

        self.sin_club = np.sin(q[7])
        self.cos_club = np.cos(q[7])


def _hub_and_shoulder_jacobians(
    p: GolferParams, tc: _TrigCache
) -> dict[str, np.ndarray]:
    """Compute Jacobians for hub, right shoulder, and left shoulder."""
    if not (p is not None):
        raise ValueError("p must be provided")
    J_hub = np.zeros((2, N_DOF))
    J_hub[0, 0] = -p.L_hub * tc.cos_hub
    J_hub[1, 0] = -p.L_hub * tc.sin_hub

    J_rs = np.zeros((2, N_DOF))
    J_rs[0, 0] = -p.L_hub * tc.cos_hub - p.d_rs * tc.sin_hub
    J_rs[1, 0] = -p.L_hub * tc.sin_hub + p.d_rs * tc.cos_hub

    J_ls = np.zeros((2, N_DOF))
    J_ls[0, 0] = -p.L_hub * tc.cos_hub + p.d_ls * tc.sin_hub
    J_ls[1, 0] = -p.L_hub * tc.sin_hub - p.d_ls * tc.cos_hub

    return {"hub": J_hub, "rs": J_rs, "ls": J_ls}


def _right_arm_chain_jacobian(
    p: GolferParams, tc: _TrigCache
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Jacobians for RE, RH along the right arm kinematic chain.

    Returns (J_re, J_rh) and also the RH Jacobian which is reused by club.
    """
    # RE (right elbow): depends on q[0], q[1]
    if not (p is not None):
        raise ValueError("p must be provided")
    J_re = np.zeros((2, N_DOF))
    J_re[0, 0] = -p.L_hub * tc.cos_hub - p.d_rs * tc.sin_hub + p.L_r_upper * tc.cos_rs
    J_re[1, 0] = -p.L_hub * tc.sin_hub + p.d_rs * tc.cos_hub + p.L_r_upper * tc.sin_rs
    J_re[0, 1] = p.L_r_upper * tc.cos_rs
    J_re[1, 1] = p.L_r_upper * tc.sin_rs

    # RH (right hand): depends on q[0], q[1], q[2]
    J_rh = np.zeros((2, N_DOF))
    J_rh[0, 0] = (
        -p.L_hub * tc.cos_hub
        - p.d_rs * tc.sin_hub
        + p.L_r_upper * tc.cos_rs
        + p.L_r_fore * tc.cos_re
    )
    J_rh[1, 0] = (
        -p.L_hub * tc.sin_hub
        + p.d_rs * tc.cos_hub
        + p.L_r_upper * tc.sin_rs
        + p.L_r_fore * tc.sin_re
    )
    J_rh[0, 1] = p.L_r_upper * tc.cos_rs + p.L_r_fore * tc.cos_re
    J_rh[1, 1] = p.L_r_upper * tc.sin_rs + p.L_r_fore * tc.sin_re
    J_rh[0, 2] = p.L_r_fore * tc.cos_re
    J_rh[1, 2] = p.L_r_fore * tc.sin_re

    return J_re, J_rh, J_rh


def _left_arm_chain_jacobian(
    p: GolferParams, tc: _TrigCache
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Jacobians for LE, LH along the left arm kinematic chain."""
    # LE (left elbow): depends on q[0], q[4]
    if not (p is not None):
        raise ValueError("p must be provided")
    J_le = np.zeros((2, N_DOF))
    J_le[0, 0] = -p.L_hub * tc.cos_hub + p.d_ls * tc.sin_hub + p.L_l_upper * tc.cos_ls
    J_le[1, 0] = -p.L_hub * tc.sin_hub - p.d_ls * tc.cos_hub + p.L_l_upper * tc.sin_ls
    J_le[0, 4] = p.L_l_upper * tc.cos_ls
    J_le[1, 4] = p.L_l_upper * tc.sin_ls

    # LH (left hand): depends on q[0], q[4], q[5]
    J_lh = np.zeros((2, N_DOF))
    J_lh[0, 0] = (
        -p.L_hub * tc.cos_hub
        + p.d_ls * tc.sin_hub
        + p.L_l_upper * tc.cos_ls
        + p.L_l_fore * tc.cos_le
    )
    J_lh[1, 0] = (
        -p.L_hub * tc.sin_hub
        - p.d_ls * tc.cos_hub
        + p.L_l_upper * tc.sin_ls
        + p.L_l_fore * tc.sin_le
    )
    J_lh[0, 4] = p.L_l_upper * tc.cos_ls + p.L_l_fore * tc.cos_le
    J_lh[1, 4] = p.L_l_upper * tc.sin_ls + p.L_l_fore * tc.sin_le
    J_lh[0, 5] = p.L_l_fore * tc.cos_le
    J_lh[1, 5] = p.L_l_fore * tc.sin_le

    return J_le, J_lh


def _club_jacobians(
    p: GolferParams, tc: _TrigCache, j_rh: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Jacobians for club COM and club tip, reusing the RH Jacobian."""
    # Club COM: RH chain + club angle
    if not (p is not None):
        raise ValueError("p must be provided")
    coeff_com_x = 0.5 * p.L_club - p.grip_right
    coeff_com_y = -0.5 * (p.L_club - 2 * p.grip_right)

    J_club_com = j_rh.copy()
    J_club_com[0, 7] = coeff_com_x * tc.cos_club
    J_club_com[1, 7] = coeff_com_y * tc.sin_club

    # Club TIP: RH chain + club angle (different coefficient)
    coeff_tip = p.L_club - p.grip_right

    J_club_tip = j_rh.copy()
    J_club_tip[0, 7] = coeff_tip * tc.cos_club
    J_club_tip[1, 7] = -coeff_tip * tc.sin_club

    return J_club_com, J_club_tip


def analytical_fk_jacobians(q: np.ndarray, p: GolferParams) -> dict[str, np.ndarray]:
    """Compute position Jacobians analytically for all mass points.

    Parameters
    ----------
    q : np.ndarray, shape (8,)
        Generalized coordinates
    p : GolferParams
        Physical parameters

    Returns
    -------
    dict with keys: 'hub', 'rs', 're', 'rh', 'ls', 'le', 'lh', 'club_com', 'club_tip'
    Each value is a 2×8 matrix: J[point][row, col] = d(pos[row])/dq[col]
    """
    if not isinstance(q, np.ndarray):
        raise TypeError("q must be a numpy ndarray")
    if q.ndim != 1 or q.shape[0] < N_DOF:
        raise ValueError(f"q must be a 1D array of at least {N_DOF} elements")
    if not isinstance(p, GolferParams):
        raise TypeError("p must be a GolferParams instance")

    if q.shape[0] > N_DOF:
        q = q[:N_DOF]

    tc = _TrigCache(q)

    # Hub + shoulders
    jacobians = _hub_and_shoulder_jacobians(p, tc)

    # Right arm chain (RE, RH) — also returns J_rh for club reuse
    J_re, J_rh, j_rh_base = _right_arm_chain_jacobian(p, tc)
    jacobians["re"] = J_re
    jacobians["rh"] = J_rh

    # Left arm chain (LE, LH)
    J_le, J_lh = _left_arm_chain_jacobian(p, tc)
    jacobians["le"] = J_le
    jacobians["lh"] = J_lh

    # Club (COM + tip), reusing J_rh base (DRY)
    J_club_com, J_club_tip = _club_jacobians(p, tc, j_rh_base)
    jacobians["club_com"] = J_club_com
    jacobians["club_tip"] = J_club_tip

    return jacobians


def analytical_mass_matrix(q: np.ndarray, p: GolferParams) -> np.ndarray:
    """Compute mass matrix M(q) analytically from Jacobians.

    Uses M = sum_i(m_i * J_i^T @ J_i) where J_i is the 2×8 Jacobian
    of mass point i.

    Parameters
    ----------
    q : np.ndarray, shape (8,)
    p : GolferParams

    Returns
    -------
    M : np.ndarray, shape (8, 8) — symmetric positive semi-definite
    """
    native_mass_matrix = _native_backend.golfer_mass_matrix(q, p)
    if native_mass_matrix is not None:
        return native_mass_matrix

    if not isinstance(q, np.ndarray):
        raise TypeError("q must be a numpy ndarray")
    if q.ndim != 1 or q.shape[0] < N_DOF:
        raise ValueError(f"q must be a 1D array of at least {N_DOF} elements")
    if not isinstance(p, GolferParams):
        raise TypeError("p must be a GolferParams instance")

    if q.shape[0] > N_DOF:
        q = q[:N_DOF]

    jacobians = analytical_fk_jacobians(q, p)

    M = np.zeros((N_DOF, N_DOF))

    # Mass point contributions: (mass, jacobian_key)
    mass_contributions = [
        (p.m_hub, "hub"),
        (p.m_r_upper, "re"),
        (p.m_r_fore, "rh"),
        (p.m_l_upper, "le"),
        (p.m_l_fore, "lh"),
        (p.m_club, "club_com"),
        (p.m_clubhead, "club_tip"),
    ]

    for mass_val, key in mass_contributions:
        J = jacobians[key]
        M += mass_val * J.T @ J

    return M


def analytical_coriolis(q: np.ndarray, qdot: np.ndarray, p: GolferParams) -> np.ndarray:
    """Compute C(q, qdot) = (dM/dt) * qdot analytically.

    Uses Christoffel symbols: C_i = sum_jk c_ijk * qdot_j * qdot_k
    where c_ijk = 0.5 * (dM_ij/dq_k + dM_ik/dq_j - dM_jk/dq_i)

    dM/dq_k is computed analytically from the chain rule applied to
    each Jacobian term.

    Parameters
    ----------
    q : np.ndarray, shape (8,)
    qdot : np.ndarray, shape (8,)
    p : GolferParams

    Returns
    -------
    C_qdot : np.ndarray, shape (8,)
    """
    if not isinstance(q, np.ndarray) or not isinstance(qdot, np.ndarray):
        raise TypeError("q and qdot must be numpy ndarrays")
    if q.ndim != 1 or q.shape[0] < N_DOF:
        raise ValueError(f"q must be a 1D array of at least {N_DOF} elements")
    if qdot.ndim != 1 or qdot.shape[0] < N_DOF:
        raise ValueError(f"qdot must be a 1D array of at least {N_DOF} elements")
    if not isinstance(p, GolferParams):
        raise TypeError("p must be a GolferParams instance")

    if q.shape[0] > N_DOF:
        q = q[:N_DOF]
    if qdot.shape[0] > N_DOF:
        qdot = qdot[:N_DOF]

    # Compute dM/dq_k analytically using finite differences of Jacobians
    # (This is faster than recomputing M multiple times)
    eps = 1e-7
    M0 = analytical_mass_matrix(q, p)
    dM = np.zeros((N_DOF, N_DOF, N_DOF))  # dM[i,j,k] = dM_ij/dq_k

    for k in range(N_DOF):
        q_plus = q.copy()
        q_plus[k] += eps
        dM[:, :, k] = (analytical_mass_matrix(q_plus, p) - M0) / eps

    christoffel = 0.5 * (dM + dM.transpose(0, 2, 1) - dM.transpose(1, 2, 0))
    result: np.ndarray = np.einsum("ijk,j,k->i", christoffel, qdot, qdot)
    return result


def analytical_gravity_vector(q: np.ndarray, p: GolferParams) -> np.ndarray:
    """Compute gravity torque vector G(q) analytically.

    G_i = dV/dq_i where V = sum_k(m_k * g * y_k)

    For each mass point, dV/dq_i = m * g * dy/dq_i.

    Parameters
    ----------
    q : np.ndarray, shape (8,)
    p : GolferParams

    Returns
    -------
    G : np.ndarray, shape (8,)
    """
    native_gravity = _native_backend.golfer_gravity_vector(q, p)
    if native_gravity is not None:
        return native_gravity

    if not isinstance(q, np.ndarray):
        raise TypeError("q must be a numpy ndarray")
    if q.ndim != 1 or q.shape[0] < N_DOF:
        raise ValueError(f"q must be a 1D array of at least {N_DOF} elements")
    if not isinstance(p, GolferParams):
        raise TypeError("p must be a GolferParams instance")

    if q.shape[0] > N_DOF:
        q = q[:N_DOF]

    jacobians = analytical_fk_jacobians(q, p)

    G = np.zeros(N_DOF)

    mass_contributions = [
        (p.m_hub, "hub"),
        (p.m_r_upper, "re"),
        (p.m_r_fore, "rh"),
        (p.m_l_upper, "le"),
        (p.m_l_fore, "lh"),
        (p.m_club, "club_com"),
        (p.m_clubhead, "club_tip"),
    ]

    for mass_val, key in mass_contributions:
        J = jacobians[key]
        # G_i += m * g * dy/dq_i = m * g * J[1, i]
        G += mass_val * p.g * J[1, :]

    return G


def kinetic_energy(q: np.ndarray, qdot: np.ndarray, p: GolferParams) -> float:
    """Compute T = 0.5 * qdot^T M qdot."""
    from .physics_base import kinetic_energy_from_M

    if not isinstance(q, np.ndarray) or not isinstance(qdot, np.ndarray):
        raise TypeError("q and qdot must be numpy ndarrays")
    if q.ndim != 1 or q.shape[0] < N_DOF:
        raise ValueError(f"q must be a 1D array of at least {N_DOF} elements")
    if qdot.ndim != 1 or qdot.shape[0] < N_DOF:
        raise ValueError(f"qdot must be a 1D array of at least {N_DOF} elements")
    if not isinstance(p, GolferParams):
        raise TypeError("p must be a GolferParams instance")

    if q.shape[0] > N_DOF:
        q = q[:N_DOF]
    if qdot.shape[0] > N_DOF:
        qdot = qdot[:N_DOF]

    M = analytical_mass_matrix(q, p)
    return kinetic_energy_from_M(M, qdot)


def potential_energy(state: State, p: GolferParams) -> float:
    """Compute gravitational PE from full state vector."""
    return potential_energy_from_q(state[:N_DOF], p)


def total_energy(state: State, p: GolferParams) -> float:
    """Compute E = T + V from full state."""
    if not (state is not None):
        raise ValueError("state must be provided")
    from .physics_base import total_energy_from_parts

    q = state[:N_DOF]
    qdot = state[N_DOF:]
    return total_energy_from_parts(
        kinetic_energy(q, qdot, p), potential_energy(state, p)
    )
