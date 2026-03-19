"""Constraint handling and joint force computation for golfer model.

Implements closed kinematic loop constraints and methods to compute
forces and accelerations at joints.
"""

from __future__ import annotations

import numpy as np

from .golfer_dynamics import analytical_fk_jacobians
from .golfer_kinematics import forward_kinematics
from .physics_golfer import N_DOF, GolferParams


def constraint_vector(q: np.ndarray, p: GolferParams) -> np.ndarray:
    """Evaluate the 4 loop-closure constraint equations.

    Phi(q) = 0 when the loop is closed:
        Phi[0:2] = RH_position - club_grip_right_position = 0
        Phi[2:4] = LH_position - club_grip_left_position = 0

    Returns
    -------
    Phi : np.ndarray, shape (4,)
    """
    if not isinstance(q, np.ndarray):
        raise TypeError("q must be a numpy ndarray")
    if q.ndim != 1 or q.shape[0] < N_DOF:
        raise ValueError(f"q must be a 1D array of at least {N_DOF} elements")
    if not isinstance(p, GolferParams):
        raise TypeError("p must be a GolferParams instance")

    if q.shape[0] > N_DOF:
        q = q[:N_DOF]

    fk = forward_kinematics(q, p)

    # Right hand must coincide with right grip on club
    # (already coincides by construction — the club is placed from RH)
    # Left hand must coincide with left grip on club
    phi = np.zeros(4)

    # The real constraint: LH (from left arm chain) must match
    # grip_left (from club geometry anchored at RH)
    phi[0] = fk["lh"][0] - fk["grip_left"][0]
    phi[1] = fk["lh"][1] - fk["grip_left"][1]

    # Additional constraint: right hand position determines club base
    # Club angle must be consistent with the grip positions
    # This is implicit when theta_club is correct, but we enforce it:
    # The vector from grip_right to grip_left on the club must match
    # the vector from RH to LH projected onto the club direction
    rh = np.array(fk["rh"])
    lh_arm = np.array(fk["lh"])
    club_dir = np.array([np.sin(q[7]), -np.cos(q[7])])
    grip_sep = p.grip_left - p.grip_right  # signed distance along club

    # Club direction constraint: RH→LH vector dot club_perp = 0
    club_perp = np.array([-club_dir[1], club_dir[0]])
    rh_to_lh = lh_arm - rh

    phi[2] = np.dot(rh_to_lh, club_perp)  # perpendicular distance = 0
    phi[3] = np.dot(rh_to_lh, club_dir) - grip_sep  # along-club distance

    return phi


def numerical_constraint_jacobian(q: np.ndarray, p: GolferParams) -> np.ndarray:
    """Compute the constraint Jacobian Phi_q = dPhi/dq numerically.

    Parameters
    ----------
    q : np.ndarray, shape (8,) — generalized coordinates

    Returns
    -------
    Phi_q : np.ndarray, shape (4, 8)
    """
    if not isinstance(q, np.ndarray):
        raise TypeError("q must be a numpy ndarray")
    if q.ndim != 1 or q.shape[0] < N_DOF:
        raise ValueError(f"q must be a 1D array of at least {N_DOF} elements")
    if not isinstance(p, GolferParams):
        raise TypeError("p must be a GolferParams instance")

    if q.shape[0] > N_DOF:
        q = q[:N_DOF]

    eps = 1e-7
    Phi_0 = constraint_vector(q, p)
    n_c = len(Phi_0)
    Phi_q = np.zeros((n_c, N_DOF))

    for j in range(N_DOF):
        q_plus = q.copy()
        q_plus[j] += eps
        Phi_q[:, j] = (constraint_vector(q_plus, p) - Phi_0) / eps

    return Phi_q


def analytical_constraint_jacobian(q: np.ndarray, p: GolferParams) -> np.ndarray:
    """Compute constraint Jacobian Phi_q analytically.

    The constraints enforce that the left hand matches the left grip position
    on the club, and that the relative positions satisfy the club geometry.

    Parameters
    ----------
    q : np.ndarray, shape (8,)
    p : GolferParams

    Returns
    -------
    Phi_q : np.ndarray, shape (4, 8)
    """
    if not isinstance(q, np.ndarray):
        raise TypeError("q must be a numpy ndarray")
    if q.ndim != 1 or q.shape[0] < N_DOF:
        raise ValueError(f"q must be a 1D array of at least {N_DOF} elements")
    if not isinstance(p, GolferParams):
        raise TypeError("p must be a GolferParams instance")

    if q.shape[0] > N_DOF:
        q = q[:N_DOF]

    jacobians = analytical_fk_jacobians(q, p)
    J_lh = jacobians["lh"]
    J_rh = jacobians["rh"]

    th_club = q[7]
    sin_club = np.sin(th_club)
    cos_club = np.cos(th_club)

    # Club direction and perpendicular vectors
    club_dir = np.array([sin_club, -cos_club])
    club_perp = np.array([-(-cos_club), sin_club])  # rotated 90° ccw

    # Constraint 1-2: LH position matches grip_left on club
    Phi_q = np.zeros((4, N_DOF))

    # dPhi[0]/dq = dLH_x/dq - d(club_base_x + grip_left*sin(th_club))/dq
    # dPhi[1]/dq = dLH_y/dq - d(club_base_y - grip_left*cos(th_club))/dq
    Phi_q[0, :] = J_lh[0, :]
    Phi_q[1, :] = J_lh[1, :]

    # Subtract RH-dependent part
    Phi_q[0, :] -= J_rh[0, :]
    Phi_q[1, :] -= J_rh[1, :]

    # d((grip_left - grip_right) * sin(th_club))/dq_7 = (grip_left - grip_right) * cos(th_club)
    # d(-(grip_left - grip_right) * cos(th_club))/dq_7 = (grip_left - grip_right) * sin(th_club)
    Phi_q[0, 7] -= (p.grip_left - p.grip_right) * cos_club
    Phi_q[1, 7] += (p.grip_left - p.grip_right) * sin_club

    # Constraint 3: perpendicular distance = 0
    # Phi[2] = (lh - rh) · club_perp = lh_x * club_perp_x + lh_y * club_perp_y
    #                                 - rh_x * club_perp_x - rh_y * club_perp_y
    Phi_q[2, :] = club_perp[0] * (J_lh[0, :] - J_rh[0, :]) + club_perp[1] * (
        J_lh[1, :] - J_rh[1, :]
    )

    # d(club_perp)/dq_7: club_perp = (-(-cos(th_club)), sin(th_club))
    #                                 = (cos(th_club), sin(th_club))
    # d/dq_7: (-sin(th_club), cos(th_club))
    d_club_perp_dth = np.array([-sin_club, cos_club])
    rh_array = np.zeros(2)
    # Evaluate RH position from jacobian: rh = rh_0 + J_rh @ q (but we need pos)
    fk = forward_kinematics(q, p)
    rh_array[0] = fk["rh"][0]
    rh_array[1] = fk["rh"][1]
    lh_array = np.zeros(2)
    lh_array[0] = fk["lh"][0]
    lh_array[1] = fk["lh"][1]

    rh_to_lh = lh_array - rh_array
    Phi_q[2, 7] += np.dot(rh_to_lh, d_club_perp_dth)

    # Constraint 4: along-club distance = grip_sep
    # Phi[3] = (lh - rh) · club_dir - (grip_left - grip_right)
    Phi_q[3, :] = club_dir[0] * (J_lh[0, :] - J_rh[0, :]) + club_dir[1] * (
        J_lh[1, :] - J_rh[1, :]
    )

    # d(club_dir)/dq_7: club_dir = (sin(th_club), -cos(th_club))
    # d/dq_7: (cos(th_club), sin(th_club))
    d_club_dir_dth = np.array([cos_club, sin_club])
    Phi_q[3, 7] += np.dot(rh_to_lh, d_club_dir_dth)

    return Phi_q


def linear_accelerations(
    q: np.ndarray, qdot: np.ndarray, qddot: np.ndarray, p: GolferParams
) -> dict:
    """Compute linear accelerations at all joints via numerical Jacobian.

    Returns dict with keys matching forward_kinematics joint names.
    """
    if (
        not isinstance(q, np.ndarray)
        or not isinstance(qdot, np.ndarray)
        or not isinstance(qddot, np.ndarray)
    ):
        raise TypeError("q, qdot, and qddot must be numpy ndarrays")
    if q.ndim != 1 or q.shape[0] < N_DOF:
        raise ValueError(f"q must be a 1D array of at least {N_DOF} elements")
    if qdot.ndim != 1 or qdot.shape[0] < N_DOF:
        raise ValueError(f"qdot must be a 1D array of at least {N_DOF} elements")
    if qddot.ndim != 1 or qddot.shape[0] < N_DOF:
        raise ValueError(f"qddot must be a 1D array of at least {N_DOF} elements")
    if not isinstance(p, GolferParams):
        raise TypeError("p must be a GolferParams instance")

    if q.shape[0] > N_DOF:
        q = q[:N_DOF]
    if qdot.shape[0] > N_DOF:
        qdot = qdot[:N_DOF]

    eps = 1e-7
    fk0 = forward_kinematics(q, p)
    joint_names = ["hub", "rs", "re", "rh", "ls", "le", "lh", "club_tip"]

    result = {}
    for name in joint_names:
        J = np.zeros((2, N_DOF))
        pos_0 = np.array(fk0[name])
        for j in range(N_DOF):
            q_plus = q.copy()
            q_plus[j] += eps
            fk_j = forward_kinematics(q_plus, p)
            pos_j = np.array(fk_j[name])
            J[:, j] = (pos_j - pos_0) / eps

        # Jdot*qdot via finite difference
        fk_dt = forward_kinematics(q + eps * qdot, p)
        J_dt = np.zeros((2, N_DOF))
        pos_dt = np.array(fk_dt[name])
        for j in range(N_DOF):
            q_plus_dt = (q + eps * qdot).copy()
            q_plus_dt[j] += eps
            fk_jdt = forward_kinematics(q_plus_dt, p)
            pos_jdt = np.array(fk_jdt[name])
            J_dt[:, j] = (pos_jdt - pos_dt) / eps
        Jdot = (J_dt - J) / eps

        acc = J @ qddot + Jdot @ qdot
        result[name] = (float(acc[0]), float(acc[1]))

    return result


def net_joint_forces(
    q: np.ndarray, qdot: np.ndarray, qddot: np.ndarray, p: GolferParams
) -> dict:
    """Compute net force at each joint using Newton's second law.

    F = m * a - m * g_vec for each point mass.

    Returns dict with joint name → (fx, fy) tuples.
    """
    assert q is not None, "q must be provided"
    acc = linear_accelerations(q, qdot, qddot, p)
    g_vec = np.array([0.0, -p.g])

    forces = {}
    joint_mass_map = {
        "hub": p.m_hub,
        "re": p.m_r_upper,
        "rh": p.m_r_fore,
        "le": p.m_l_upper,
        "lh": p.m_l_fore,
        "club_tip": p.m_club + p.m_clubhead,
    }

    for name, mass in joint_mass_map.items():
        a = np.array(acc[name])
        f = mass * a - mass * g_vec
        forces[name] = (float(f[0]), float(f[1]))

    return forces


def friction_torque_vector(qdot: np.ndarray, p: GolferParams) -> np.ndarray:
    """Compute viscous damping torque at all joints.

    Returns
    -------
    tau_f : np.ndarray, shape (8,)
        Note: club DOF (index 7) has no independent damping.
    """
    if not isinstance(qdot, np.ndarray):
        raise TypeError("qdot must be a numpy ndarray")
    if qdot.ndim != 1 or qdot.shape[0] < N_DOF:
        raise ValueError(f"qdot must be a 1D array of at least {N_DOF} elements")
    if not isinstance(p, GolferParams):
        raise TypeError("p must be a GolferParams instance")

    if qdot.shape[0] > N_DOF:
        qdot = qdot[:N_DOF]

    b = np.array([p.b_hub, p.b_rs, p.b_re, p.b_rh, p.b_ls, p.b_le, p.b_lh, 0.0])
    result: np.ndarray = -b * qdot
    return result
