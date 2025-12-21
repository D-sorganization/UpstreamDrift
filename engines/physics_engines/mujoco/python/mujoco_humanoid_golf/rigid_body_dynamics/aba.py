"""
Articulated Body Algorithm (ABA) for forward dynamics.

Computes joint accelerations given applied torques.
"""

from __future__ import annotations

import numpy as np
from mujoco_humanoid_golf.spatial_algebra import cross_force, cross_motion, jcalc

GRAVITY_M_S2 = 9.81
TOLERANCE = 1e-10  # Numerical tolerance to avoid division by zero


def aba(  # noqa: C901, PLR0912, PLR0915
    model: dict,
    q: np.ndarray,
    qd: np.ndarray,
    tau: np.ndarray,
    f_ext: np.ndarray | None = None,
) -> np.ndarray:
    """
    Articulated Body Algorithm for forward dynamics.

    Computes the forward dynamics of a kinematic tree. Given joint positions q,
    velocities qd, and applied torques tau, this algorithm computes the
    resulting joint accelerations qdd.

    This is an O(n) algorithm that is much more efficient than inverting
    the mass matrix.

    Args:
        model: Robot model dictionary with fields:
            NB: Number of bodies (int)
            parent: Parent body indices (array of length NB)
            jtype: Joint types (list of strings, length NB)
            Xtree: Joint transforms (NB-length list of 6x6 arrays)
            I: Spatial inertias (NB-length list of 6x6 arrays)
            gravity: 6x1 spatial gravity vector (optional)
        q: Joint positions (NB,)
        qd: Joint velocities (NB,)
        tau: Joint forces/torques (NB,)
        f_ext: External forces (6, NB) (optional)

    Returns:
        Joint accelerations (NB,)

    Algorithm:
        Pass 1: Kinematics - compute velocities and bias accelerations
        Pass 2: Articulated bodies - compute articulated-body inertias
        Pass 3: Accelerations - compute joint and spatial accelerations

    References:
        Featherstone, R. (2008). Rigid Body Dynamics Algorithms.
        Chapter 7: Articulated-Body Algorithm, Algorithm 7.1

    Example:
        >>> model = create_robot_model()
        >>> q = np.array([0.5, -0.3])
        >>> qd = np.array([0.1, 0.2])
        >>> tau = np.array([1.5, 0.5])
        >>> qdd = aba(model, q, qd, tau)
    """
    # Use ravel() to avoid copying data when possible
    q = np.asarray(q).ravel()
    qd = np.asarray(qd).ravel()
    tau = np.asarray(tau).ravel()

    nb = model["NB"]

    if len(q) != nb:
        msg = f"q must have length {nb}, got {len(q)}"
        raise ValueError(msg)
    if len(qd) != nb:
        msg = f"qd must have length {nb}, got {len(qd)}"
        raise ValueError(msg)
    if len(tau) != nb:
        msg = f"tau must have length {nb}, got {len(tau)}"
        raise ValueError(msg)

    if f_ext is None:
        f_ext = np.zeros((6, nb))

    # Get gravity vector
    a_grav = model.get("gravity", np.array([0, 0, 0, 0, 0, -GRAVITY_M_S2]))

    # Initialize arrays
    xup: list[np.ndarray] = [None] * nb  # type: ignore[assignment, list-item] # Transforms from body to parent
    s_subspace: list[np.ndarray] = [None] * nb  # type: ignore[assignment, list-item] # Motion subspaces
    v = np.zeros((6, nb))  # Spatial velocities
    c = np.zeros((6, nb))  # Velocity-product accelerations (bias)
    ia_articulated: list[np.ndarray] = [None] * nb  # type: ignore[assignment, list-item] # Articulated-body inertias
    pa_bias = np.zeros((6, nb))  # Articulated-body bias forces
    u_force = np.zeros((6, nb))  # IA * S
    d = np.zeros(nb)  # S.T @ U (joint-space inertia)
    u = np.zeros(nb)  # tau - S.T @ pA (bias force)
    a = np.zeros((6, nb))  # Spatial accelerations
    qdd = np.zeros(nb)  # Joint accelerations

    # --- Pass 1: Forward kinematics ---
    for i in range(nb):
        xj_transform, s_subspace[i] = jcalc(model["jtype"][i], q[i])
        xup[i] = xj_transform @ model["Xtree"][i]

        vj_velocity = s_subspace[i] * qd[i]  # Joint velocity

        if model["parent"][i] == -1:
            v[:, i] = vj_velocity
            c[:, i] = np.zeros(6)  # No bias for base-connected bodies
        else:
            p = model["parent"][i]
            v[:, i] = xup[i] @ v[:, p] + vj_velocity
            c[:, i] = cross_motion(
                v[:, i], vj_velocity
            )  # Velocity-product acceleration

        # Initialize articulated-body inertia with rigid-body inertia
        ia_articulated[i] = model["I"][i].copy()

        # Bias force: Coriolis + external forces
        pa_bias[:, i] = cross_force(v[:, i], model["I"][i] @ v[:, i]) - f_ext[:, i]

    # --- Pass 2: Backward recursion (articulated-body inertias) ---
    for i in range(nb - 1, -1, -1):
        u_force[:, i] = ia_articulated[i] @ s_subspace[i]
        d[i] = s_subspace[i] @ u_force[:, i]  # Joint-space inertia
        u[i] = tau[i] - s_subspace[i] @ pa_bias[:, i]  # Bias torque

        # Articulated-body inertia update for parent
        if model["parent"][i] != -1:
            p = model["parent"][i]

            # Inverse of joint-space inertia
            # For 1-DOF joints, this is just 1/d(i)
            if abs(d[i]) < TOLERANCE:
                d[i] = np.sign(d[i]) * TOLERANCE if d[i] != 0 else TOLERANCE

            dinv = 1 / d[i]

            # Update articulated inertia
            ia_update = np.outer(u_force[:, i], u_force[:, i]) * dinv
            ia_articulated[p] = (
                ia_articulated[p] + xup[i].T @ (ia_articulated[i] - ia_update) @ xup[i]
            )

            # Update bias force
            pa_update = (
                pa_bias[:, i]
                + ia_articulated[i] @ c[:, i]
                + u_force[:, i] * dinv * u[i]
            )
            pa_bias[:, p] = pa_bias[:, p] + xup[i].T @ pa_update

    # --- Pass 3: Forward recursion (accelerations) ---
    for i in range(nb):
        if model["parent"][i] == -1:
            # For base-connected bodies, apply gravity through Xup transform
            a[:, i] = xup[i] @ (-a_grav) + c[:, i]
        else:
            p = model["parent"][i]
            a[:, i] = xup[i] @ a[:, p] + c[:, i]

        qdd[i] = (u[i] - u_force[:, i] @ a[:, i]) / d[i]
        a[:, i] = a[:, i] + s_subspace[i] * qdd[i]

    return qdd
