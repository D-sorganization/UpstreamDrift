"""
Recursive Newton-Euler Algorithm (RNEA) for inverse dynamics.

Computes the joint forces/torques required to produce a given motion.
"""

from __future__ import annotations

import numpy as np
from mujoco_humanoid_golf.spatial_algebra import cross_force, cross_motion, jcalc

GRAVITY_M_S2 = 9.81


def rnea(  # noqa: PLR0915
    model: dict,
    q: np.ndarray,
    qd: np.ndarray,
    qdd: np.ndarray,
    f_ext: np.ndarray | None = None,
) -> np.ndarray:
    """
    Recursive Newton-Euler Algorithm for inverse dynamics.

    Computes the inverse dynamics of a kinematic tree. Given joint positions q,
    velocities qd, and accelerations qdd, this algorithm computes the joint
    forces/torques tau required to produce that motion.

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
        qdd: Joint accelerations (NB,)
        f_ext: External forces (6, NB) (optional)

    Returns:
        Joint forces/torques (NB,)

    Algorithm:
        Forward pass: compute velocities and accelerations
        Backward pass: compute forces and project to joint torques

    References:
        Featherstone, R. (2008). Rigid Body Dynamics Algorithms.
        Chapter 5: Independent Joint Equations of Motion, Algorithm 5.1

    Example:
        >>> model = create_robot_model()
        >>> q = np.array([0.5, -0.3])
        >>> qd = np.array([0.1, 0.2])
        >>> qdd = np.array([0.5, -0.2])
        >>> tau = rnea(model, q, qd, qdd)
    """
    # Use ravel() to avoid copying data when possible
    q = np.asarray(q).ravel()
    qd = np.asarray(qd).ravel()
    qdd = np.asarray(qdd).ravel()

    nb = model["NB"]

    if len(q) != nb:
        msg = f"q must have length {nb}, got {len(q)}"
        raise ValueError(msg)
    if len(qd) != nb:
        msg = f"qd must have length {nb}, got {len(qd)}"
        raise ValueError(msg)
    if len(qdd) != nb:
        msg = f"qdd must have length {nb}, got {len(qdd)}"
        raise ValueError(msg)

    if f_ext is None:
        f_ext = np.zeros((6, nb))

    # Get gravity vector
    a_grav = model.get("gravity", np.array([0, 0, 0, 0, 0, -GRAVITY_M_S2]))

    # Initialize arrays
    v = np.zeros((6, nb))  # Spatial velocities
    a = np.zeros((6, nb))  # Spatial accelerations
    f = np.zeros((6, nb))  # Spatial forces
    tau = np.zeros(nb)  # Joint torques
    xup: list[np.ndarray] = [None] * nb  # type: ignore[assignment, list-item] # Cache transforms
    s_subspace_list: list[np.ndarray] = [None] * nb  # type: ignore[assignment, list-item] # Cache motion subspaces

    # --- Forward pass: kinematics ---
    for i in range(nb):
        # Calculate joint transform and motion subspace
        xj_transform, s_subspace = jcalc(model["jtype"][i], q[i])
        s_subspace_list[i] = s_subspace

        # Joint velocity in joint frame
        vj_velocity = s_subspace * qd[i]

        # Composite transform from body i to parent/base
        if model["parent"][i] == -1:  # Python uses -1 for no parent
            # Body i is connected to base
            # Use Xj directly (not Xj * Xtree) per MATLAB reference
            v[:, i] = vj_velocity
            a[:, i] = xj_transform @ (-a_grav) + s_subspace * qdd[i]
        else:
            # Body i has a parent
            p = model["parent"][i]
            xp_transform = (
                xj_transform @ model["Xtree"][i]
            )  # Transform from parent to i
            xup[i] = xp_transform

            # Velocity: transform parent velocity and add joint velocity
            v[:, i] = xp_transform @ v[:, p] + vj_velocity

            # Acceleration: transform parent accel + bias accel + joint accel
            a[:, i] = (
                xp_transform @ a[:, p]
                + s_subspace * qdd[i]
                + cross_motion(v[:, i], vj_velocity)
            )

    # --- Backward pass: dynamics ---
    for i in range(nb - 1, -1, -1):
        # Newton-Euler equation: f = I*a + v x* I*v - f_ext
        # Compute body force
        f_body = (
            model["I"][i] @ a[:, i]
            + cross_force(v[:, i], model["I"][i] @ v[:, i])
            - f_ext[:, i]
        )

        # Accumulate with any forces already propagated from children
        # (f is initialized to zero, but children may have already propagated forces)
        f[:, i] = f[:, i] + f_body

        # Project force to joint torque
        s_subspace = s_subspace_list[i]
        tau[i] = s_subspace @ f[:, i]

        # Propagate force to parent
        if model["parent"][i] != -1:
            p = model["parent"][i]
            xp_transform = xup[i]
            f[:, p] = f[:, p] + xp_transform.T @ f[:, i]

    return tau
