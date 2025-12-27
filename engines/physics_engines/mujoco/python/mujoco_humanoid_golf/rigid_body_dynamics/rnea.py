"""
Recursive Newton-Euler Algorithm (RNEA) for inverse dynamics.

Computes the joint forces/torques required to produce a given motion.
"""

from __future__ import annotations

import numpy as np
from mujoco_humanoid_golf.spatial_algebra import (
    cross_force_fast,
    cross_motion_fast,
    jcalc,
)
from shared.python import constants

DEFAULT_GRAVITY = np.array([0, 0, 0, 0, 0, -constants.GRAVITY_M_S2])
DEFAULT_GRAVITY.flags.writeable = False


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
    a_grav = model.get("gravity", DEFAULT_GRAVITY)

    # Initialize arrays
    # OPTIMIZATION: use np.empty instead of np.zeros for arrays that are fully
    # overwritten
    v = np.empty((6, nb))  # Spatial velocities
    a = np.empty((6, nb))  # Spatial accelerations
    f = np.zeros((6, nb))  # Spatial forces (must be zero for accumulation)
    tau = np.empty(nb)  # Joint torques

    # OPTIMIZATION: Pre-allocate buffers
    # Stores transform from parent to i for each body
    # Using a single 3D array is more cache-friendly than list of arrays
    xup = np.empty((nb, 6, 6))

    # Pre-allocate temporary buffers to avoid allocation in loop
    xj_buf = np.empty((6, 6))
    scratch_vec = np.empty(6)
    i_v_buf = np.empty(6)
    cross_buf = np.empty(6)

    s_subspace_list: list[np.ndarray] = [None] * nb  # type: ignore[assignment, list-item] # Cache motion subspaces

    # --- Forward pass: kinematics ---
    for i in range(nb):
        # Calculate joint transform and motion subspace
        # OPTIMIZATION: Use pre-allocated buffer
        xj_transform, s_subspace = jcalc(model["jtype"][i], q[i], out=xj_buf)
        s_subspace_list[i] = s_subspace

        # Joint velocity in joint frame
        # OPTIMIZATION: Use pre-allocated buffer (reuse i_v_buf)
        # vj_velocity = s_subspace * qd[i] (Avoids allocation)
        np.multiply(s_subspace, qd[i], out=i_v_buf)
        vj_velocity = i_v_buf

        # Composite transform from body i to parent/base
        if model["parent"][i] == -1:  # Python uses -1 for no parent
            # Body i is connected to base
            # Use Xj directly (not Xj * Xtree) per MATLAB reference
            v[:, i] = vj_velocity

            # Optimized a[:, i] = xj_transform @ (-a_grav) + s_subspace * qdd[i]
            np.matmul(xj_transform, -a_grav, out=scratch_vec)
            # Optimization: Avoid allocation for s_subspace * qdd[i]
            # scratch_vec += s_subspace * qdd[i]
            np.multiply(s_subspace, qdd[i], out=cross_buf)
            scratch_vec += cross_buf
            a[:, i] = scratch_vec
        else:
            # Body i has a parent
            p = model["parent"][i]

            # Optimized xp_transform = xj_transform @ model["Xtree"][i]
            # Write directly to pre-allocated xup buffer
            np.matmul(xj_transform, model["Xtree"][i], out=xup[i])

            # Velocity: transform parent velocity and add joint velocity
            # Optimized v[:, i] = xup[i] @ v[:, p] + vj_velocity
            np.matmul(xup[i], v[:, p], out=scratch_vec)
            scratch_vec += vj_velocity
            v[:, i] = scratch_vec

            # Acceleration: transform parent accel + bias accel + joint accel
            # Optimized a[:, i] = (xup[i] @ a[:, p] + ... )
            np.matmul(xup[i], a[:, p], out=scratch_vec)

            # Optimization: Avoid allocation for s_subspace * qdd[i]
            # Use cross_buf as temporary buffer before it's needed for cross_motion
            # scratch_vec += s_subspace * qdd[i]
            np.multiply(s_subspace, qdd[i], out=cross_buf)
            scratch_vec += cross_buf

            # Optimization: Use pre-allocated buffer for cross product
            # Overwrites cross_buf, which is fine as we are done with qdd term
            cross_motion_fast(v[:, i], vj_velocity, out=cross_buf)
            scratch_vec += cross_buf
            a[:, i] = scratch_vec

    # --- Backward pass: dynamics ---
    for i in range(nb - 1, -1, -1):
        # Newton-Euler equation: f = I*a + v x* I*v - f_ext
        # Compute body force using optimized buffers
        # 1. inertia @ accel
        np.matmul(model["I"][i], a[:, i], out=scratch_vec)
        f_body = scratch_vec  # Alias (copy will happen on += next if we aren't careful)

        # 2. inertia @ vel -> i_v_buf
        np.matmul(model["I"][i], v[:, i], out=i_v_buf)

        # 3. Add Coriolis (cross_force allocates, but we add to buffer)
        # Optimization: Use pre-allocated buffer for cross product
        cross_force_fast(v[:, i], i_v_buf, out=cross_buf)
        f_body += cross_buf
        f_body -= f_ext[:, i]

        # Accumulate with any forces already propagated from children
        # (f is initialized to zero, but children may have already propagated forces)
        f[:, i] += f_body

        # Project force to joint torque
        s_subspace = s_subspace_list[i]
        tau[i] = s_subspace @ f[:, i]

        # Propagate force to parent
        if model["parent"][i] != -1:
            p = model["parent"][i]
            # Optimized f[:, p] = f[:, p] + xup[i].T @ f[:, i]
            np.matmul(xup[i].T, f[:, i], out=scratch_vec)
            f[:, p] += scratch_vec

    return tau
