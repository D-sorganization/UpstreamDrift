"""
Articulated Body Algorithm (ABA) for forward dynamics.

Computes joint accelerations given applied torques.
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
    a_grav = model.get("gravity", DEFAULT_GRAVITY)
    # OPTIMIZATION: Pre-compute negative gravity to avoid allocation in loop
    neg_a_grav = -a_grav

    # Initialize arrays
    # OPTIMIZATION: Pre-allocate 3D arrays instead of lists of arrays
    # Stores transform from body to parent for each body (NB, 6, 6)
    xup = np.empty((nb, 6, 6))

    # Motion subspaces (NB, 6)
    # Using list for subspaces is fine (refs to global constants mostly).
    # Be careful if modifying them (jcalc returns new array/ref).
    # Stored in list to be consistent with RNEA optimization.
    s_subspace: list[np.ndarray] = [None] * nb  # type: ignore[assignment, list-item]
    dof_indices: list[int] = [-1] * nb

    v = np.empty((6, nb))  # Spatial velocities
    c = np.empty((6, nb))  # Velocity-product accelerations (bias)

    # Articulated-body inertias (NB, 6, 6)
    # OPTIMIZATION: Pre-allocate to avoid new array creation in loop
    ia_articulated = np.empty((nb, 6, 6))

    pa_bias = np.zeros((6, nb))  # Articulated-body bias forces
    u_force = np.zeros((6, nb))  # IA * S
    d = np.zeros(nb)  # S.T @ U (joint-space inertia)
    u = np.zeros(nb)  # tau - S.T @ pA (bias force)
    a = np.zeros((6, nb))  # Spatial accelerations
    qdd = np.zeros(nb)  # Joint accelerations

    # Optimization: temporary buffers
    xj_buf = np.empty((6, 6))
    cross_buf = np.empty(6)
    scratch_vec = np.empty(6)
    scratch_mat = np.empty((6, 6))
    i_v_buf = np.empty(6)
    temp_vec = np.empty(6)  # Additional scratch vector
    # outer_buf = np.empty((6, 6))  # Unused in favor of xj_buf reuse

    # --- Pass 1: Forward kinematics ---
    for i in range(nb):
        # OPTIMIZATION: Use pre-allocated buffer for jcalc
        xj_transform, s_vec, dof_idx = jcalc(model["jtype"][i], q[i], out=xj_buf)
        s_subspace[i] = s_vec
        dof_indices[i] = dof_idx

        # OPTIMIZATION: Write directly to pre-allocated xup buffer
        np.matmul(xj_transform, model["Xtree"][i], out=xup[i])

        # vj_velocity = s_subspace[i] * qd[i]  # Joint velocity
        if dof_idx != -1:
            temp_vec.fill(0)
            temp_vec[dof_idx] = qd[i]
            vj_velocity = temp_vec
        else:
            vj_velocity = s_subspace[i] * qd[i]

        if model["parent"][i] == -1:
            v[:, i] = vj_velocity
            c[:, i] = np.zeros(6)  # No bias for base-connected bodies
        else:
            p = model["parent"][i]
            # v[:, i] = xup[i] @ v[:, p] + vj_velocity
            # OPTIMIZATION: Use matmul with out
            np.matmul(xup[i], v[:, p], out=scratch_vec)
            scratch_vec += vj_velocity
            v[:, i] = scratch_vec

            # OPTIMIZATION: Use pre-allocated buffer for cross product
            # Use fast version to avoid overhead
            cross_motion_fast(v[:, i], vj_velocity, out=c[:, i])

        # Initialize articulated-body inertia with rigid-body inertia
        # OPTIMIZATION: Copy into pre-allocated buffer instead of creating new array
        # This is safe because ia_articulated[i] is a slice of the contiguous 3D array
        ia_articulated[i] = model["I"][i]

        # Bias force: Coriolis + external forces
        # pa_bias[:, i] = cross_force(v[:, i], model["I"][i] @ v[:, i]) - f_ext[:, i]
        # Optimization: Use temporary buffer
        # i_v = model["I"][i] @ v[:, i]
        np.matmul(model["I"][i], v[:, i], out=i_v_buf)

        # Use fast version to avoid overhead
        cross_force_fast(v[:, i], i_v_buf, out=cross_buf)
        # pa_bias[:, i] = cross_buf - f_ext[:, i]
        np.subtract(cross_buf, f_ext[:, i], out=pa_bias[:, i])

    # --- Pass 2: Backward recursion (articulated-body inertias) ---
    for i in range(nb - 1, -1, -1):
        dof_idx = dof_indices[i]

        if dof_idx != -1:
            # Optimized: u_force is just a column of ia_articulated
            u_force[:, i] = ia_articulated[i][:, dof_idx]
        else:
            np.matmul(ia_articulated[i], s_subspace[i], out=u_force[:, i])

        # d[i] = s_subspace[i] @ u_force[:, i]  # Joint-space inertia
        if dof_idx != -1:
            d[i] = u_force[dof_idx, i]
        else:
            d[i] = np.dot(s_subspace[i], u_force[:, i])

        # u[i] = tau[i] - s_subspace[i] @ pa_bias[:, i]  # Bias torque
        if dof_idx != -1:
            u[i] = tau[i] - pa_bias[dof_idx, i]
        else:
            u[i] = tau[i] - np.dot(s_subspace[i], pa_bias[:, i])

        # Articulated-body inertia update for parent
        if model["parent"][i] != -1:
            p = model["parent"][i]

            # Inverse of joint-space inertia
            # For 1-DOF joints, this is just 1/d(i)
            if abs(d[i]) < TOLERANCE:
                d[i] = np.sign(d[i]) * TOLERANCE if d[i] != 0 else TOLERANCE

            dinv = 1 / d[i]

            # Update articulated inertia
            # ia_update = np.outer(u_force[:, i], u_force[:, i]) * dinv
            # ia_articulated[p] = (
            #     ia_articulated[p] + xup[i].T @ (ia_articulated[i]-ia_update) @ xup[i]
            # )

            # OPTIMIZATION: Minimize allocations in the update
            # 1. ia_prev = ia_articulated[i] - ia_update
            # We can construct (ia_articulated[i] - ia_update) efficiently
            # ia_update is rank-1.
            # Let's compute term1 = (ia_articulated[i] - ia_update) @ xup[i]
            # term1 = ia_articulated[i] @ xup[i] - ia_update @ xup[i]
            # term1 = ia_articulated[i] @ xup[i] - u * (u.T @ xup[i]) * dinv

            # scratch_mat = ia_articulated[i] @ xup[i]
            np.matmul(ia_articulated[i], xup[i], out=scratch_mat)

            # scratch_vec = xup[i].T @ u_force[:, i]  (equiv to u_force.T @ xup[i])
            np.matmul(xup[i].T, u_force[:, i], out=scratch_vec)

            # Subtract rank-1 update from scratch_mat
            # scratch_mat -= np.outer(u_force[:, i] * dinv, scratch_vec)
            # OPTIMIZATION: Avoid temporary allocations
            np.multiply(u_force[:, i], dinv, out=temp_vec)
            # Re-use xj_buf as temporary buffer for outer product
            np.outer(temp_vec, scratch_vec, out=xj_buf)
            scratch_mat -= xj_buf

            # Now add xup[i].T @ scratch_mat to ia_articulated[p]
            # This is ia_articulated[p] += xup[i].T @ scratch_mat
            # Reuse another buffer or just accumulate directly if possible?
            # We can reuse xj_buf as a temp buffer for the result of matmul
            np.matmul(xup[i].T, scratch_mat, out=xj_buf)
            ia_articulated[p] += xj_buf

            # Update bias force
            # pa_update = (
            #     pa_bias[:, i]
            #     + ia_articulated[i] @ c[:, i]
            #     + u_force[:, i] * dinv * u[i]
            # )
            # pa_bias[:, p] = pa_bias[:, p] + xup[i].T @ pa_update

            # OPTIMIZATION:
            # 1. term = ia_articulated[i] @ c[:, i]
            np.matmul(ia_articulated[i], c[:, i], out=scratch_vec)

            # 2. Add other terms
            scratch_vec += pa_bias[:, i]
            # OPTIMIZATION: Avoid allocation for u_force * scalar
            scalar = dinv * u[i]
            np.multiply(u_force[:, i], scalar, out=temp_vec)
            scratch_vec += temp_vec

            # 3. Transform and add to parent
            # pa_bias[:, p] += xup[i].T @ scratch_vec
            # Re-use cross_buf as temp
            np.matmul(xup[i].T, scratch_vec, out=cross_buf)
            pa_bias[:, p] += cross_buf

    # --- Pass 3: Forward recursion (accelerations) ---
    for i in range(nb):
        if model["parent"][i] == -1:
            # For base-connected bodies, apply gravity through Xup transform
            # a[:, i] = xup[i] @ (-a_grav) + c[:, i]
            np.matmul(xup[i], neg_a_grav, out=scratch_vec)
            scratch_vec += c[:, i]
            a[:, i] = scratch_vec
        else:
            p = model["parent"][i]
            # a[:, i] = xup[i] @ a[:, p] + c[:, i]
            np.matmul(xup[i], a[:, p], out=scratch_vec)
            scratch_vec += c[:, i]
            a[:, i] = scratch_vec

        # qdd[i] = (u[i] - u_force[:, i] @ a[:, i]) / d[i]
        qdd[i] = (u[i] - np.dot(u_force[:, i], a[:, i])) / d[i]

        # a[:, i] = a[:, i] + s_subspace[i] * qdd[i]
        dof_idx = dof_indices[i]
        if dof_idx != -1:
            # Optimize: a[dof_idx, i] += qdd[i]
            # But a is (6, nb), so a[:, i] is a 1D slice.
            a[dof_idx, i] += qdd[i]
        else:
            a[:, i] += s_subspace[i] * qdd[i]

    return qdd
