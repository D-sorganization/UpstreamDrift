"""
Composite Rigid Body Algorithm (CRBA) for computing mass matrix.
"""

import numpy as np
from mujoco_humanoid_golf.spatial_algebra import jcalc


def crba(model: dict, q: np.ndarray) -> np.ndarray:
    """
    Composite Rigid Body Algorithm for computing mass matrix.

    Computes the joint-space mass matrix (inertia matrix) H(q) of a kinematic tree.
    The mass matrix H satisfies the equation of motion:
        H(q) * qdd + C(q, qd) * qd + g(q) = tau

    Args:
        model: Robot model dictionary with fields:
            NB: Number of bodies (int)
            parent: Parent body indices (array of length NB)
            jtype: Joint types (list of strings, length NB)
            Xtree: Joint transforms (NB-length list of 6x6 arrays)
            I: Spatial inertias (NB-length list of 6x6 arrays)
        q: Joint positions (NB,)

    Returns:
        Symmetric positive-definite mass matrix H (NB, NB)

    Algorithm:
        1. Compute composite inertias (backward pass)
        2. Compute mass matrix elements using motion subspaces

    References:
        Featherstone, R. (2008). Rigid Body Dynamics Algorithms.
        Chapter 6: Operational Space Dynamics, Algorithm 6.1

    Example:
        >>> model = create_robot_model()
        >>> q = np.array([0.5, -0.3])
        >>> h_matrix = crba(model, q)
        >>> h_matrix.shape
        (2, 2)
        >>> np.allclose(h_matrix, h_matrix.T)  # Check symmetry
        True
    """
    # Use ravel() to avoid copying data when possible
    q = np.asarray(q).ravel()

    nb = model["NB"]
    if len(q) != nb:
        msg = f"q must have length {nb}, got {len(q)}"
        raise ValueError(msg)

    # Initialize lists (no need to pre-allocate numpy arrays since we replace them)
    # OPTIMIZATION: Avoid allocating initial zero arrays that are immediately discarded
    # We use lists and append to avoid Optional[np.ndarray] types for strict typing
    xup = np.empty((nb, 6, 6))
    s_subspace: list[np.ndarray] = []
    dof_indices: list[int] = []  # Cache active DOF indices
    # ic_composite will be initialized later

    h_matrix = np.zeros((nb, nb))

    # Pre-allocate temporary buffers to avoid allocation in loop
    xj_buf = np.empty((6, 6))
    tmp_6x6 = np.empty((6, 6))
    f_force = np.empty(6)
    scratch_vec = np.empty(6)

    # --- Forward pass: compute transforms and motion subspaces ---
    for i in range(nb):
        xj_transform, s_vec, dof_idx = jcalc(model["jtype"][i], q[i], out=xj_buf)
        s_subspace.append(s_vec)
        dof_indices.append(dof_idx)
        # Optimized xup[i] = xj_transform @ model["Xtree"][i]
        np.matmul(xj_transform, model["Xtree"][i], out=xup[i])

    # --- Backward pass: compute composite inertias ---
    # Initialize composite inertias with body inertias
    ic_composite: list[np.ndarray] = [i_mat.copy() for i_mat in model["I"]]

    # Accumulate inertias from children to parents
    for i in range(nb - 1, -1, -1):
        if model["parent"][i] != -1:
            p = model["parent"][i]
            # Transform composite inertia to parent frame and add
            # ic_composite[p] += xup[i].T @ ic_composite[i] @ xup[i]

            # OPTIMIZATION: Break down to minimize allocation using dot
            # 1. tmp_6x6 = ic_composite[i] @ xup[i]
            np.dot(ic_composite[i], xup[i], out=tmp_6x6)

            # 2. Add xup[i].T @ tmp_6x6 to ic_composite[p]
            # Reuse xj_buf as scratch space
            np.dot(xup[i].T, tmp_6x6, out=xj_buf)

            ic_composite[p] += xj_buf

    # --- Compute mass matrix ---
    # H(i,j) represents the coupling between joints i and j
    for i in range(nb):
        # f_force is the force transmitted through joint i due to unit acceleration
        # at joint i, affecting the composite body rooted at i

        idx = dof_indices[i]
        if idx != -1:
            # OPTIMIZATION: For standard joints, s_subspace is sparse (one 1.0)
            # f_force = ic_composite[i] @ s_subspace[i] becomes just a column copy
            # This saves 36 muls and 30 adds
            f_force[:] = ic_composite[i][:, idx]
        else:
            # Fallback for generic joints
            np.dot(ic_composite[i], s_subspace[i], out=f_force)

        # h_matrix[i, i] = s_subspace[i] @ f_force  # Diagonal element
        if idx != -1:
            # OPTIMIZATION: sparse dot product
            h_matrix[i, i] = f_force[idx]
        else:
            h_matrix[i, i] = np.dot(s_subspace[i], f_force)

        # Propagate force up the tree to compute off-diagonal elements
        j = i
        while model["parent"][j] != -1:
            p = model["parent"][j]

            # f_force = xup[j].T @ f_force  # Transform force to parent frame
            # OPTIMIZATION: Use scratch buffer
            np.dot(xup[j].T, f_force, out=scratch_vec)
            f_force[:] = scratch_vec

            # h_matrix[i, p] = s_subspace[p] @ f_force  # Off-diagonal element
            # This is the hottest part of CRBA (O(n^2))
            idx_p = dof_indices[p]
            if idx_p != -1:
                # OPTIMIZATION: sparse dot product
                val = f_force[idx_p]
            else:
                val = np.dot(s_subspace[p], f_force)

            h_matrix[i, p] = val
            h_matrix[p, i] = val  # Symmetric
            j = p

    # Ensure exact symmetry (numerical precision) - optional but safe
    # Also cleans up any tiny asymmetries
    # OPTIMIZATION: Using h_matrix directly as we fill symmetric elements manually
    return h_matrix
