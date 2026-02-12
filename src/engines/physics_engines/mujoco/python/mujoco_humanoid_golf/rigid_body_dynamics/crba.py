"""
Composite Rigid Body Algorithm (CRBA) for computing mass matrix.
"""

import numpy as np
from mujoco_humanoid_golf.spatial_algebra import jcalc


def _crba_forward_pass(
    nb: int,
    model_jtype: list,
    model_xtree: list,
    q: np.ndarray,
    xup: np.ndarray,
    xup_T: np.ndarray,
    xj_buf: np.ndarray,
) -> tuple[list[np.ndarray], list[int]]:
    """Forward pass: compute joint transforms and motion subspaces.

    For each body, computes the joint-to-parent transform (xup) and
    caches the motion subspace vector and DOF index from jcalc.

    Args:
        nb: Number of bodies.
        model_jtype: Joint types list.
        model_xtree: Joint tree transforms list.
        q: Joint positions.
        xup: Output array for joint-to-parent transforms (nb, 6, 6).
        xup_T: Output array for pre-computed contiguous transposes (nb, 6, 6).
        xj_buf: Scratch buffer for jcalc output (6, 6).

    Returns:
        Tuple of (s_subspace, dof_indices) where s_subspace is a list of
        motion subspace vectors and dof_indices caches active DOF indices.
    """
    s_subspace: list[np.ndarray] = []
    dof_indices: list[int] = []

    for i in range(nb):
        xj_transform, s_vec, dof_idx = jcalc(model_jtype[i], q[i], out=xj_buf)
        s_subspace.append(s_vec)
        dof_indices.append(dof_idx)
        # Optimized xup[i] = xj_transform @ model_xtree[i]
        np.matmul(xj_transform, model_xtree[i], out=xup[i])
        # xup[i] is C-contiguous, .T is strided. Assigning to xup_T[i] makes it
        # contiguous for fast usage in dot products
        xup_T[i][:] = xup[i].T

    return s_subspace, dof_indices


def _crba_backward_pass(
    nb: int,
    model_parent: np.ndarray,
    model_inertia: list,
    xup: np.ndarray,
    xup_T: np.ndarray,
    tmp_6x6: np.ndarray,
    xj_buf: np.ndarray,
) -> np.ndarray:
    """Backward pass: compute composite inertias by accumulating from leaves to root.

    Initializes composite inertias with body inertias, then propagates each
    body's composite inertia to its parent frame and accumulates.

    Args:
        nb: Number of bodies.
        model_parent: Parent body indices array.
        model_inertia: Body spatial inertias list.
        xup: Joint-to-parent transforms (nb, 6, 6).
        xup_T: Pre-computed contiguous transposes (nb, 6, 6).
        tmp_6x6: Scratch buffer (6, 6).
        xj_buf: Scratch buffer (6, 6).

    Returns:
        Composite inertia array (nb, 6, 6).
    """
    # OPTIMIZATION: Bulk copy to 3D array is faster than list comprehension with copy()
    ic_composite = np.array(model_inertia, dtype=float)

    for i in range(nb - 1, -1, -1):
        if model_parent[i] != -1:
            p = model_parent[i]
            # ic_composite[p] += xup[i].T @ ic_composite[i] @ xup[i]
            # OPTIMIZATION: Break down to minimize allocation using dot
            np.dot(ic_composite[i], xup[i], out=tmp_6x6)
            # OPTIMIZATION: Use pre-computed contiguous transpose xup_T[i]
            np.dot(xup_T[i], tmp_6x6, out=xj_buf)
            ic_composite[p] += xj_buf

    return ic_composite


def _crba_mass_matrix(
    nb: int,
    model_parent: np.ndarray,
    ic_composite: np.ndarray,
    s_subspace: list[np.ndarray],
    dof_indices: list[int],
    xup_T: np.ndarray,
    h_matrix: np.ndarray,
    f_force: np.ndarray,
    scratch_vec: np.ndarray,
) -> np.ndarray:
    """Compute mass matrix elements using composite inertias and motion subspaces.

    Fills the diagonal elements from each body's composite inertia, then
    propagates forces up the kinematic tree to compute off-diagonal (coupling)
    elements. The result is symmetric.

    Args:
        nb: Number of bodies.
        model_parent: Parent body indices array.
        ic_composite: Composite inertia array (nb, 6, 6).
        s_subspace: Motion subspace vectors list.
        dof_indices: Cached active DOF indices list.
        xup_T: Pre-computed contiguous transposes (nb, 6, 6).
        h_matrix: Output mass matrix (nb, nb), should be zero-initialized.
        f_force: Scratch buffer for force vector (6,).
        scratch_vec: Scratch buffer for vector operations (6,).

    Returns:
        Symmetric positive-definite mass matrix H (nb, nb).
    """
    for i in range(nb):
        idx = dof_indices[i]
        if idx != -1:
            # OPTIMIZATION: For standard joints, s_subspace is sparse (one 1.0)
            # f_force = ic_composite[i] @ s_subspace[i] becomes just a column copy
            f_force[:] = ic_composite[i][:, idx]
        else:
            np.dot(ic_composite[i], s_subspace[i], out=f_force)

        # Diagonal element
        if idx != -1:
            h_matrix[i, i] = f_force[idx]
        else:
            h_matrix[i, i] = np.dot(s_subspace[i], f_force)

        # Propagate force up the tree to compute off-diagonal elements
        j = i
        while model_parent[j] != -1:
            p = model_parent[j]

            # Transform force to parent frame
            # OPTIMIZATION: Use scratch buffer and pre-computed contiguous transpose
            np.dot(xup_T[j], f_force, out=scratch_vec)
            # Swap references to avoid copy (speedup ~15%)
            f_force, scratch_vec = scratch_vec, f_force

            # Off-diagonal element (hottest part of CRBA, O(n^2))
            idx_p = dof_indices[p]
            if idx_p != -1:
                val = f_force[idx_p]
            else:
                val = np.dot(s_subspace[p], f_force)

            h_matrix[i, p] = val
            h_matrix[p, i] = val  # Symmetric
            j = p

    return h_matrix


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
    q = np.asarray(q).ravel()

    nb = model["NB"]
    if len(q) != nb:
        msg = f"q must have length {nb}, got {len(q)}"
        raise ValueError(msg)

    # OPTIMIZATION: Cache dictionary lookups to local variables
    model_parent = model["parent"]
    model_jtype = model["jtype"]
    model_xtree = model["Xtree"]
    model_inertia = model["I"]

    # Pre-allocate arrays and scratch buffers
    xup = np.empty((nb, 6, 6))
    xup_T = np.empty((nb, 6, 6))
    h_matrix = np.zeros((nb, nb))
    xj_buf = np.empty((6, 6))
    tmp_6x6 = np.empty((6, 6))
    f_force = np.empty(6)
    scratch_vec = np.empty(6)

    # Phase 1: Forward pass - compute transforms and motion subspaces
    s_subspace, dof_indices = _crba_forward_pass(
        nb, model_jtype, model_xtree, q, xup, xup_T, xj_buf
    )

    # Phase 2: Backward pass - compute composite inertias
    ic_composite = _crba_backward_pass(
        nb, model_parent, model_inertia, xup, xup_T, tmp_6x6, xj_buf
    )

    # Phase 3: Compute mass matrix from composite inertias
    return _crba_mass_matrix(
        nb,
        model_parent,
        ic_composite,
        s_subspace,
        dof_indices,
        xup_T,
        h_matrix,
        f_force,
        scratch_vec,
    )
