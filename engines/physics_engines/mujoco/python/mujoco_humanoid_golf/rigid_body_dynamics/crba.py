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
    xup = [None] * nb
    s_subspace = [None] * nb
    ic_composite = [None] * nb

    h_matrix = np.zeros((nb, nb))

    # --- Forward pass: compute transforms and motion subspaces ---
    for i in range(nb):
        xj_transform, s_subspace[i] = jcalc(model["jtype"][i], q[i])
        xup[i] = xj_transform @ model["Xtree"][i]

    # --- Backward pass: compute composite inertias ---
    # Initialize composite inertias with body inertias
    for i in range(nb):
        ic_composite[i] = model["I"][i].copy()

    # Accumulate inertias from children to parents
    for i in range(nb - 1, -1, -1):
        if model["parent"][i] != -1:
            p = model["parent"][i]
            # Transform composite inertia to parent frame and add
            ic_composite[p] = ic_composite[p] + xup[i].T @ ic_composite[i] @ xup[i]

    # --- Compute mass matrix ---
    # H(i,j) represents the coupling between joints i and j
    for i in range(nb):
        # f_force is the force transmitted through joint i due to unit acceleration
        # at joint i, affecting the composite body rooted at i
        f_force = ic_composite[i] @ s_subspace[i]
        h_matrix[i, i] = s_subspace[i] @ f_force  # Diagonal element

        # Propagate force up the tree to compute off-diagonal elements
        j = i
        while model["parent"][j] != -1:
            p = model["parent"][j]
            f_force = xup[j].T @ f_force  # Transform force to parent frame
            h_matrix[i, p] = s_subspace[p] @ f_force  # Off-diagonal element
            h_matrix[p, i] = h_matrix[i, p]  # Symmetric
            j = p

    # Ensure exact symmetry (numerical precision) - optional but safe
    # Also cleans up any tiny asymmetries
    # OPTIMIZATION: Using h_matrix directly as we fill symmetric elements manually
    return h_matrix
