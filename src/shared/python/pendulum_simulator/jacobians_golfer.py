"""
Jacobian and manipulability ellipsoid computations for the golfer model.

DRY: Reuses the shared ellipsoid_from_jacobian kernel from jacobians.py.
Computes task-space Jacobians for all endpoints (both arms + clubhead).
"""

from __future__ import annotations

import logging

import numpy as np

from .jacobians import ellipsoid_from_jacobian
from .physics_golfer import N_DOF, GolferParams, forward_kinematics

logger = logging.getLogger(__name__)


def _numerical_jacobian(
    q: np.ndarray,
    p: GolferParams,
    joint_name: str,
    eps: float = 1e-7,
) -> np.ndarray:
    """Compute 2×8 task-space Jacobian for a named joint via finite differences.

    J maps joint velocities qdot to 2D Cartesian endpoint velocity.

    Parameters
    ----------
    q : np.ndarray, shape (8,)
    p : GolferParams
    joint_name : str — key in forward_kinematics output

    Returns
    -------
    J : np.ndarray, shape (2, 8)
    """
    if not (q.shape == (N_DOF,)):
        raise ValueError(f"q shape must be ({N_DOF},)")
    fk0 = forward_kinematics(q, p)
    pos0 = np.array(fk0[joint_name])

    J = np.zeros((2, N_DOF))
    for j in range(N_DOF):
        q_plus = q.copy()
        q_plus[j] += eps
        fk_j = forward_kinematics(q_plus, p)
        pos_j = np.array(fk_j[joint_name])
        J[:, j] = (pos_j - pos0) / eps

    return J


def jacobian_golfer(q: np.ndarray, p: GolferParams) -> dict[str, np.ndarray]:
    """Compute task-space Jacobians for all key endpoints.

    Returns
    -------
    dict mapping joint names to (2, 8) Jacobian matrices.
    """
    if q is None:
        raise ValueError("q must be provided")
    if q.shape[0] > N_DOF:
        q = q[:N_DOF]

    joints = ["rh", "lh", "club_tip", "re", "le", "hub"]
    return {name: _numerical_jacobian(q, p, name) for name in joints}


def ellipsoids_golfer(q: np.ndarray, p: GolferParams) -> dict[str, dict]:
    """Compute mobility and force ellipsoid data for golfer endpoints.

    Returns
    -------
    dict with joint name keys, each containing:
        'jacobian', 'directions', 'mob_semi_axes',
        'force_semi_axes', 'singular_values'
    """
    if q is None:
        raise ValueError("q must be provided")
    if q.shape[0] > N_DOF:
        q = q[:N_DOF]

    jacs = jacobian_golfer(q, p)
    result: dict[str, dict] = {}

    for name, J in jacs.items():
        dirs, mob, force, svs = ellipsoid_from_jacobian(J)
        result[name] = {
            "jacobian": J,
            "directions": dirs,
            "mob_semi_axes": mob,
            "force_semi_axes": force,
            "singular_values": svs,
        }

    return result


def delta_matrix(q: np.ndarray, p: GolferParams) -> np.ndarray:
    """Compute the Delta matrix (ZTC F matrix) for the golfer.

    Delta = M^{-1} at the current configuration, which maps
    joint torques to joint accelerations when velocity = 0.

    Returns
    -------
    Delta : np.ndarray, shape (8, 8)
    """
    if q is None:
        raise ValueError("q must be provided")
    if q.shape[0] > N_DOF:
        q = q[:N_DOF]

    from .physics_golfer import mass_matrix

    M = mass_matrix(q, p)
    # The golfer mass matrix is rank-deficient (rank 6 of 8) because the
    # closed kinematic loop imposes 4 holonomic constraints on 8 DOFs.
    # Use Moore-Penrose pseudoinverse so delta lives in the feasible subspace.
    return np.linalg.pinv(M)


def ztcf_matrix(
    q: np.ndarray, p: GolferParams, joint_name: str = "club_tip"
) -> np.ndarray:
    """Compute the Zero-Torque Constraint Force transfer matrix.

    Maps applied joint torques to endpoint forces via:
        F_endpoint = (J M^{+} J^T)^{-1} J M^{+} tau

    The ZTCF matrix is: T = (J M^{+} J^T)^{-1} J M^{+}

    Uses the pseudoinverse M^{+} because the golfer mass matrix is
    rank-deficient due to the closed kinematic loop.

    Returns
    -------
    T : np.ndarray, shape (2, 8) or None if singular
    """
    if q is None:
        raise ValueError("q must be provided")
    if q.shape[0] > N_DOF:
        q = q[:N_DOF]

    from .physics_golfer import mass_matrix

    J = _numerical_jacobian(q, p, joint_name)
    M = mass_matrix(q, p)
    M_inv = np.linalg.pinv(M)

    JMinv = J @ M_inv
    A = JMinv @ J.T

    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        logger.warning("ZTCF matrix A singular, cannot compute transfer matrix")
        return None  # type: ignore[return-value]

    result: np.ndarray = A_inv @ JMinv
    return result
