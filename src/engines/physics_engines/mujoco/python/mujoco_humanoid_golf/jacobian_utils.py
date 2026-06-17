"""Jacobian computation and mass-matrix utilities for golf swing analysis.

This module provides standalone functions for:
- Jacobian computation with buffer management
- Jacobian rank checking
- Mass matrix and Coriolis matrix computation
- Effective mass helpers (direction validation, conditioning checks)

These are extracted from KinematicForceAnalyzer to keep that class focused
on orchestration logic.
"""

from __future__ import annotations

import warnings

import mujoco
import numpy as np

# Import numerical constants (Assessment B-004, B-007)
from src.shared.python.core.numerical_constants import (
    EPSILON_FINITE_DIFF_JACOBIAN,
    EPSILON_SINGULARITY_DETECTION,
)

from ._effective_mass_kernel import compute_effective_mass_from_solve


def compute_jacobian(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
    jacp: np.ndarray,
    jacr: np.ndarray,
    use_reshaped_arrays: bool,
    nv: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Jacobian for a body using pre-allocated buffers.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        body_id: Body ID
        jacp: Pre-allocated translational Jacobian buffer
        jacr: Pre-allocated rotational Jacobian buffer
        use_reshaped_arrays: Whether to use (3, nv) shaped arrays
        nv: Number of velocity DOFs

    Returns:
        Tuple of (jacp, jacr) as (3, nv) arrays.
        Note: Returns views into the provided buffers or copies depending on usage.
    """
    if body_id is None:
        raise ValueError("body_id must be provided")

    if use_reshaped_arrays:
        mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
        return jacp, jacr
    else:
        mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
        return (
            jacp.reshape(3, nv),
            jacr.reshape(3, nv),
        )


def check_jacobian_rank(jacp: np.ndarray) -> None:
    """Warn if the translational Jacobian is rank-deficient.

    Args:
        jacp: Translational Jacobian (3 x nv)
    """
    J_rank = np.linalg.matrix_rank(jacp)
    if J_rank < 3:
        warnings.warn(
            f"Jacobian is rank deficient: rank={J_rank} < 3. "
            "Robot has lost mobility in some directions. "
            "Effective mass may not be well-defined.",
            category=RuntimeWarning,
            stacklevel=2,
        )


def compute_mass_matrix(
    model: mujoco.MjModel,
    perturb_data: mujoco.MjData,
    qpos: np.ndarray,
) -> np.ndarray:
    """Compute configuration-dependent mass matrix M(q).

    Uses a dedicated perturb_data to prevent state corruption.

    Args:
        model: MuJoCo model
        perturb_data: Scratch MjData (will be modified)
        qpos: Joint positions [nv]

    Returns:
        Mass matrix [nv x nv]
    """
    if qpos is None:
        raise ValueError("qpos must be provided")
    perturb_data.qpos[:] = qpos
    mujoco.mj_forward(model, perturb_data)

    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M, perturb_data.qM)

    return M


def compute_coriolis_matrix(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    qvel: np.ndarray,
    compute_coriolis_fn,
) -> np.ndarray:
    """Compute Coriolis matrix C(q,q̇) via finite differences.

    The Coriolis matrix satisfies: C(q,q̇)q̇ = coriolis forces

    Args:
        model: MuJoCo model (used for nv)
        qpos: Joint positions [nv]
        qvel: Joint velocities [nv]
        compute_coriolis_fn: Callable(qpos, qvel) -> np.ndarray[nv]
            (e.g. KinematicForceAnalyzer.compute_coriolis_forces)

    Returns:
        Coriolis matrix [nv x nv]
    """
    if qpos is None:
        raise ValueError("qpos must be provided")
    epsilon = EPSILON_FINITE_DIFF_JACOBIAN
    C = np.zeros((model.nv, model.nv))

    c_ref = compute_coriolis_fn(qpos, qvel)

    for i in range(model.nv):
        qvel_perturb = qvel.copy()
        qvel_perturb[i] += epsilon

        c_perturb = compute_coriolis_fn(qpos, qvel_perturb)
        C[:, i] = (c_perturb - c_ref) / epsilon

    return C


def validate_effective_mass_direction(direction: np.ndarray) -> np.ndarray:
    """Normalize and validate an effective-mass direction vector.

    Args:
        direction: Direction vector [3]

    Returns:
        Normalized direction vector [3]

    Raises:
        ValueError: If direction has near-zero magnitude
    """
    direction_norm = np.linalg.norm(direction)
    if direction_norm < EPSILON_SINGULARITY_DETECTION:
        raise ValueError(
            f"Direction vector has near-zero magnitude: {direction_norm:.2e}. "
            "Cannot compute effective mass for zero-length direction."
        )
    return direction / direction_norm


def check_mass_matrix_conditioning(M: np.ndarray) -> None:
    """Warn if mass matrix is ill-conditioned or not positive definite.

    Args:
        M: Mass matrix [nv x nv]

    Raises:
        ValueError: If mass matrix is not positive definite
    """
    M_cond = np.linalg.cond(M)
    if M_cond > 1e6:
        warnings.warn(
            f"Mass matrix is ill-conditioned: κ(M) = {M_cond:.2e} > 1e6. "
            "Effective mass computation may be numerically unstable. "
            "This often indicates the robot is near a kinematic singularity.",
            category=UserWarning,
            stacklevel=2,
        )

    eigenvalues = np.linalg.eigvalsh(M)
    if np.any(eigenvalues <= 0):
        raise ValueError(
            f"Mass matrix is not positive definite. "
            f"Minimum eigenvalue: {eigenvalues.min():.2e}. "
            "This indicates a modeling error or numerical instability."
        )


def compute_effective_mass_value(
    direction: np.ndarray, jacp: np.ndarray, M: np.ndarray
) -> float:
    """Compute scalar effective mass from direction, Jacobian, and mass matrix.

    Args:
        direction: Normalized direction vector [3]
        jacp: Translational Jacobian (3 x nv)
        M: Mass matrix (nv x nv)

    Returns:
        Effective mass [kg]

    Warns:
        UserWarning: If denominator is near zero (kinematic singularity)
        UserWarning: If result is non-finite

    Raises:
        ValueError: If computed effective mass is negative
    """
    return compute_effective_mass_from_solve(direction, jacp, M)
