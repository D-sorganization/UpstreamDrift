"""Manipulability and Jacobian conditioning diagnostics.

Assessment A Finding F-004 / Guideline C2 Implementation

This module provides real-time Jacobian condition number monitoring to detect
near-singularities and prevent silent failures at gimbal lock or kinematic singular
configurations.

Per Guideline C2:
- κ > 1e6: Warning threshold (near-singularity)
- κ > 1e10: Automatic fallback to pseudoinverse (severe ill-conditioning)
- κ > 1e12: Catastrophic error (cannot proceed)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from shared.python.interfaces import PhysicsEngine

logger = logging.getLogger(__name__)

# Thresholds from Guideline C2
SINGULARITY_WARNING_THRESHOLD = 1e6  # [dimensionless]
# Source: Guideline C2, empirically validated threshold for gimbal lock proximity

SINGULARITY_FALLBACK_THRESHOLD = 1e10  # [dimensionless]
# Source: Guideline C2, switch to damped least squares / pseudoinverse

CATASTROPHIC_SINGULARITY_THRESHOLD = 1e12  # [dimensionless]
# Source: Guideline C2, unrecoverable singularity


class SingularityError(Exception):
    """Raised when Jacobian is catastrophically ill-conditioned (κ > 1e12)."""

    pass


def check_jacobian_conditioning(
    J: np.ndarray, body_name: str, warn: bool = True
) -> float:
    """Compute condition number and warn if near singular.

    Guideline C2: κ > 1e6 triggers warning, κ > 1e10 triggers pseudoinverse.

    Args:
        J: Jacobian matrix (m×n), typically (6×n) for spatial Jacobian
        body_name: Name of body for logging context (e.g., "clubhead", "grip")
        warn: Whether to emit warnings (default: True)

    Returns:
        Condition number κ = σ_max / σ_min

    Raises:
        SingularityError: If κ > 1e12 (catastrophic)

    Example:
        >>> J = engine.compute_jacobian("clubhead")["spatial"]
        >>> kappa = check_jacobian_conditioning(J, "clubhead")
        >>> if kappa > 1e6:
        ...     logger.warning(f"Near-singularity: κ = {kappa:.2e}")
    """
    if J.size == 0 or J.shape[0] == 0 or J.shape[1] == 0:
        logger.warning(f"{body_name}: Empty Jacobian provided")
        return float(np.inf)

    # Compute condition number
    # κ = ||J|| · ||J^+|| = σ_max / σ_min
    kappa = np.linalg.cond(J)

    # Catastrophic singularity (cannot proceed)
    if kappa > CATASTROPHIC_SINGULARITY_THRESHOLD:
        raise SingularityError(
            f"❌ CATASTROPHIC SINGULARITY at {body_name}:\\n"
            f"  Condition number: κ = {kappa:.2e}\\n"
            f"  Threshold: {CATASTROPHIC_SINGULARITY_THRESHOLD:.2e}\\n"
            f"  This configuration is kinematically degenerate.\\n"
            f"  Possible causes:\\n"
            f"    - Fully extended limb (elbow/knee locked)\\n"
            f"    - Gimbal lock (Euler angle singularity)\\n"
            f"    - Closed-chain constraint singular configuration\\n"
            f"  ACTION: Simulation cannot continue. Abort."
        )

    # Severe ill-conditioning (use pseudoinverse)
    elif kappa > SINGULARITY_FALLBACK_THRESHOLD and warn:
        sigma = np.linalg.svd(J, compute_uv=False)
        sigma_min = sigma.min()
        sigma_max = sigma.max()

        logger.error(
            f"⚠️ SEVERE ILL-CONDITIONING at {body_name}:\\n"
            f"  Condition number: κ = {kappa:.2e}\\n"
            f"  Smallest singular value: σ_min = {sigma_min:.2e}\\n"
            f"  Largest singular value: σ_max = {sigma_max:.2e}\\n"
            f"  Threshold: {SINGULARITY_FALLBACK_THRESHOLD:.2e}\\n"
            f"  ACTION: Switching to pseudoinverse (damped least squares)\\n"
            f"  Possible causes:\\n"
            f"    - Near-extended configuration (θ ≈ 0 or θ ≈ π)\\n"
            f"    - Workspace boundary approach\\n"
            f"    - Multiple joints aligned"
        )

    # Near-singularity (warning only)
    elif kappa > SINGULARITY_WARNING_THRESHOLD and warn:
        logger.warning(
            f"⚠️ Near-singularity at {body_name}:\\n"
            f"  Condition number: κ = {kappa:.2e} (threshold: {SINGULARITY_WARNING_THRESHOLD:.2e})\\n"
            f"  This configuration has reduced manipulability.\\n"
            f"  Possible causes: Extended limb, gimbal lock, singular constraint config\\n"
            f"  Recommendation: Avoid inverse kinematics / force control near this config"
        )

    return float(kappa)


def get_jacobian_conditioning(
    engine: PhysicsEngine, body_name: str, warn: bool = True
) -> float:
    """Convenience function to get conditioning directly from engine.

    Args:
        engine: Physics engine implementing PhysicsEngine protocol
        body_name: Name of body (e.g., "clubhead", "right_hand")
        warn: Whether to emit warnings if κ > threshold

    Returns:
        Condition number κ

    Example:
        >>> from shared.python.manipulability import get_jacobian_conditioning
        >>> kappa = get_jacobian_conditioning(mujoco_engine, "clubhead")
        >>> print(f"Clubhead Jacobian condition: {kappa:.2e}")
    """
    jac_dict = engine.compute_jacobian(body_name)

    if jac_dict is None:
        logger.warning(f"Body '{body_name}' not found in model")
        return float(np.inf)

    # Prefer spatial Jacobian (6×n) if available
    if "spatial" in jac_dict:
        J = jac_dict["spatial"]
    elif "linear" in jac_dict:
        # Use linear component if spatial not available
        J = jac_dict["linear"]
    else:
        logger.warning(f"No Jacobian data for '{body_name}'")
        return float(np.inf)

    return check_jacobian_conditioning(J, body_name, warn=warn)


def compute_manipulability_ellipsoid(J: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute manipulability ellipsoid principal axes and radii.

    The manipulability ellipsoid visualizes the directional capabilities
    of a manipulator at a given configuration.

    Velocity manipulability: J · q̇ = ẋ
    Force manipulability: J^T · f = τ

    Args:
        J: Jacobian matrix (m×n), typically spatial (6×n)

    Returns:
        Tuple of (radii, axes):
            radii: Principal radii (singular values σ_i) (m,)
            axes: Principal axes (right singular vectors V) (n×n)

    Example:
        >>> J = engine.compute_jacobian("clubhead")["spatial"]
        >>> radii, axes = compute_manipulability_ellipsoid(J)
        >>> print(f"Max manipulability: {radii.max():.3f}")
        >>> print(f"Min manipulability: {radii.min():.3f}")
        >>> print(f"Condition number: {radii.max() / radii.min():.2e}")
    """
    # SVD: J = U Σ V^T
    # Σ contains singular values (ellipsoid radii)
    # V contains principal axes
    U, sigma, Vt = np.linalg.svd(J, full_matrices=False)

    return sigma, Vt.T


def compute_manipulability_index(J: np.ndarray) -> float:
    """Compute Yoshikawa manipulability index.

    The manipulability index μ measures the "distance" from singularity:
        μ = √det(J · J^T) = ∏ σ_i

    Interpretation:
        μ = 0: Singular configuration (lost DOF)
        μ > 0: Non-singular (larger is better)

    Args:
        J: Jacobian matrix (m×n)

    Returns:
        Manipulability index μ [dimensionless]

    Reference:
        Yoshikawa, T. (1985). "Manipulability of Robotic Mechanisms"
        The International Journal of Robotics Research, 4(2), 3-9.

    Example:
        >>> J = engine.compute_jacobian("clubhead")["spatial"]
        >>> mu = compute_manipulability_index(J)
        >>> print(f"Manipulability: {mu:.3e}")
    """
    # Manipulability index = product of singular values
    # Equivalent to sqrt(det(J @ J.T))
    sigma = np.linalg.svd(J, compute_uv=False)
    return float(np.prod(sigma))
