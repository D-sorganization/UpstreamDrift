"""
Gravity vector interpolation for non-linear paths in configuration space.

Handles proper interpolation of gravity direction and magnitude when moving
between poses with different gravity vectors. This is critical for motion
matching across different environments where gravity direction changes
(e.g., motion on rotating platforms, underwater, or exotic environments).

Issue #4106: Gravity vector interpolation fails for non-linear parametric
paths through config space.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def interpolate_gravity_vector(
    gravity_start: npt.NDArray[np.float64],
    gravity_end: npt.NDArray[np.float64],
    alpha: float,
) -> npt.NDArray[np.float64]:
    """
    Interpolate between two gravity vectors using SLERP on vector representations.

    Handles non-linear paths through gravity direction space by treating gravity
    as a 3D vector and interpolating the direction via spherical linear
    interpolation (SLERP), while linearly interpolating the magnitude.

    This ensures gravity direction changes smoothly along the path while
    preserving the magnitude interpolation. The result is normalized in
    direction but magnitude-aware.

    Args:
        gravity_start: Start gravity vector [gx, gy, gz] (not necessarily
            normalized). Typical value: [0, 0, -9.81] for Earth gravity.
        gravity_end: End gravity vector [gx, gy, gz] (not necessarily
            normalized). Must be non-zero.
        alpha: Interpolation parameter [0, 1].
            alpha=0 returns gravity_start (in normalized magnitude form).
            alpha=1 returns gravity_end (in normalized magnitude form).

    Returns:
        Interpolated gravity vector with interpolated magnitude.
        Shape: (3,), dtype: float64

    Raises:
        ValueError: If gravity vectors are zero or nearly zero (< 1e-10 in magnitude)
        ValueError: If input shapes are incorrect

    Examples:
        >>> import numpy as np
        >>> g_down = np.array([0.0, 0.0, -9.81])
        >>> g_right = np.array([9.81, 0.0, 0.0])
        >>> g_mid = interpolate_gravity_vector(g_down, g_right, 0.5)
        >>> np.allclose(np.linalg.norm(g_mid), 9.81)  # magnitude preserved
        True
    """
    if not (gravity_start is not None):
        raise ValueError("gravity_start must be provided")
    if not (gravity_end is not None):
        raise ValueError("gravity_end must be provided")

    gravity_start = np.asarray(gravity_start, dtype=np.float64)
    gravity_end = np.asarray(gravity_end, dtype=np.float64)

    # Validate input dimensions
    if gravity_start.shape != (3,):
        raise ValueError(
            f"gravity_start must be shape (3,), got {gravity_start.shape}"
        )
    if gravity_end.shape != (3,):
        raise ValueError(f"gravity_end must be shape (3,), got {gravity_end.shape}")

    # Compute magnitudes and validate
    mag_start = np.linalg.norm(gravity_start)
    mag_end = np.linalg.norm(gravity_end)

    if mag_start < 1e-10:
        raise ValueError(f"gravity_start is near-zero: magnitude {mag_start}")
    if mag_end < 1e-10:
        raise ValueError(f"gravity_end is near-zero: magnitude {mag_end}")

    # Normalize directions
    dir_start = gravity_start / mag_start
    dir_end = gravity_end / mag_end

    # Use SLERP on pure quaternion representations of direction vectors.
    # We represent directions as quaternions [0, x, y, z] and normalize.
    q_start = np.concatenate([[0.0], dir_start])  # [0, x, y, z]
    q_end = np.concatenate([[0.0], dir_end])  # [0, x, y, z]

    # Normalize these "quaternions" to unit magnitude
    q_start = q_start / np.linalg.norm(q_start)
    q_end = q_end / np.linalg.norm(q_end)

    # Apply SLERP to interpolate direction (see slerp function in pose6dof.py)
    dot = np.clip(np.dot(q_start, q_end), -1.0, 1.0)

    # Handle antipodal directions
    if dot < 0:
        q_end = -q_end
        dot = -dot

    # Linear interpolation for very close directions
    if dot > 0.9995:
        q_interp = q_start + alpha * (q_end - q_start)
        q_interp = q_interp / np.linalg.norm(q_interp)
    else:
        # Standard SLERP formula
        theta_0 = np.arccos(dot)
        theta = theta_0 * alpha
        sin_theta = np.sin(theta)
        sin_theta_0 = np.sin(theta_0)

        s1 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s2 = sin_theta / sin_theta_0

        q_interp = s1 * q_start + s2 * q_end

    # Extract direction from interpolated quaternion (skip w component)
    dir_interp = q_interp[1:] / np.linalg.norm(q_interp[1:])

    # Interpolate magnitudes linearly
    mag_interp = (1.0 - alpha) * mag_start + alpha * mag_end

    # Return interpolated gravity vector (direction * magnitude)
    result: npt.NDArray[np.float64] = dir_interp * mag_interp
    return result
