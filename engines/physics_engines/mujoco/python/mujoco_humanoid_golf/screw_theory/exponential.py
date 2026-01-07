"""
Exponential and logarithmic maps between se(3) and SE(3).
"""

import numpy as np
from mujoco_humanoid_golf.spatial_algebra.spatial_vectors import skew


def exponential_map(S: np.ndarray, theta: float) -> np.ndarray:
    """
    Screw motion via matrix exponential.

    Computes the rigid body transformation resulting from moving along
    screw axis S by amount theta.

    This implements the exponential map from se(3) to SE(3):
        T = exp([S]*theta)

    For pure rotation (||omega|| = 1, pitch = 0):
        Uses Rodrigues' formula

    For pure translation (omega = 0):
        T = [I, v*theta; 0, 1]

    Args:
        S: 6x1 screw axis [omega; v]
        theta: Scalar displacement along screw (radians or meters)

    Returns:
        4x4 homogeneous transformation matrix in SE(3)

    References:
        Murray, Li, Sastry (1994). A Mathematical Introduction to Robotic Manipulation.
        Lynch, Park (2017). Modern Robotics: Mechanics, Planning, and Control.
        Chapter 3: Rigid-Body Motions

    Examples:
        >>> # Rotation of 90 degrees about z-axis
        >>> from .screws import screw_axis
        >>> S = screw_axis(np.array([0, 0, 1]), np.array([0, 0, 0]))
        >>> T = exponential_map(S, np.pi/2)

        >>> # Translation of 0.5m along x-axis
        >>> S = screw_axis(np.array([1, 0, 0]), np.array([0, 0, 0]), np.inf)
        >>> T = exponential_map(S, 0.5)
    """
    s_screw = np.asarray(S).flatten()
    assert s_screw.shape == (6,), f"S must be 6x1, got shape {s_screw.shape}"

    omega = s_screw[:3]
    v = s_screw[3:]

    omega_norm = np.linalg.norm(omega)

    if omega_norm < np.finfo(float).eps:
        # Pure translation (prismatic motion)
        r_rot = np.eye(3)
        p = v * theta
    else:
        # Rotation or screw motion
        omega_hat = skew(omega)

        # Rodrigues' formula for rotation matrix
        # R = I + sin(theta)*omega_hat + (1-cos(theta))*omega_hat^2
        r_rot = (
            np.eye(3)
            + np.sin(theta) * omega_hat
            + (1 - np.cos(theta)) * (omega_hat @ omega_hat)
        )

        # Position component (Proposition 3.14 in Lynch & Park)
        # p = (I*theta + (1-cos(theta))*omega_hat + (theta-sin(theta))*omega_hat^2) * v
        p = (
            np.eye(3) * theta
            + (1 - np.cos(theta)) * omega_hat
            + (theta - np.sin(theta)) * (omega_hat @ omega_hat)
        ) @ v

    # Construct homogeneous transformation matrix
    # Optimization: Direct assignment is faster than np.block
    res = np.eye(4)
    res[:3, :3] = r_rot
    res[:3, 3] = p
    return res


def logarithmic_map(T: np.ndarray) -> tuple[np.ndarray, float]:  # noqa: PLR0911
    """
    Extract screw axis and displacement from transformation.

    Computes the screw axis S and displacement theta from a homogeneous
    transformation matrix T.

    This implements the logarithmic map from SE(3) to se(3):
        [S]*theta = log(T)

    This is the inverse of the exponential map.

    Args:
        T: 4x4 homogeneous transformation matrix

    Returns:
        Tuple of (S, theta) where:
            S: 6x1 normalized screw axis [omega; v]
            theta: Scalar displacement (radians or meters)

    References:
        Murray, Li, Sastry (1994). A Mathematical Introduction to Robotic Manipulation.
        Lynch, Park (2017). Modern Robotics: Mechanics, Planning, and Control.

    Example:
        >>> # Extract screw from a rotation matrix
        >>> T = np.array([[0, -1, 0, 1],
        ...               [1,  0, 0, 0],
        ...               [0,  0, 1, 0],
        ...               [0,  0, 0, 1]])
        >>> S, theta = logarithmic_map(T)
    """
    t_transform = np.asarray(T)
    assert t_transform.shape == (4, 4), f"T must be 4x4, got shape {t_transform.shape}"

    # Extract rotation and position
    r_rot = t_transform[:3, :3]
    p = t_transform[:3, 3]

    # Check if rotation is identity (pure translation or identity)
    if np.linalg.norm(r_rot - np.eye(3), "fro") < 1e-10:
        # Pure translation or identity transformation
        p_norm = np.linalg.norm(p)
        if p_norm < np.finfo(float).eps:
            # Identity transformation: zero rotation and zero translation
            return np.zeros(6), 0.0
        # Pure translation
        s_screw = np.concatenate([np.zeros(3), p / p_norm])
        theta = float(p_norm)
        return s_screw, theta

    # Compute rotation angle using trace
    # trace(R) = 1 + 2*cos(theta)
    cos_theta = (np.trace(r_rot) - 1) / 2
    cos_theta = np.clip(cos_theta, -1, 1)
    theta = np.arccos(cos_theta)

    # Handle special cases
    if abs(theta) < np.finfo(float).eps:
        # No rotation, pure translation or identity
        p_norm = np.linalg.norm(p)
        if p_norm < np.finfo(float).eps:
            # Identity transformation: zero rotation and zero translation
            return np.zeros(6), 0.0
        # Pure translation
        s_screw = np.concatenate([np.zeros(3), p / p_norm])
        theta = float(p_norm)
        return s_screw, theta

    if abs(theta - np.pi) < 1e-10:
        # 180 degree rotation - need special handling
        eigvals, eigvecs = np.linalg.eig(r_rot)
        idx = np.argmin(np.abs(eigvals - 1))
        omega = np.real(eigvecs[:, idx])
        omega = omega / np.linalg.norm(omega)
    else:
        # General case: extract axis from skew-symmetric part
        # omega_hat = (R - R.T) / (2*sin(theta))
        omega_skew = (r_rot - r_rot.T) / (2 * np.sin(theta))
        omega = np.array([omega_skew[2, 1], omega_skew[0, 2], omega_skew[1, 0]])

    # Compute linear velocity component
    omega_hat = skew(omega)
    g_inv = (
        (1 / theta) * np.eye(3)
        - 0.5 * omega_hat
        + (1 / theta - 0.5 / np.tan(theta / 2)) * (omega_hat @ omega_hat)
    )
    v = g_inv @ p

    return np.concatenate([omega, v]), float(theta)
