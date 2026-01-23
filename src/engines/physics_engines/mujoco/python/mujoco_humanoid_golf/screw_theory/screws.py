"""
Screw axis representations and conversions.
"""

import numpy as np

from .exponential import exponential_map


def screw_axis(axis: np.ndarray, point: np.ndarray, pitch: float = 0.0) -> np.ndarray:
    """
    Compute screw axis representation.

    A screw axis represents the instantaneous motion of a rigid body
    rotating about and/or translating along an axis in space.

    For pure rotation (pitch = 0):
        S = [omega; -omega × q]
    where omega is the unit axis direction and q is a point on the axis.

    For pure translation (pitch = inf):
        S = [0; v]
    where v is the unit direction of translation.

    For general screw motion:
        S = [omega; v + h*omega]
    where h is the pitch (distance translated per radian).

    Args:
        axis: 3x1 unit direction vector of the screw axis
        point: 3x1 point on the screw axis (m)
        pitch: Scalar pitch value (m/rad) (default: 0 for pure rotation)
               Use np.inf for pure translation

    Returns:
        6x1 normalized screw axis

    References:
        Murray, Li, Sastry (1994). A Mathematical Introduction to Robotic Manipulation.
        Lynch, Park (2017). Modern Robotics: Mechanics, Planning, and Control.

    Examples:
        >>> # Rotation about z-axis passing through origin
        >>> S = screw_axis(np.array([0, 0, 1]), np.array([0, 0, 0]))
        >>> S
        array([0., 0., 1., 0., 0., 0.])

        >>> # Translation along x-axis
        >>> S = screw_axis(np.array([1, 0, 0]), np.array([0, 0, 0]), np.inf)
        >>> S
        array([0., 0., 0., 1., 0., 0.])
    """
    axis = np.asarray(axis).flatten()
    point = np.asarray(point).flatten()

    assert axis.shape == (3,), f"axis must be 3x1, got shape {axis.shape}"
    assert point.shape == (3,), f"point must be 3x1, got shape {point.shape}"

    # Normalize axis direction
    omega = axis / np.linalg.norm(axis)

    if np.isinf(pitch):
        # Pure translation (prismatic joint)
        return np.concatenate([np.zeros(3), omega])
    # Rotation or screw motion
    # Linear velocity component: v = -omega × q + h*omega
    v = -np.cross(omega, point) + pitch * omega
    return np.concatenate([omega, v])


def screw_to_transform(
    axis: np.ndarray,
    point: np.ndarray,
    pitch: float,
    theta: float,
) -> np.ndarray:
    """
    Compute transformation from screw motion parameters.

    This is a convenience function combining screw_axis and exponential_map.

    Args:
        axis: 3x1 unit direction vector of screw axis
        point: 3x1 point on the screw axis
        pitch: Scalar pitch (m/rad), use np.inf for pure translation
        theta: Displacement along screw (rad or m)

    Returns:
        4x4 homogeneous transformation matrix

    Examples:
        >>> # 90° rotation about z-axis through origin
        >>> t_transform = screw_to_transform(
        ...     np.array([0, 0, 1]), np.array([0, 0, 0]), 0, np.pi/2
        ... )

        >>> # Translation of 1m along x-axis
        >>> t_transform = screw_to_transform(
        ...     np.array([1, 0, 0]), np.array([0, 0, 0]), np.inf, 1.0
        ... )
    """
    s_screw = screw_axis(axis, point, pitch)
    return exponential_map(s_screw, theta)
