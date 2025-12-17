"""
Twist and wrench representations for rigid body motions and forces.
"""

from __future__ import annotations

import numpy as np


def twist_to_spatial(
    omega: np.ndarray,
    v: np.ndarray,
    point: np.ndarray | None = None,
) -> np.ndarray:
    """
    Convert twist (angular + linear velocity) to spatial vector.

    A twist represents the instantaneous motion of a rigid body:
    - omega: angular velocity vector
    - v: linear velocity of a point on the body

    Args:
        omega: 3x1 angular velocity vector (rad/s)
        v: 3x1 linear velocity vector (m/s)
        point: 3x1 reference point (optional, default: origin)

    Returns:
        6x1 spatial motion vector [omega; v]

    References:
        Murray, Li, Sastry (1994). A Mathematical Introduction to Robotic Manipulation.
        Lynch, Park (2017). Modern Robotics: Mechanics, Planning, and Control.

    Example:
        >>> # Body rotating about z-axis at 1 rad/s
        >>> omega = np.array([0, 0, 1])
        >>> v = np.array([0, 0, 0])
        >>> V = twist_to_spatial(omega, v)
        >>> V
        array([0., 0., 1., 0., 0., 0.])
    """
    omega = np.asarray(omega).flatten()
    v = np.asarray(v).flatten()

    assert omega.shape == (3,), f"omega must be 3x1, got shape {omega.shape}"
    assert v.shape == (3,), f"v must be 3x1, got shape {v.shape}"

    if point is not None:
        point = np.asarray(point).flatten()
        assert point.shape == (3,), f"point must be 3x1, got shape {point.shape}"

        # Adjust linear velocity: v_new = v_old - omega × point
        v = v - np.cross(omega, point)

    return np.concatenate([omega, v])


def wrench_to_spatial(
    moment: np.ndarray,
    force: np.ndarray,
    point: np.ndarray | None = None,
) -> np.ndarray:
    """
    Convert wrench (moment + force) to spatial force vector.

    A wrench represents forces and moments acting on a rigid body:
    - moment: torque vector
    - force: force vector

    Args:
        moment: 3x1 moment vector (N·m)
        force: 3x1 force vector (N)
        point: 3x1 reference point (optional, default: origin)

    Returns:
        6x1 spatial force vector [moment; force]

    References:
        Murray, Li, Sastry (1994). A Mathematical Introduction to Robotic Manipulation.
        Lynch, Park (2017). Modern Robotics: Mechanics, Planning, and Control.

    Example:
        >>> # Pure force along x-axis applied at origin
        >>> moment = np.array([0, 0, 0])
        >>> force = np.array([10, 0, 0])
        >>> F = wrench_to_spatial(moment, force)
        >>> F
        array([ 0.,  0.,  0., 10.,  0.,  0.])
    """
    moment = np.asarray(moment).flatten()
    force = np.asarray(force).flatten()

    assert moment.shape == (3,), f"moment must be 3x1, got shape {moment.shape}"
    assert force.shape == (3,), f"force must be 3x1, got shape {force.shape}"

    if point is not None:
        point = np.asarray(point).flatten()
        assert point.shape == (3,), f"point must be 3x1, got shape {point.shape}"

        # Adjust moment: moment_new = moment_old + point × force
        moment = moment + np.cross(point, force)

    return np.concatenate([moment, force])
