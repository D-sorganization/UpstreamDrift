"""
Joint kinematics and motion subspaces.

Implements joint transformation and motion subspace calculations
for various joint types.
"""

from typing import Any

import numpy as np

from .transforms import xlt, xrot


def jcalc(jtype: str, q: float) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """
    Calculate joint transform and motion subspace.

    Calculates the joint transformation matrix and motion subspace vector
    for a given joint type and position.

    Supported joint types:
        'Rx' - Revolute joint about x-axis
        'Ry' - Revolute joint about y-axis
        'Rz' - Revolute joint about z-axis
        'Px' - Prismatic joint along x-axis
        'Py' - Prismatic joint along y-axis
        'Pz' - Prismatic joint along z-axis

    Args:
        jtype: String specifying joint type
        q: Scalar joint position (radians for revolute, meters for prismatic)

    Returns:
        Tuple of (xj_transform, s_subspace) where:
            xj_transform: 6x6 spatial transformation from successor to predecessor
            s_subspace: 6x1 motion subspace vector (joint axis)

    Raises:
        ValueError: If jtype is not supported

    References:
        Featherstone, R. (2008). Rigid Body Dynamics Algorithms.
        Chapter 4: Kinematics

    Examples:
        >>> # Revolute joint about z-axis at 45 degrees
        >>> xj_transform, s_subspace = jcalc('Rz', np.pi/4)
        >>> s_subspace
        array([0., 0., 1., 0., 0., 0.])

        >>> # Prismatic joint along x-axis extended 0.5m
        >>> xj_transform, s_subspace = jcalc('Px', 0.5)
        >>> s_subspace
        array([0., 0., 0., 1., 0., 0.])
    """
    if jtype == "Rx":  # Revolute about x-axis
        c = np.cos(q)
        s = np.sin(q)
        e_rot = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        xj_transform = xrot(e_rot)
        s_subspace = np.array([1, 0, 0, 0, 0, 0])  # Angular velocity about x

    elif jtype == "Ry":  # Revolute about y-axis
        c = np.cos(q)
        s = np.sin(q)
        e_rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        xj_transform = xrot(e_rot)
        s_subspace = np.array([0, 1, 0, 0, 0, 0])  # Angular velocity about y

    elif jtype == "Rz":  # Revolute about z-axis
        c = np.cos(q)
        s = np.sin(q)
        e_rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        xj_transform = xrot(e_rot)
        s_subspace = np.array([0, 0, 1, 0, 0, 0])  # Angular velocity about z

    elif jtype == "Px":  # Prismatic along x-axis
        r = np.array([q, 0, 0])
        xj_transform = xlt(r)
        s_subspace = np.array([0, 0, 0, 1, 0, 0])  # Linear velocity along x

    elif jtype == "Py":  # Prismatic along y-axis
        r = np.array([0, q, 0])
        xj_transform = xlt(r)
        s_subspace = np.array([0, 0, 0, 0, 1, 0])  # Linear velocity along y

    elif jtype == "Pz":  # Prismatic along z-axis
        r = np.array([0, 0, q])
        xj_transform = xlt(r)
        s_subspace = np.array([0, 0, 0, 0, 0, 1])  # Linear velocity along z

    else:
        msg = (
            f"Unsupported joint type: {jtype}. Supported types: Rx, Ry, Rz, Px, Py, Pz"
        )
        raise ValueError(
            msg,
        )

    return xj_transform, s_subspace
