"""
Joint kinematics and motion subspaces.

Implements joint transformation and motion subspace calculations
for various joint types.
"""

import numpy as np

# Constants for motion subspaces to avoid reallocation
S_RX = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
S_RX.flags.writeable = False
S_RY = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
S_RY.flags.writeable = False
S_RZ = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
S_RZ.flags.writeable = False
S_PX = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
S_PX.flags.writeable = False
S_PY = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
S_PY.flags.writeable = False
S_PZ = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
S_PZ.flags.writeable = False


def jcalc(
    jtype: str, q: float, out: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:  # noqa: PLR0915
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
        out: Optional 6x6 array to store the transform matrix. If None, a new array is
            created.

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
    if out is None:
        xj_transform = np.zeros((6, 6))
    else:
        xj_transform = out
        xj_transform.fill(0.0)

    if jtype == "Rx":  # Revolute about x-axis
        c = np.cos(q)
        s = np.sin(q)
        # Top-left 3x3
        xj_transform[0, 0] = 1.0
        xj_transform[1, 1] = c
        xj_transform[1, 2] = -s
        xj_transform[2, 1] = s
        xj_transform[2, 2] = c
        # Bottom-right 3x3
        xj_transform[3, 3] = 1.0
        xj_transform[4, 4] = c
        xj_transform[4, 5] = -s
        xj_transform[5, 4] = s
        xj_transform[5, 5] = c
        s_subspace = S_RX

    elif jtype == "Ry":  # Revolute about y-axis
        c = np.cos(q)
        s = np.sin(q)
        # Top-left 3x3
        xj_transform[0, 0] = c
        xj_transform[0, 2] = s
        xj_transform[1, 1] = 1.0
        xj_transform[2, 0] = -s
        xj_transform[2, 2] = c
        # Bottom-right 3x3
        xj_transform[3, 3] = c
        xj_transform[3, 5] = s
        xj_transform[4, 4] = 1.0
        xj_transform[5, 3] = -s
        xj_transform[5, 5] = c
        s_subspace = S_RY

    elif jtype == "Rz":  # Revolute about z-axis
        c = np.cos(q)
        s = np.sin(q)
        # Top-left 3x3
        xj_transform[0, 0] = c
        xj_transform[0, 1] = -s
        xj_transform[1, 0] = s
        xj_transform[1, 1] = c
        xj_transform[2, 2] = 1.0
        # Bottom-right 3x3
        xj_transform[3, 3] = c
        xj_transform[3, 4] = -s
        xj_transform[4, 3] = s
        xj_transform[4, 4] = c
        xj_transform[5, 5] = 1.0
        s_subspace = S_RZ

    elif jtype == "Px":  # Prismatic along x-axis
        np.fill_diagonal(xj_transform, 1.0)
        xj_transform[4, 2] = q
        xj_transform[5, 1] = -q
        s_subspace = S_PX

    elif jtype == "Py":  # Prismatic along y-axis
        np.fill_diagonal(xj_transform, 1.0)
        xj_transform[3, 2] = -q
        xj_transform[5, 0] = q
        s_subspace = S_PY

    elif jtype == "Pz":  # Prismatic along z-axis
        np.fill_diagonal(xj_transform, 1.0)
        xj_transform[3, 1] = q
        xj_transform[4, 0] = -q
        s_subspace = S_PZ

    else:
        msg = (
            f"Unsupported joint type: {jtype}. "
            f"Supported types: Rx, Ry, Rz, Px, Py, Pz"
        )
        raise ValueError(
            msg,
        )

    return xj_transform, s_subspace
