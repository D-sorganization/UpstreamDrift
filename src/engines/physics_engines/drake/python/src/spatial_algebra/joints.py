"""
Joint kinematics and motion subspaces.

Implements joint transformation and motion subspace calculations
for various joint types.
"""

import numpy as np
import numpy.typing as npt

# Pre-allocate motion subspaces (S vectors) to avoid repeated array creation
S_RX = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
S_RY = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
S_RZ = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
S_PX = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float64)
S_PY = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)
S_PZ = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)


def jcalc(  # noqa: PLR0915
    jtype: str, q: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
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
    # Performance optimization: manually construct matrices to avoid overhead of
    # intermediate arrays, function calls (xrot/xlt), and determinant checks.
    if jtype == "Rx":  # Revolute about x-axis
        c = np.cos(q)
        s = np.sin(q)
        xj_transform = np.zeros((6, 6), dtype=np.float64)
        # Top-left and bottom-right blocks are E
        xj_transform[0, 0] = xj_transform[3, 3] = 1.0
        xj_transform[1, 1] = xj_transform[4, 4] = c
        xj_transform[1, 2] = xj_transform[4, 5] = -s
        xj_transform[2, 1] = xj_transform[5, 4] = s
        xj_transform[2, 2] = xj_transform[5, 5] = c
        return xj_transform, S_RX

    elif jtype == "Ry":  # Revolute about y-axis
        c = np.cos(q)
        s = np.sin(q)
        xj_transform = np.zeros((6, 6), dtype=np.float64)
        xj_transform[1, 1] = xj_transform[4, 4] = 1.0
        xj_transform[0, 0] = xj_transform[3, 3] = c
        xj_transform[0, 2] = xj_transform[3, 5] = s
        xj_transform[2, 0] = xj_transform[5, 3] = -s
        xj_transform[2, 2] = xj_transform[5, 5] = c
        return xj_transform, S_RY

    elif jtype == "Rz":  # Revolute about z-axis
        c = np.cos(q)
        s = np.sin(q)
        xj_transform = np.zeros((6, 6), dtype=np.float64)
        xj_transform[2, 2] = xj_transform[5, 5] = 1.0
        xj_transform[0, 0] = xj_transform[3, 3] = c
        xj_transform[0, 1] = xj_transform[3, 4] = -s
        xj_transform[1, 0] = xj_transform[4, 3] = s
        xj_transform[1, 1] = xj_transform[4, 4] = c
        return xj_transform, S_RZ

    elif jtype == "Px":  # Prismatic along x-axis
        xj_transform = np.eye(6, dtype=np.float64)  # type: ignore[assignment]
        # -skew([q, 0, 0]) -> [0, 0, 0; 0, 0, q; 0, -q, 0]
        xj_transform[4, 2] = q
        xj_transform[5, 1] = -q
        return xj_transform, S_PX

    elif jtype == "Py":  # Prismatic along y-axis
        xj_transform = np.eye(6, dtype=np.float64)  # type: ignore[assignment]
        # -skew([0, q, 0]) -> [0, 0, -q; 0, 0, 0; q, 0, 0]
        xj_transform[3, 2] = -q
        xj_transform[5, 0] = q
        return xj_transform, S_PY

    elif jtype == "Pz":  # Prismatic along z-axis
        xj_transform = np.eye(6, dtype=np.float64)  # type: ignore[assignment]
        # -skew([0, 0, q]) -> [0, q, 0; -q, 0, 0; 0, 0, 0]
        xj_transform[3, 1] = q
        xj_transform[4, 0] = -q
        return xj_transform, S_PZ

    else:
        msg = (
            f"Unsupported joint type: {jtype}. Supported types: Rx, Ry, Rz, Px, Py, Pz"
        )
        raise ValueError(msg)
