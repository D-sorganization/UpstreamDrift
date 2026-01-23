"""
Spatial vector operations and cross products.

Implements spatial cross product operators for motion and force vectors
following Featherstone's spatial vector algebra notation.
"""

from typing import Literal  # noqa: ICN003

import numpy as np
import numpy.typing as npt


def skew(v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Create 3x3 skew-symmetric matrix from 3x1 vector.

    The skew-symmetric matrix satisfies: skew(v) @ u = cross(v, u)

    Args:
        v: 3x1 vector

    Returns:
        3x3 skew-symmetric matrix

    Example:
        >>> v = np.array([1, 2, 3])
        >>> S = skew(v)
        >>> u = np.array([4, 5, 6])
        >>> np.allclose(S @ u, np.cross(v, u))
        True
    """
    # Performance optimization: use ravel() instead of flatten() to avoid copy
    v = np.asarray(v).ravel()
    if v.shape != (3,):
        msg = f"Input must be 3x1 vector, got shape {v.shape}"
        raise ValueError(msg)

    # Performance optimization: manual assignment is faster than np.array creation
    v0, v1, v2 = v[0], v[1], v[2]
    res = np.zeros((3, 3), dtype=np.float64)
    res[0, 1] = -v2
    res[0, 2] = v1
    res[1, 0] = v2
    res[1, 2] = -v0
    res[2, 0] = -v1
    res[2, 1] = v0

    return res


def crm(v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Spatial cross product operator for motion vectors.

    Returns the 6x6 matrix X such that X @ m = v × m for any
    spatial motion vector m, where × is the spatial cross product.

    The cross product operator has the form:
        crm(v) = [ skew(ω)    0      ]
                 [ skew(v)  skew(ω)  ]

    where v = [ω; v] with ω being angular velocity and v being linear velocity.

    Args:
        v: 6x1 spatial motion vector [angular; linear]

    Returns:
        6x6 spatial cross product matrix

    References:
        Featherstone, R. (2008). Rigid Body Dynamics Algorithms.
        Chapter 2: Spatial Vector Algebra

    Example:
        >>> v = np.array([1, 0, 0, 0, 1, 0])  # Angular and linear velocity
        >>> X = crm(v)
        >>> X.shape
        (6, 6)
    """
    v = np.asarray(v).ravel()
    if v.shape != (6,):
        msg = f"Input must be 6x1 spatial vector, got shape {v.shape}"
        raise ValueError(msg)

    w0, w1, w2 = v[0], v[1], v[2]
    v0, v1, v2 = v[3], v[4], v[5]

    # Performance optimization: manual construction avoids np.block and arrays
    res = np.zeros((6, 6), dtype=np.float64)

    # Top-left block: skew(w)
    res[0, 1] = -w2
    res[0, 2] = w1
    res[1, 0] = w2
    res[1, 2] = -w0
    res[2, 0] = -w1
    res[2, 1] = w0

    # Bottom-left block: skew(v_lin)
    res[3, 1] = -v2
    res[3, 2] = v1
    res[4, 0] = v2
    res[4, 2] = -v0
    res[5, 0] = -v1
    res[5, 1] = v0

    # Bottom-right block: skew(w)
    res[3, 4] = -w2
    res[3, 5] = w1
    res[4, 3] = w2
    res[4, 5] = -w0
    res[5, 3] = -w1
    res[5, 4] = w0

    return res


def crf(v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Spatial cross product operator for force vectors (dual).

    Returns the 6x6 matrix X such that X @ f = v ×* f for any
    spatial force vector f, where ×* is the dual spatial cross product.

    The dual cross product operator has the form:
        crf(v) = -crm(v)ᵀ = [ skew(ω)   skew(v) ]
                            [   0       skew(ω) ]

    where v = [ω; v].

    Args:
        v: 6x1 spatial motion vector [angular; linear]

    Returns:
        6x6 dual spatial cross product matrix

    References:
        Featherstone, R. (2008). Rigid Body Dynamics Algorithms.
        Chapter 2: Spatial Vector Algebra

    Example:
        >>> v = np.array([1, 0, 0, 0, 1, 0])
        >>> X_crf = crf(v)
        >>> X_crm = crm(v)
        >>> np.allclose(X_crf, -X_crm.T)
        True
    """
    v = np.asarray(v).ravel()
    if v.shape != (6,):
        msg = f"Input must be 6x1 spatial vector, got shape {v.shape}"
        raise ValueError(msg)

    w0, w1, w2 = v[0], v[1], v[2]
    v0, v1, v2 = v[3], v[4], v[5]

    # Performance optimization: manual construction avoids np.block and arrays
    res = np.zeros((6, 6), dtype=np.float64)

    # Top-left block: skew(w)
    res[0, 1] = -w2
    res[0, 2] = w1
    res[1, 0] = w2
    res[1, 2] = -w0
    res[2, 0] = -w1
    res[2, 1] = w0

    # Top-right block: skew(v_lin)
    res[0, 4] = -v2
    res[0, 5] = v1
    res[1, 3] = v2
    res[1, 5] = -v0
    res[2, 3] = -v1
    res[2, 4] = v0

    # Bottom-right block: skew(w)
    res[3, 4] = -w2
    res[3, 5] = w1
    res[4, 3] = w2
    res[4, 5] = -w0
    res[5, 3] = -w1
    res[5, 4] = w0

    return res


def spatial_cross(  # noqa: PLR0915
    v: npt.NDArray[np.float64],
    u: npt.NDArray[np.float64],
    cross_type: Literal["motion", "force"] = "motion",
) -> npt.NDArray[np.float64]:
    """
    Compute spatial cross product.

    This function implements the spatial cross product directly without
    constructing the intermediate 6x6 matrices (crm/crf), providing
    better performance.

    Args:
        v: 6x1 spatial motion vector [angular; linear]
        u: 6x1 spatial vector (motion or force depending on type)
        cross_type: Type of cross product ('motion' or 'force')

    Returns:
        6x1 spatial vector resulting from cross product

    Raises:
        ValueError: If cross_type is not 'motion' or 'force'

    Examples:
        >>> # Motion cross product (acceleration)
        >>> v = np.array([1, 0, 0, 0, 1, 0])  # Velocity
        >>> a = np.array([0, 1, 0, 0, 0, 1])  # Acceleration
        >>> bias = spatial_cross(v, a, 'motion')

        >>> # Force cross product (wrench transformation)
        >>> v = np.array([1, 0, 0, 0, 1, 0])  # Velocity
        >>> f = np.array([0, 0, 10, 0, 0, 0])  # Force/torque
        >>> f_transformed = spatial_cross(v, f, 'force')
    """
    v = np.asarray(v).ravel()
    u = np.asarray(u).ravel()

    if v.shape != (6,):
        msg = f"v must be 6x1 spatial vector, got shape {v.shape}"
        raise ValueError(msg)
    if u.shape != (6,):
        msg = f"u must be 6x1 spatial vector, got shape {u.shape}"
        raise ValueError(msg)

    # Decompose vectors: v = [w; v_lin], u = [u_rot; u_lin]
    # We use manual cross product calculation for performance
    # (avoiding np.cross overhead)
    w0, w1, w2 = v[0], v[1], v[2]
    v0, v1, v2 = v[3], v[4], v[5]
    ur0, ur1, ur2 = u[0], u[1], u[2]
    ul0, ul1, ul2 = u[3], u[4], u[5]

    if cross_type == "motion":
        # crm(v) * u = [w x u_rot; v_lin x u_rot + w x u_lin]

        # w x u_rot
        rx = w1 * ur2 - w2 * ur1
        ry = w2 * ur0 - w0 * ur2
        rz = w0 * ur1 - w1 * ur0

        # v_lin x u_rot
        vx = v1 * ur2 - v2 * ur1
        vy = v2 * ur0 - v0 * ur2
        vz = v0 * ur1 - v1 * ur0

        # w x u_lin
        wx = w1 * ul2 - w2 * ul1
        wy = w2 * ul0 - w0 * ul2
        wz = w0 * ul1 - w1 * ul0

        return np.array([rx, ry, rz, vx + wx, vy + wy, vz + wz])

    if cross_type == "force":
        # crf(v) * u = [w x u_rot + v_lin x u_lin; w x u_lin]
        # Note: For force vectors, u_rot is torque/moment, u_lin is force

        # w x u_rot
        rx = w1 * ur2 - w2 * ur1
        ry = w2 * ur0 - w0 * ur2
        rz = w0 * ur1 - w1 * ur0

        # v_lin x u_lin
        vx = v1 * ul2 - v2 * ul1
        vy = v2 * ul0 - v0 * ul2
        vz = v0 * ul1 - v1 * ul0

        # w x u_lin
        wx = w1 * ul2 - w2 * ul1
        wy = w2 * ul0 - w0 * ul2
        wz = w0 * ul1 - w1 * ul0

        return np.array([rx + vx, ry + vy, rz + vz, wx, wy, wz])

    # Runtime check for invalid cross_type
    # Note: mypy flags this as unreachable due to Literal type,
    # but it's needed for runtime safety
    msg = f"cross_type must be 'motion' or 'force', got '{cross_type}'"  # type: ignore[unreachable]
    raise ValueError(msg)
