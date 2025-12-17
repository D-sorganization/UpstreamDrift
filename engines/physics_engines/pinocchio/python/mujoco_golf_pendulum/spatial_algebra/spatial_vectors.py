"""
Spatial vector operations and cross products.

Implements spatial cross product operators for motion and force vectors
following Featherstone's spatial vector algebra notation.
"""

from typing import Literal

import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
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
    v = np.asarray(v).flatten()
    assert v.shape == (3,), f"Input must be 3x1 vector, got shape {v.shape}"

    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def crm(v: np.ndarray) -> np.ndarray:
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
    v = np.asarray(v).flatten()
    assert v.shape == (6,), f"Input must be 6x1 spatial vector, got shape {v.shape}"

    w = v[:3]  # Angular velocity
    vlin = v[3:]  # Linear velocity

    w_skew = skew(w)
    v_skew = skew(vlin)

    return np.block([[w_skew, np.zeros((3, 3))], [v_skew, w_skew]])


def crf(v: np.ndarray) -> np.ndarray:
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
    v = np.asarray(v).flatten()
    assert v.shape == (6,), f"Input must be 6x1 spatial vector, got shape {v.shape}"

    w = v[:3]  # Angular velocity
    vlin = v[3:]  # Linear velocity

    w_skew = skew(w)
    v_skew = skew(vlin)

    return np.block([[w_skew, v_skew], [np.zeros((3, 3)), w_skew]])


def spatial_cross(
    v: np.ndarray,
    u: np.ndarray,
    cross_type: Literal["motion", "force"] = "motion",
) -> np.ndarray:
    """
    Compute spatial cross product.

    This is a convenience function that uses crm or crf operators.

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
    v = np.asarray(v).flatten()
    u = np.asarray(u).flatten()

    assert v.shape == (6,), f"v must be 6x1 spatial vector, got shape {v.shape}"
    assert u.shape == (6,), f"u must be 6x1 spatial vector, got shape {u.shape}"

    if cross_type == "motion":
        return crm(v) @ u
    if cross_type == "force":
        return crf(v) @ u
    msg = f"cross_type must be 'motion' or 'force', got '{cross_type}'"
    raise ValueError(msg)  # type: ignore[unreachable]
