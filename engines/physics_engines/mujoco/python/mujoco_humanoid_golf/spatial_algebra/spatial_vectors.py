"""
Spatial vector operations and cross products.

Implements spatial cross product operators for motion and force vectors
following Featherstone's spatial vector algebra notation.
"""

import typing

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
    v = np.asarray(v)
    if v.shape != (3,):
        original_shape = v.shape
        v = v.ravel()  # Optimization: Avoid copy if possible
        if v.shape != (3,):
            msg = f"Input must be 3x1 vector, got shape {original_shape}"
            raise ValueError(msg)

    # Optimization: Direct assignment to pre-allocated array is faster than
    # creating a list of lists and converting to array (~1.5x speedup)
    res = np.zeros((3, 3), dtype=v.dtype)
    res[0, 1] = -v[2]
    res[0, 2] = v[1]
    res[1, 0] = v[2]
    res[1, 2] = -v[0]
    res[2, 0] = -v[1]
    res[2, 1] = v[0]
    return res


def crm(v: np.ndarray) -> np.ndarray:
    """
    Spatial cross product operator for motion vectors.

    Returns the 6x6 matrix X such that X @ m = v x m for any
    spatial motion vector m, where x is the spatial cross product.

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
    v = np.asarray(v)
    if v.shape != (6,):
        original_shape = v.shape
        v = v.ravel()  # Optimization: Avoid copy if possible
        if v.shape != (6,):
            msg = f"Input must be 6x1 spatial vector, got shape {original_shape}"
            raise ValueError(msg)

    w = v[:3]  # Angular velocity
    vlin = v[3:]  # Linear velocity

    # Optimization: Direct assignment is faster than np.block
    res = np.zeros((6, 6))

    # w_skew in top left
    res[0, 1] = -w[2]
    res[0, 2] = w[1]
    res[1, 0] = w[2]
    res[1, 2] = -w[0]
    res[2, 0] = -w[1]
    res[2, 1] = w[0]

    # Copy w_skew to bottom right
    res[3:6, 3:6] = res[0:3, 0:3]

    # v_skew in bottom left
    # indices relative to block start (3,0)
    res[3, 1] = -vlin[2]
    res[3, 2] = vlin[1]
    res[4, 0] = vlin[2]
    res[4, 2] = -vlin[0]
    res[5, 0] = -vlin[1]
    res[5, 1] = vlin[0]

    return res


def crf(v: np.ndarray) -> np.ndarray:
    """
    Spatial cross product operator for force vectors (dual).

    Returns the 6x6 matrix X such that X @ f = v x* f for any
    spatial force vector f, where x* is the dual spatial cross product.

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
    v = np.asarray(v)
    if v.shape != (6,):
        original_shape = v.shape
        v = v.ravel()  # Optimization: Avoid copy if possible
        if v.shape != (6,):
            msg = f"Input must be 6x1 spatial vector, got shape {original_shape}"
            raise ValueError(msg)

    w = v[:3]  # Angular velocity
    vlin = v[3:]  # Linear velocity

    # Optimization: Direct assignment is faster than np.block
    res = np.zeros((6, 6))

    # w_skew (top left)
    res[0, 1] = -w[2]
    res[0, 2] = w[1]
    res[1, 0] = w[2]
    res[1, 2] = -w[0]
    res[2, 0] = -w[1]
    res[2, 1] = w[0]

    # w_skew (bottom right)
    res[3:6, 3:6] = res[0:3, 0:3]

    # v_skew (top right)
    res[0, 4] = -vlin[2]
    res[0, 5] = vlin[1]
    res[1, 3] = vlin[2]
    res[1, 5] = -vlin[0]
    res[2, 3] = -vlin[1]
    res[2, 4] = vlin[0]

    return res


def cross_motion(
    v: np.ndarray, m: np.ndarray, out: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute spatial cross product v x m efficiently.

    Equivalent to crm(v) @ m but avoids constructing the 6x6 matrix.

    Args:
        v: 6x1 spatial motion vector [angular; linear]
        m: 6x1 spatial motion vector
        out: Optional 6x1 output array to store result. If None, a new array is created.

    Returns:
        6x1 spatial vector
    """
    v = np.asarray(v)
    if v.shape != (6,):
        orig_v_shape = v.shape
        v = v.ravel()  # Optimization: Avoid copy if possible
        if v.shape != (6,):
            msg = f"v must be 6x1 spatial vector, got shape {orig_v_shape}"
            raise ValueError(msg)

    m = np.asarray(m)
    if m.shape != (6,):
        orig_m_shape = m.shape
        m = m.ravel()  # Optimization: Avoid copy if possible
        if m.shape != (6,):
            msg = f"m must be 6x1 spatial vector, got shape {orig_m_shape}"
            raise ValueError(msg)

    # Optimization: Direct assignment to pre-allocated array is faster than
    # creating a list of values and converting to array (~7% speedup).
    # Use result_type to correctly handle mixed types (e.g., int/float)
    if out is None:
        res = np.empty(6, dtype=np.result_type(v, m))
    else:
        res = out

    # Unpack components (v = [w; v_lin])
    # Result:
    # [ w x m_w ]
    # [ v_lin x m_w + w x m_lin ]

    # w x m_w
    res[0] = v[1] * m[2] - v[2] * m[1]
    res[1] = v[2] * m[0] - v[0] * m[2]
    res[2] = v[0] * m[1] - v[1] * m[0]

    # v_lin x m_w + w x m_lin
    res[3] = v[4] * m[2] - v[5] * m[1] + v[1] * m[5] - v[2] * m[4]
    res[4] = v[5] * m[0] - v[3] * m[2] + v[2] * m[3] - v[0] * m[5]
    res[5] = v[3] * m[1] - v[4] * m[0] + v[0] * m[4] - v[1] * m[3]

    return res


def cross_force(
    v: np.ndarray, f: np.ndarray, out: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute spatial force cross product v x* f efficiently.

    Equivalent to crf(v) @ f but avoids constructing the 6x6 matrix.

    Args:
        v: 6x1 spatial motion vector [angular; linear]
        f: 6x1 spatial force vector [moment; force]
        out: Optional 6x1 output array to store result. If None, a new array is created.

    Returns:
        6x1 spatial vector
    """
    v = np.asarray(v)
    if v.shape != (6,):
        orig_v_shape = v.shape
        v = v.ravel()  # Optimization: Avoid copy if possible
        if v.shape != (6,):
            msg = f"v must be 6x1 spatial vector, got shape {orig_v_shape}"
            raise ValueError(msg)

    f = np.asarray(f)
    if f.shape != (6,):
        orig_f_shape = f.shape
        f = f.ravel()  # Optimization: Avoid copy if possible
        if f.shape != (6,):
            msg = f"f must be 6x1 spatial vector, got shape {orig_f_shape}"
            raise ValueError(msg)

    # Optimization: Direct assignment to pre-allocated array is faster than
    # creating a list of values and converting to array (~7% speedup).
    # Use result_type to correctly handle mixed types (e.g., int/float)
    if out is None:
        res = np.empty(6, dtype=np.result_type(v, f))
    else:
        res = out

    # Unpack components (v = [w; v_lin])
    # Result:
    # [ w x tau + v_lin x force ]
    # [ w x force ]

    # Top: w x tau + v_lin x force
    res[0] = v[1] * f[2] - v[2] * f[1] + v[4] * f[5] - v[5] * f[4]
    res[1] = v[2] * f[0] - v[0] * f[2] + v[5] * f[3] - v[3] * f[5]
    res[2] = v[0] * f[1] - v[1] * f[0] + v[3] * f[4] - v[4] * f[3]

    # Bot: w x force
    res[3] = v[1] * f[5] - v[2] * f[4]
    res[4] = v[2] * f[3] - v[0] * f[5]
    res[5] = v[0] * f[4] - v[1] * f[3]

    return res


def cross_motion_fast(v: np.ndarray, m: np.ndarray, out: np.ndarray) -> None:
    """
    Optimized version of cross_motion without shape checks or asarray.
    Assumes inputs are 6x1 arrays and out is provided.

    Args:
        v: 6x1 spatial motion vector
        m: 6x1 spatial motion vector
        out: 6x1 output array
    """
    # w x m_w
    out[0] = v[1] * m[2] - v[2] * m[1]
    out[1] = v[2] * m[0] - v[0] * m[2]
    out[2] = v[0] * m[1] - v[1] * m[0]

    # v_lin x m_w + w x m_lin
    out[3] = v[4] * m[2] - v[5] * m[1] + v[1] * m[5] - v[2] * m[4]
    out[4] = v[5] * m[0] - v[3] * m[2] + v[2] * m[3] - v[0] * m[5]
    out[5] = v[3] * m[1] - v[4] * m[0] + v[0] * m[4] - v[1] * m[3]


def cross_force_fast(v: np.ndarray, f: np.ndarray, out: np.ndarray) -> None:
    """
    Optimized version of cross_force without shape checks or asarray.
    Assumes inputs are 6x1 arrays and out is provided.

    Args:
        v: 6x1 spatial motion vector
        f: 6x1 spatial force vector
        out: 6x1 output array
    """
    # Top: w x tau + v_lin x force
    out[0] = v[1] * f[2] - v[2] * f[1] + v[4] * f[5] - v[5] * f[4]
    out[1] = v[2] * f[0] - v[0] * f[2] + v[5] * f[3] - v[3] * f[5]
    out[2] = v[0] * f[1] - v[1] * f[0] + v[3] * f[4] - v[4] * f[3]

    # Bot: w x force
    out[3] = v[1] * f[5] - v[2] * f[4]
    out[4] = v[2] * f[3] - v[0] * f[5]
    out[5] = v[0] * f[4] - v[1] * f[3]


def cross_motion_axis(
    v: np.ndarray, axis_idx: int, val: float, out: np.ndarray
) -> None:
    """
    Computes v x m where m is a sparse vector with only m[axis_idx] = val.

    Optimized to skip multiplications by zero.

    Args:
        v: 6x1 spatial motion vector
        axis_idx: Index of non-zero component in m (0-5)
        val: Value of the non-zero component
        out: 6x1 output array
    """
    # Unrolled logic for sparse m
    # indices: 0=Rx, 1=Ry, 2=Rz, 3=Px, 4=Py, 5=Pz

    if axis_idx == 0:  # Rx: m = [val, 0, 0, 0, 0, 0]
        out[0] = 0.0
        out[1] = v[2] * val
        out[2] = -v[1] * val
        out[3] = 0.0
        out[4] = v[5] * val
        out[5] = -v[4] * val

    elif axis_idx == 1:  # Ry: m = [0, val, 0, 0, 0, 0]
        out[0] = -v[2] * val
        out[1] = 0.0
        out[2] = v[0] * val
        out[3] = -v[5] * val
        out[4] = 0.0
        out[5] = v[3] * val

    elif axis_idx == 2:  # Rz: m = [0, 0, val, 0, 0, 0]
        out[0] = v[1] * val
        out[1] = -v[0] * val
        out[2] = 0.0
        out[3] = v[4] * val
        out[4] = -v[3] * val
        out[5] = 0.0

    elif axis_idx == 3:  # Px: m = [0, 0, 0, val, 0, 0]
        out[0] = 0.0
        out[1] = 0.0
        out[2] = 0.0
        out[3] = 0.0
        out[4] = v[2] * val
        out[5] = -v[1] * val

    elif axis_idx == 4:  # Py: m = [0, 0, 0, 0, val, 0]
        out[0] = 0.0
        out[1] = 0.0
        out[2] = 0.0
        out[3] = -v[2] * val
        out[4] = 0.0
        out[5] = v[0] * val

    elif axis_idx == 5:  # Pz: m = [0, 0, 0, 0, 0, val]
        out[0] = 0.0
        out[1] = 0.0
        out[2] = 0.0
        out[3] = v[1] * val
        out[4] = -v[0] * val
        out[5] = 0.0


def spatial_cross(
    v: np.ndarray,
    u: np.ndarray,
    cross_type: typing.Literal["motion", "force"] = "motion",
) -> np.ndarray:
    """
    Compute spatial cross product.

    This is a convenience function that uses optimized cross_motion or cross_force.

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
    if cross_type == "motion":
        return cross_motion(v, u)
    if cross_type == "force":
        return cross_force(v, u)
    msg = f"cross_type must be 'motion' or 'force', got '{cross_type}'"
    raise ValueError(msg)
