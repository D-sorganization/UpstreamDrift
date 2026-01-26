"""
Spatial vector operations and cross products.

Implements spatial cross product operators for motion and force vectors
following Featherstone's spatial vector algebra notation.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt


def skew(v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Create 3x3 skew-symmetric matrix from 3x1 vector.

    The skew-symmetric matrix satisfies: skew(v) @ u = cross(v, u)
    """
    v = np.asarray(v).ravel()
    if v.shape != (3,):
        msg = f"Input must be 3x1 vector, got shape {v.shape}"
        raise ValueError(msg)

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
    """Spatial cross product operator for motion vectors."""
    v = np.asarray(v).ravel()
    if v.shape != (6,):
        msg = f"Input must be 6x1 spatial vector, got shape {v.shape}"
        raise ValueError(msg)

    w0, w1, w2 = v[0], v[1], v[2]
    v0, v1, v2 = v[3], v[4], v[5]

    res = np.zeros((6, 6), dtype=np.float64)

    res[0, 1] = -w2
    res[0, 2] = w1
    res[1, 0] = w2
    res[1, 2] = -w0
    res[2, 0] = -w1
    res[2, 1] = w0

    res[3, 1] = -v2
    res[3, 2] = v1
    res[4, 0] = v2
    res[4, 2] = -v0
    res[5, 0] = -v1
    res[5, 1] = v0

    res[3, 4] = -w2
    res[3, 5] = w1
    res[4, 3] = w2
    res[4, 5] = -w0
    res[5, 3] = -w1
    res[5, 4] = w0

    return res


def crf(v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Spatial cross product operator for force vectors (dual)."""
    v = np.asarray(v).ravel()
    if v.shape != (6,):
        msg = f"Input must be 6x1 spatial vector, got shape {v.shape}"
        raise ValueError(msg)

    w0, w1, w2 = v[0], v[1], v[2]
    v0, v1, v2 = v[3], v[4], v[5]

    res = np.zeros((6, 6), dtype=np.float64)

    res[0, 1] = -w2
    res[0, 2] = w1
    res[1, 0] = w2
    res[1, 2] = -w0
    res[2, 0] = -w1
    res[2, 1] = w0

    res[0, 4] = -v2
    res[0, 5] = v1
    res[1, 3] = v2
    res[1, 5] = -v0
    res[2, 3] = -v1
    res[2, 4] = v0

    res[3, 4] = -w2
    res[3, 5] = w1
    res[4, 3] = w2
    res[4, 5] = -w0
    res[5, 3] = -w1
    res[5, 4] = w0

    return res


def cross_motion(
    v: npt.NDArray[np.float64],
    m: npt.NDArray[np.float64],
    out: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """Compute spatial cross product v x m efficiently."""
    v = np.asarray(v)
    if v.shape != (6,):
        orig_v_shape = v.shape
        v = v.ravel()
        if v.shape != (6,):
            msg = f"v must be 6x1 spatial vector, got shape {orig_v_shape}"
            raise ValueError(msg)

    m = np.asarray(m)
    if m.shape != (6,):
        orig_m_shape = m.shape
        m = m.ravel()
        if m.shape != (6,):
            msg = f"m must be 6x1 spatial vector, got shape {orig_m_shape}"
            raise ValueError(msg)

    if out is None:
        res = np.empty(6, dtype=np.result_type(v, m))
    else:
        res = out

    res[0] = v[1] * m[2] - v[2] * m[1]
    res[1] = v[2] * m[0] - v[0] * m[2]
    res[2] = v[0] * m[1] - v[1] * m[0]

    res[3] = v[4] * m[2] - v[5] * m[1] + v[1] * m[5] - v[2] * m[4]
    res[4] = v[5] * m[0] - v[3] * m[2] + v[2] * m[3] - v[0] * m[5]
    res[5] = v[3] * m[1] - v[4] * m[0] + v[0] * m[4] - v[1] * m[3]

    return res


def cross_force(
    v: npt.NDArray[np.float64],
    f: npt.NDArray[np.float64],
    out: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """Compute spatial force cross product v x* f efficiently."""
    v = np.asarray(v)
    if v.shape != (6,):
        orig_v_shape = v.shape
        v = v.ravel()
        if v.shape != (6,):
            msg = f"v must be 6x1 spatial vector, got shape {orig_v_shape}"
            raise ValueError(msg)

    f = np.asarray(f)
    if f.shape != (6,):
        orig_f_shape = f.shape
        f = f.ravel()
        if f.shape != (6,):
            msg = f"f must be 6x1 spatial vector, got shape {orig_f_shape}"
            raise ValueError(msg)

    if out is None:
        res = np.empty(6, dtype=np.result_type(v, f))
    else:
        res = out

    res[0] = v[1] * f[2] - v[2] * f[1] + v[4] * f[5] - v[5] * f[4]
    res[1] = v[2] * f[0] - v[0] * f[2] + v[5] * f[3] - v[3] * f[5]
    res[2] = v[0] * f[1] - v[1] * f[0] + v[3] * f[4] - v[4] * f[3]

    res[3] = v[1] * f[5] - v[2] * f[4]
    res[4] = v[2] * f[3] - v[0] * f[5]
    res[5] = v[0] * f[4] - v[1] * f[3]

    return res


def cross_motion_fast(
    v: npt.NDArray[np.float64], m: npt.NDArray[np.float64], out: npt.NDArray[np.float64]
) -> None:
    """Optimized version of cross_motion without shape checks."""
    out[0] = v[1] * m[2] - v[2] * m[1]
    out[1] = v[2] * m[0] - v[0] * m[2]
    out[2] = v[0] * m[1] - v[1] * m[0]

    out[3] = v[4] * m[2] - v[5] * m[1] + v[1] * m[5] - v[2] * m[4]
    out[4] = v[5] * m[0] - v[3] * m[2] + v[2] * m[3] - v[0] * m[5]
    out[5] = v[3] * m[1] - v[4] * m[0] + v[0] * m[4] - v[1] * m[3]


def cross_force_fast(
    v: npt.NDArray[np.float64], f: npt.NDArray[np.float64], out: npt.NDArray[np.float64]
) -> None:
    """Optimized version of cross_force without shape checks."""
    out[0] = v[1] * f[2] - v[2] * f[1] + v[4] * f[5] - v[5] * f[4]
    out[1] = v[2] * f[0] - v[0] * f[2] + v[5] * f[3] - v[3] * f[5]
    out[2] = v[0] * f[1] - v[1] * f[0] + v[3] * f[4] - v[4] * f[3]

    out[3] = v[1] * f[5] - v[2] * f[4]
    out[4] = v[2] * f[3] - v[0] * f[5]
    out[5] = v[0] * f[4] - v[1] * f[3]


def cross_motion_axis(
    v: npt.NDArray[np.float64], axis_idx: int, val: float, out: npt.NDArray[np.float64]
) -> None:
    """Compute v x m where m is sparse with only one non-zero component."""
    if axis_idx == 0:
        out[0] = 0.0
        out[1] = v[2] * val
        out[2] = -v[1] * val
        out[3] = 0.0
        out[4] = v[5] * val
        out[5] = -v[4] * val
    elif axis_idx == 1:
        out[0] = -v[2] * val
        out[1] = 0.0
        out[2] = v[0] * val
        out[3] = -v[5] * val
        out[4] = 0.0
        out[5] = v[3] * val
    elif axis_idx == 2:
        out[0] = v[1] * val
        out[1] = -v[0] * val
        out[2] = 0.0
        out[3] = v[4] * val
        out[4] = -v[3] * val
        out[5] = 0.0
    elif axis_idx == 3:
        out[0] = 0.0
        out[1] = 0.0
        out[2] = 0.0
        out[3] = 0.0
        out[4] = v[2] * val
        out[5] = -v[1] * val
    elif axis_idx == 4:
        out[0] = 0.0
        out[1] = 0.0
        out[2] = 0.0
        out[3] = -v[2] * val
        out[4] = 0.0
        out[5] = v[0] * val
    elif axis_idx == 5:
        out[0] = 0.0
        out[1] = 0.0
        out[2] = 0.0
        out[3] = v[1] * val
        out[4] = -v[0] * val
        out[5] = 0.0


def spatial_cross(
    v: npt.NDArray[np.float64],
    u: npt.NDArray[np.float64],
    cross_type: Literal["motion", "force"] = "motion",
) -> npt.NDArray[np.float64]:
    """Compute spatial cross product."""
    if cross_type == "motion":
        return cross_motion(v, u)
    if cross_type == "force":
        return cross_force(v, u)
    msg = f"cross_type must be 'motion' or 'force', got '{cross_type}'"
    raise ValueError(msg)
