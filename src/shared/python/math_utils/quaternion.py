"""Quaternion helpers shared by motion, pose, and engine adapters."""

from __future__ import annotations

import math
import numpy as np

# Above this absolute dot product the two quaternions are nearly identical and
# the ``sin(theta_0)`` denominator in the SLERP formula approaches zero, so we
# fall back to normalized linear interpolation (nlerp) for numerical stability.
SLERP_LERP_FALLBACK_THRESHOLD = 0.9995


def rotmat_to_quat(rot: np.ndarray) -> np.ndarray:
    """Convert rotation matrix data to canonical ``[w, x, y, z]`` quaternions.

    Args:
        rot: ``(3, 3)`` or ``(N, 3, 3)`` float array.

    Returns:
        Unit quaternion array with the scalar component sign-canonicalized
        so ``w >= 0``.
    """
    arr = np.asarray(rot, dtype=np.float64)
    if arr.ndim == 2 and arr.shape == (3, 3):
        return _canonicalize_sign(_single_rotmat_to_quat(arr))
    if arr.ndim == 3 and arr.shape[1:] == (3, 3):
        out = np.empty((arr.shape[0], 4), dtype=np.float64)
        for i in range(arr.shape[0]):
            out[i] = _single_rotmat_to_quat(arr[i])
        return _canonicalize_sign(out)
    raise ValueError(f"rot must have shape (3, 3) or (..., 3, 3); got {arr.shape}")


def _single_rotmat_to_quat(r: np.ndarray) -> np.ndarray:
    """Shepperd's method for a single rotation matrix."""
    m00, m01, m02 = r[0, 0], r[0, 1], r[0, 2]
    m10, m11, m12 = r[1, 0], r[1, 1], r[1, 2]
    m20, m21, m22 = r[2, 0], r[2, 1], r[2, 2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif (m00 > m11) and (m00 > m22):
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    norm = math.sqrt(
        np.dot(q, q)
    )  # ⚡ Bolt: math.sqrt(np.dot) is faster than np.linalg.norm
    if norm == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def _canonicalize_sign(q: np.ndarray) -> np.ndarray:
    """Flip rows whose ``w`` is negative so ``q`` and ``-q`` collapse to one."""
    if q.ndim == 1:
        return -q if q[0] < 0.0 else q.copy()
    flips = q[:, 0] < 0.0
    if np.any(flips):
        q = q.copy()
        q[flips] = -q[flips]
    return q


def quat_inverse_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """Return the SO(3) rotation angle ``θ`` in radians between two orientations.

    Quaternions use ``[w, x, y, z]`` order. Inputs are normalized; antipodal
    ``q`` and ``-q`` represent the same rotation and yield ``θ = 0``. The angle
    lies in ``[0, π]`` and matches
    :func:`motion_matching._geodesic.quaternion_geodesic_angles` for a single pair.
    """
    a = np.asarray(q1, dtype=np.float64).reshape(-1)
    b = np.asarray(q2, dtype=np.float64).reshape(-1)
    if a.shape != (4,) or b.shape != (4,):
        raise ValueError(
            f"quaternions must have shape (4,); got {a.shape} and {b.shape}"
        )
    a_norm = math.sqrt(np.dot(a, a))
    b_norm = math.sqrt(np.dot(b, b))
    if a_norm == 0.0 or b_norm == 0.0:
        raise ValueError("quaternions must be non-zero")
    a = a / a_norm
    b = b / b_norm
    dot = float(np.dot(a, b))
    flipped = dot < 0.0
    if flipped:
        b = -b
        dot = -dot
    diff_norm = math.sqrt(np.dot(a - b, a - b))
    if diff_norm < 1e-15:
        return 0.0
    if dot > 1.0 - 1e-15 and not flipped:
        w0, x0, y0, z0 = a
        w1, x1, y1, z1 = b
        rw = w0 * w1 + x0 * x1 + y0 * y1 + z0 * z1
        rx = w0 * x1 - x0 * w1 - y0 * z1 + z0 * y1
        ry = w0 * y1 - x0 * z1 + y0 * w1 - z0 * x1
        rz = w0 * z1 - x0 * y1 + y0 * x1 + z0 * w1
        imag = math.sqrt(rx * rx + ry * ry + rz * rz)
        return float(2.0 * math.atan2(imag, abs(rw)))
    dot = min(1.0, max(-1.0, dot))
    return float(2.0 * math.atan2(math.sqrt(max(0.0, 1.0 - dot * dot)), dot))


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two unit quaternions."""
    a = np.asarray(q0, dtype=np.float64)
    b = np.asarray(q1, dtype=np.float64)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    if dot > SLERP_LERP_FALLBACK_THRESHOLD:
        result = a + t * (b - a)
        return result / math.sqrt(
            np.dot(result, result)
        )  # ⚡ Bolt: math.sqrt(np.dot) is faster than np.linalg.norm
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return s0 * a + s1 * b


def slerp_series(
    query_t: np.ndarray, raw_t: np.ndarray, raw_q: np.ndarray
) -> np.ndarray:
    """SLERP an ``(N, 4)`` quaternion series onto ``query_t``, clamping endpoints."""
    query = np.asarray(query_t, dtype=np.float64)
    source_t = np.asarray(raw_t, dtype=np.float64)
    source_q = np.asarray(raw_q, dtype=np.float64)
    out = np.empty((query.shape[0], 4), dtype=np.float64)
    last = source_t.shape[0] - 1
    for i, t in enumerate(query):
        if t <= source_t[0]:
            out[i] = source_q[0]
            continue
        if t >= source_t[last]:
            out[i] = source_q[last]
            continue
        j = int(np.searchsorted(source_t, t)) - 1
        j = max(0, min(j, last - 1))
        span = source_t[j + 1] - source_t[j]
        alpha = 0.0 if span == 0.0 else float((t - source_t[j]) / span)
        out[i] = slerp(source_q[j], source_q[j + 1], alpha)
    # ⚡ Bolt: np.sum(**2) is faster than einsum for this shape
    norms = np.sqrt(np.sum(out**2, axis=1, keepdims=True))
    norms[norms == 0.0] = 1.0
    return out / norms
