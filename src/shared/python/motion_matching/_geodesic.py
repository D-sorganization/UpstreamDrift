"""Quaternion geodesic-distance and canonical orientation helpers (UD-24, #13868).

Consolidates quaternion operations across motion-matching engines so non-unit
simulation quaternions produce consistent behavior.

Canonical conventions:
- Quaternions are ordered ``[w, x, y, z]``.
- Geodesic angle uses ``2 * acos(|q1 . q2|)`` on normalised inputs.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "normalise_quaternions",
    "quaternion_geodesic_angles",
    "quaternion_multiply",
    "quaternion_to_rotmat_series",
    "rotmat_frobenius_distances",
    "rotmat_to_quaternion",
]


def normalise_quaternions(
    q: NDArray[np.float64],
    *,
    eps: float = 1e-12,
) -> NDArray[np.float64]:
    """Normalise quaternion array to unit Euclidean norm along the last axis.

    Args:
        q: Array of shape ``(..., 4)`` representing quaternions in ``[w, x, y, z]``.
        eps: Threshold below which near-zero quaternions fall back to identity.

    Returns:
        Array of unit quaternions with matching shape and float64 dtype.

    Raises:
        ValueError: If trailing dimension is not 4.
    """
    if q.ndim == 0 or q.shape[-1] != 4:
        raise ValueError(f"quaternions must have shape (..., 4); got {q.shape}")
    arr = np.asarray(q, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    zero_mask = (norms < eps).squeeze(-1)
    out = np.where(norms >= eps, arr / np.maximum(norms, eps), 0.0)
    if np.any(zero_mask):
        identity = np.zeros(arr.shape[-1], dtype=np.float64)
        identity[0] = 1.0
        out[zero_mask] = identity
    return out


def _quaternion_abs_dot_products(
    q1: NDArray[np.float64],
    q2: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Validate shapes, normalise inputs (DbC postcondition), and return clamped dots."""
    if q1.shape != q2.shape:
        raise ValueError(f"shape mismatch: q1 {q1.shape} vs q2 {q2.shape}")
    if q1.ndim != 2 or q1.shape[1] != 4:
        raise ValueError(f"quaternions must have shape (N, 4); got {q1.shape}")
    q1_u = normalise_quaternions(q1)
    q2_u = normalise_quaternions(q2)
    # ⚡ Bolt: np.einsum is ~3x faster than np.sum(q1 * q2, axis=1)
    dots = np.abs(np.einsum("ij,ij->i", q1_u, q2_u))
    return np.clip(dots, -1.0, 1.0)


def quaternion_geodesic_angles(
    q1: NDArray[np.float64],
    q2: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Geodesic angle (radians) between matching rows of ``q1`` and ``q2``.

    Normalises inputs prior to computing angles so non-unit simulation quaternions
    give consistent results across all engines.

    Args:
        q1: Quaternions, shape ``(N, 4)`` ordered ``[w, x, y, z]``.
        q2: Quaternions, shape ``(N, 4)`` ordered ``[w, x, y, z]``.

    Returns:
        ``(N,)`` array of angles in radians, each in ``[0, pi]``.

    Raises:
        ValueError: If shapes mismatch or are not ``(N, 4)``.
    """
    dots = _quaternion_abs_dot_products(q1, q2)
    return 2.0 * np.arccos(dots)


def rotmat_frobenius_distances(
    q1: NDArray[np.float64],
    q2: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Frobenius distance ||R1 - R2||_F between matching rotation matrices of q1 and q2.

    For unit quaternions q1 and q2, the squared Frobenius distance between
    their corresponding 3x3 rotation matrices in SO(3) is:
        ||R1 - R2||_F^2 = 8 * (1 - (q1 . q2)^2)

    Args:
        q1: Quaternions, shape ``(N, 4)`` ordered ``[w, x, y, z]``.
        q2: Quaternions, shape ``(N, 4)`` ordered ``[w, x, y, z]``.

    Returns:
        ``(N,)`` array of Frobenius norm distances.

    Raises:
        ValueError: If shapes mismatch or are not ``(N, 4)``.
    """
    dots = _quaternion_abs_dot_products(q1, q2)
    frob_sq = np.maximum(0.0, 8.0 * (1.0 - dots * dots))
    return np.sqrt(frob_sq)


def quaternion_to_rotmat_series(
    quats: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convert (N, 4) quaternions [w, x, y, z] to (N, 3, 3) SO(3) rotation matrices.

    Normalises inputs prior to conversion.

    Args:
        quats: Shape ``(N, 4)`` array of quaternions ``[w, x, y, z]``.

    Returns:
        Shape ``(N, 3, 3)`` array of rotation matrices.

    Raises:
        ValueError: If shape is not ``(N, 4)``.
    """
    if quats.ndim != 2 or quats.shape[1] != 4:
        raise ValueError(f"quaternions must have shape (N, 4); got {quats.shape}")
    q = normalise_quaternions(quats)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    n = len(quats)
    rotmats = np.empty((n, 3, 3), dtype=np.float64)
    rotmats[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
    rotmats[:, 0, 1] = 2.0 * (xy - wz)
    rotmats[:, 0, 2] = 2.0 * (xz + wy)
    rotmats[:, 1, 0] = 2.0 * (xy + wz)
    rotmats[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
    rotmats[:, 1, 2] = 2.0 * (yz - wx)
    rotmats[:, 2, 0] = 2.0 * (xz - wy)
    rotmats[:, 2, 1] = 2.0 * (yz + wx)
    rotmats[:, 2, 2] = 1.0 - 2.0 * (xx + yy)
    return rotmats


def quaternion_multiply(
    q1: NDArray[np.float64],
    q2: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Hamilton product of quaternion arrays, ordered [w, x, y, z].

    Supports broadcasting across matching leading dimensions.

    Args:
        q1: Shape ``(..., 4)`` quaternions ``[w, x, y, z]``.
        q2: Shape ``(..., 4)`` quaternions ``[w, x, y, z]``.

    Returns:
        Product quaternion(s) with broadcast shape.

    Raises:
        ValueError: If trailing dimension of either input is not 4.
    """
    if q1.shape[-1] != 4 or q2.shape[-1] != 4:
        raise ValueError(
            f"quaternions must have trailing dim 4; got {q1.shape} and {q2.shape}"
        )

    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.stack([w, x, y, z], axis=-1)


def rotmat_to_quaternion(
    rotmats: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convert (N, 3, 3) rotation matrices to (N, 4) unit quaternions [w, x, y, z].

    Uses Shepperd's algorithm for numerical stability across all orientations.

    Args:
        rotmats: Shape ``(N, 3, 3)`` or ``(3, 3)`` array of rotation matrices.

    Returns:
        Shape ``(N, 4)`` or ``(4,)`` unit quaternions in canonical ``[w, x, y, z]`` format.

    Raises:
        ValueError: If input shape is not ``(..., 3, 3)``.
    """
    arr = np.asarray(rotmats, dtype=np.float64)
    if arr.ndim < 2 or arr.shape[-2:] != (3, 3):
        raise ValueError(
            f"rotation matrices must have shape (..., 3, 3); got {arr.shape}"
        )

    squeeze_output = arr.ndim == 2
    if squeeze_output:
        arr = arr[np.newaxis, ...]

    n = len(arr)
    quats = np.empty((n, 4), dtype=np.float64)

    for i in range(n):
        r = arr[i]
        trace = r[0, 0] + r[1, 1] + r[2, 2]
        if trace > 0.0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (r[2, 1] - r[1, 2]) * s
            y = (r[0, 2] - r[2, 0]) * s
            z = (r[1, 0] - r[0, 1]) * s
        elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
            w = (r[2, 1] - r[1, 2]) / s
            x = 0.25 * s
            y = (r[0, 1] + r[1, 0]) / s
            z = (r[0, 2] + r[2, 0]) / s
        elif r[1, 1] > r[2, 2]:
            s = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
            w = (r[0, 2] - r[2, 0]) / s
            x = (r[0, 1] + r[1, 0]) / s
            y = 0.25 * s
            z = (r[1, 2] + r[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
            w = (r[1, 0] - r[0, 1]) / s
            x = (r[0, 2] + r[2, 0]) / s
            y = (r[1, 2] + r[2, 1]) / s
            z = 0.25 * s
        quats[i] = [w, x, y, z]

    out_quats = normalise_quaternions(quats)
    return out_quats[0] if squeeze_output else out_quats
