"""Minimal SE(3) helpers for the canonical pose convention.

The canonical convention represents a rigid-body transform as a
``(translation_xyz_m, rotation_xyz_deg)`` pair where
``rotation_xyz_deg`` is **intrinsic XYZ Euler in degrees**, matching
:func:`forward_kinematics` in
``src.shared.python.motion_matching.diagnostics.forward_kinematics``.

This module exposes pure-numpy helpers; it has no engine dependencies.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt

_EPS_ORTHO = 1e-9


def _rx(deg: float) -> npt.NDArray[np.float64]:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _ry(deg: float) -> npt.NDArray[np.float64]:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def _rz(deg: float) -> npt.NDArray[np.float64]:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def euler_xyz_deg_to_matrix(
    rotation_xyz_deg: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Convert intrinsic XYZ Euler (degrees) to a 3x3 rotation matrix.

    Matches the convention in
    :func:`src.shared.python.motion_matching.diagnostics.forward_kinematics.forward_kinematics`:
    ``R = Rx(x) @ Ry(y) @ Rz(z)`` (intrinsic / body-fixed XYZ).
    """
    arr = np.asarray(rotation_xyz_deg, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"rotation_xyz_deg must have shape (3,), got {arr.shape}")
    return _rx(arr[0]) @ _ry(arr[1]) @ _rz(arr[2])


def matrix_to_euler_xyz_deg(
    matrix: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Inverse of :func:`euler_xyz_deg_to_matrix` for non-singular matrices.

    Decomposes a 3x3 rotation matrix produced by ``Rx(x) @ Ry(y) @ Rz(z)``
    back to ``(x, y, z)`` in degrees. Singular at ``y = +-90 deg``
    (gimbal lock); the canonical golfer pose stays comfortably away from
    that range so the caller does not need to handle it.
    """
    m = np.asarray(matrix, dtype=float)
    if m.shape != (3, 3):
        raise ValueError(f"matrix must have shape (3, 3), got {m.shape}")
    sy = m[0, 2]
    sy = float(np.clip(sy, -1.0, 1.0))
    y = np.arcsin(sy)
    cy = np.cos(y)
    if abs(cy) < 1e-9:
        # Gimbal lock; pin x, fold rotation into z.
        x = 0.0
        z = float(np.arctan2(m[1, 0], m[1, 1]))
    else:
        x = float(np.arctan2(-m[1, 2], m[2, 2]))
        z = float(np.arctan2(-m[0, 1], m[0, 0]))
    return np.degrees(np.array([x, float(y), z]))


def se3_from_xyz_xyz_deg(
    translation_m: npt.ArrayLike,
    rotation_xyz_deg: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Build a 4x4 SE(3) matrix from (translation_m, intrinsic_xyz_deg)."""
    t = np.asarray(translation_m, dtype=float)
    if t.shape != (3,):
        raise ValueError(f"translation_m must have shape (3,), got {t.shape}")
    out = np.eye(4, dtype=float)
    out[:3, :3] = euler_xyz_deg_to_matrix(rotation_xyz_deg)
    out[:3, 3] = t
    return out


def se3_to_xyz_xyz_deg(
    matrix: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Inverse of :func:`se3_from_xyz_xyz_deg`.

    Returns ``(translation_m, rotation_xyz_deg)``.
    """
    m = np.asarray(matrix, dtype=float)
    if m.shape != (4, 4):
        raise ValueError(f"matrix must have shape (4, 4), got {m.shape}")
    return m[:3, 3].copy(), matrix_to_euler_xyz_deg(m[:3, :3])


def is_valid_se3(matrix: npt.ArrayLike, *, tol: float = _EPS_ORTHO) -> bool:
    """Return True if *matrix* is a valid 4x4 SE(3) (orthonormal R, [0 0 0 1] last row)."""
    m = np.asarray(matrix, dtype=float)
    if m.shape != (4, 4):
        return False
    if not np.all(np.isfinite(m)):
        return False
    if not np.allclose(m[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=tol):
        return False
    r = m[:3, :3]
    if not np.allclose(r @ r.T, np.eye(3), atol=tol * 100):
        return False
    return abs(np.linalg.det(r) - 1.0) <= tol * 100


def compose_se3(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return ``a @ b`` after validating both inputs are SE(3)."""
    am = np.asarray(a, dtype=float)
    bm = np.asarray(b, dtype=float)
    if not is_valid_se3(am):
        raise ValueError("compose_se3: first argument is not a valid SE(3) matrix")
    if not is_valid_se3(bm):
        raise ValueError("compose_se3: second argument is not a valid SE(3) matrix")
    return am @ bm


def inverse_se3(matrix: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return the SE(3) inverse of *matrix* (transpose-rotation, negate-translation)."""
    m = np.asarray(matrix, dtype=float)
    if not is_valid_se3(m):
        raise ValueError("inverse_se3: input is not a valid SE(3) matrix")
    out = np.eye(4, dtype=float)
    rt = m[:3, :3].T
    out[:3, :3] = rt
    out[:3, 3] = -rt @ m[:3, 3]
    return out


# ---------------------------------------------------------------------------
# canonical-v2 quaternion + manifold helpers (CC-2, ADR-0026)
#
# Quaternions are unit, scalar-first ``(w, x, y, z)`` per
# ``docs/conventions/canonical-v2.md``. ``quat_exp`` maps a rotation vector
# (axis * angle, radians) to a unit quaternion; ``quat_log`` is its inverse on
# the principal branch ``||rotvec|| < pi``. These are the building blocks the
# canonical-v2 ``CanonicalState`` and the engine adapters (CC-9/CC-10) use to
# update the floating base on its manifold instead of by naive vector addition.
# ---------------------------------------------------------------------------

_EPS_QUAT: Final[float] = 1e-12
_EPS_SMALL_ANGLE: Final[float] = 1e-8


def quat_normalize(q: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return *q* scaled to unit norm. Raises if *q* has (near-)zero norm."""
    arr = np.asarray(q, dtype=float)
    if arr.shape != (4,):
        raise ValueError(f"quaternion must have shape (4,), got {arr.shape}")
    norm = float(np.linalg.norm(arr))
    if norm < _EPS_QUAT:
        raise ValueError("cannot normalize a zero-norm quaternion")
    return arr / norm


def quat_conjugate(q: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return the conjugate ``(w, -x, -y, -z)``; the inverse for a unit quaternion."""
    arr = np.asarray(q, dtype=float)
    if arr.shape != (4,):
        raise ValueError(f"quaternion must have shape (4,), got {arr.shape}")
    return np.array([arr[0], -arr[1], -arr[2], -arr[3]], dtype=float)


def quat_multiply(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Hamilton product ``a (x) b`` of two scalar-first quaternions."""
    qa = np.asarray(a, dtype=float)
    qb = np.asarray(b, dtype=float)
    if qa.shape != (4,) or qb.shape != (4,):
        raise ValueError("both quaternions must have shape (4,)")
    w1, x1, y1, z1 = qa
    w2, x2, y2, z2 = qb
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def quat_exp(rotvec: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Exponential map: rotation vector (axis * angle, rad) -> unit quaternion.

    ``rotvec`` is ``axis * theta`` with ``theta`` in radians. Returns the unit
    quaternion ``(cos(theta/2), sin(theta/2) * axis)``, scalar-first. The
    small-angle branch uses a Taylor series for numerical stability.
    """
    v = np.asarray(rotvec, dtype=float)
    if v.shape != (3,):
        raise ValueError(f"rotvec must have shape (3,), got {v.shape}")
    theta = float(np.linalg.norm(v))
    half = 0.5 * theta
    w = float(np.cos(half))
    if theta < _EPS_SMALL_ANGLE:
        # sin(theta/2) / theta -> 1/2 - theta^2/48 as theta -> 0
        scale = 0.5 - theta * theta / 48.0
    else:
        scale = float(np.sin(half)) / theta
    return quat_normalize(np.array([w, v[0] * scale, v[1] * scale, v[2] * scale]))


def quat_log(q: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Logarithm map: unit quaternion -> rotation vector (axis * angle, rad).

    Inverse of :func:`quat_exp` on the principal branch ``||rotvec|| < pi``.
    The sign is canonicalised so ``q`` and ``-q`` (the same rotation) return the
    same minimal rotation vector.
    """
    arr = quat_normalize(q)
    if arr[0] < 0.0:  # shortest-path: keep the half-angle in [0, pi/2]
        arr = -arr
    w = float(np.clip(arr[0], -1.0, 1.0))
    v = arr[1:4]
    nv = float(np.linalg.norm(v))
    if nv < _EPS_SMALL_ANGLE:
        # theta -> 0: rotvec = 2 v / w to leading order (w -> 1 here)
        return v * (2.0 / w)
    theta = 2.0 * float(np.arctan2(nv, w))
    return v * (theta / nv)


def quat_to_matrix(q: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Convert a scalar-first unit quaternion to a 3x3 rotation matrix."""
    w, x, y, z = quat_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def matrix_to_quat(matrix: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Convert a 3x3 rotation matrix to a scalar-first unit quaternion (Shepperd)."""
    m = np.asarray(matrix, dtype=float)
    if m.shape != (3, 3):
        raise ValueError(f"matrix must have shape (3, 3), got {m.shape}")
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return quat_normalize(np.array([w, x, y, z], dtype=float))


def euler_xyz_deg_to_quat_wxyz(
    rotation_xyz_deg: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Convert a ``canonical-v1`` intrinsic-XYZ-degrees rotation to a quaternion.

    Reuses :func:`euler_xyz_deg_to_matrix` so the v1 -> v2 migration shares one
    Euler convention (DRY); the result is a scalar-first unit quaternion.
    """
    return matrix_to_quat(euler_xyz_deg_to_matrix(rotation_xyz_deg))
