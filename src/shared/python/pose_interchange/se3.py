"""Minimal SE(3) helpers for the canonical pose convention.

The canonical convention represents a rigid-body transform as a
``(translation_xyz_m, rotation_xyz_deg)`` pair where
``rotation_xyz_deg`` is **intrinsic XYZ Euler in degrees**, matching
:func:`forward_kinematics` in
``src.shared.python.motion_matching.diagnostics.forward_kinematics``.

This module exposes pure-numpy helpers; it has no engine dependencies.
"""

from __future__ import annotations

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
