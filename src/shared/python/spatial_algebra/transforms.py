"""
Spatial coordinate transformations (Plücker transforms).

Implements spatial transformation matrices for converting spatial vectors
between different coordinate frames.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .spatial_vectors import skew


def xrot(e_rot: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Spatial coordinate transformation for pure rotation."""
    e_rot = np.asarray(e_rot)
    if e_rot.shape != (3, 3):
        msg = f"E must be 3x3 rotation matrix, got shape {e_rot.shape}"
        raise ValueError(msg)

    det_e = np.linalg.det(e_rot)
    if not np.isclose(det_e, 1.0, atol=1e-6):
        msg = f"E may not be a valid rotation matrix (det={det_e})"
        raise ValueError(msg)

    res = np.zeros((6, 6), dtype=np.float64)
    res[0:3, 0:3] = e_rot
    res[3:6, 3:6] = e_rot
    return res


def xlt(r: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Spatial coordinate transformation for pure translation."""
    r = np.asarray(r).ravel()
    if r.shape != (3,):
        msg = f"r must be 3x1 vector, got shape {r.shape}"
        raise ValueError(msg)

    r_skew = skew(r)
    res = np.eye(6, dtype=np.float64)
    res[3:6, 0:3] = -r_skew
    return res


def xtrans(
    e_rot: npt.NDArray[np.float64], r: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """General spatial coordinate transformation (Plücker transform)."""
    e_rot = np.asarray(e_rot)
    r = np.asarray(r).ravel()

    if e_rot.shape != (3, 3):
        msg = f"E must be 3x3 matrix, got shape {e_rot.shape}"
        raise ValueError(msg)
    if r.shape != (3,):
        msg = f"r must be 3x1 vector, got shape {r.shape}"
        raise ValueError(msg)

    r_skew = skew(r)

    res = np.zeros((6, 6), dtype=np.float64)
    res[0:3, 0:3] = e_rot
    res[3:6, 0:3] = -e_rot @ r_skew
    res[3:6, 3:6] = e_rot
    return res


def inv_xtrans(
    e_rot: npt.NDArray[np.float64],
    r: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Inverse of spatial coordinate transformation."""
    e_rot = np.asarray(e_rot)
    r = np.asarray(r).ravel()

    if e_rot.shape != (3, 3):
        msg = f"E must be 3x3 matrix, got shape {e_rot.shape}"
        raise ValueError(msg)
    if r.shape != (3,):
        msg = f"r must be 3x1 vector, got shape {r.shape}"
        raise ValueError(msg)

    e_t = e_rot.T
    r_skew = skew(r)

    res = np.zeros((6, 6), dtype=np.float64)
    res[0:3, 0:3] = e_t
    res[3:6, 0:3] = r_skew @ e_t
    res[3:6, 3:6] = e_t
    return res
