"""
Spatial inertia matrices and transformations.

Implements functions for constructing and transforming spatial inertia matrices
used in rigid body dynamics.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .spatial_vectors import skew


def mcI(
    mass: float,
    com: npt.NDArray[np.float64],
    I_com: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Construct spatial inertia matrix from mass, COM, and rotational inertia.

    The spatial inertia matrix has the form:
        I_spatial = [ I_com + m*c_cross*c_cross^T    m*c_cross ]
                    [      m*c_cross^T               m*I_3     ]

    where c is the COM vector and c× is its skew-symmetric matrix.
    """
    com = np.asarray(com).ravel()
    i_com = np.asarray(I_com)

    if not isinstance(mass, (int, float)) or mass <= 0:
        msg = f"mass must be positive scalar, got {mass}"
        raise ValueError(msg)
    if com.shape != (3,):
        msg = f"com must be 3x1 vector, got shape {com.shape}"
        raise ValueError(msg)
    if i_com.shape != (3, 3):
        msg = f"I_com must be 3x3 matrix, got shape {i_com.shape}"
        raise ValueError(msg)

    # Symmetrize inertia tensor (numerical precision)
    i_com = (i_com + i_com.T) / 2

    c_skew = skew(com)
    i_ref = i_com + mass * (c_skew @ c_skew.T)

    res = np.zeros((6, 6), dtype=np.float64)
    res[0:3, 0:3] = i_ref

    mass_c_skew = mass * c_skew
    res[0:3, 3:6] = mass_c_skew
    res[3:6, 0:3] = mass_c_skew.T

    res[3, 3] = mass
    res[4, 4] = mass
    res[5, 5] = mass

    return res


def mci(
    mass: float, com: npt.NDArray[np.float64], i_com: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Alias for mcI to preserve legacy naming."""
    return mcI(mass, com, i_com)


def transform_spatial_inertia(
    I_B: npt.NDArray[np.float64],
    X: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Transform spatial inertia between frames.

    The transformation formula is:
        I_A = Xᵀ · I_B · X
    """
    I_B = np.asarray(I_B)
    X = np.asarray(X)

    if I_B.shape != (6, 6):
        msg = f"I_B must be 6x6 matrix, got shape {I_B.shape}"
        raise ValueError(msg)
    if X.shape != (6, 6):
        msg = f"X must be 6x6 matrix, got shape {X.shape}"
        raise ValueError(msg)

    i_b = (I_B + I_B.T) / 2
    result = X.T @ i_b @ X
    return np.array((result + result.T) / 2, dtype=np.float64)
