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

    Args:
        mass: Scalar mass of the body (kg)
        com: 3x1 vector from reference point to center of mass (m)
        I_com: 3x3 rotational inertia tensor about COM (kg·m²)

    Returns:
        6x6 spatial inertia matrix

    References:
        Featherstone, R. (2008). Rigid Body Dynamics Algorithms.
        Chapter 2: Spatial Vector Algebra, Section 2.8

    Example:
        >>> # Uniform density sphere of radius 0.1m and mass 1kg
        >>> mass = 1.0
        >>> radius = 0.1
        >>> com = np.array([0, 0, 0])  # COM at reference point
        >>> i_sphere = (2/5) * mass * radius**2 * np.eye(3)
        >>> i_spatial = mcI(mass, com, i_sphere)
        >>> i_spatial.shape
        (6, 6)
    """
    com = np.asarray(com).flatten()
    i_com = np.asarray(I_com)

    if mass <= 0:
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

    # Create skew-symmetric matrix for COM vector
    c_skew = skew(com)

    # Parallel axis theorem: transform inertia to reference point
    # I = I_com + m * c× * c×ᵀ
    i_ref = i_com + mass * (c_skew @ c_skew.T)

    # Performance optimization: manual construction avoids np.block overhead
    # and temporary array creation for mass * np.eye(3)
    res = np.zeros((6, 6), dtype=np.float64)
    res[0:3, 0:3] = i_ref

    mass_c_skew = mass * c_skew
    res[0:3, 3:6] = mass_c_skew
    res[3:6, 0:3] = mass_c_skew.T

    # Diagonal mass block
    res[3, 3] = mass
    res[4, 4] = mass
    res[5, 5] = mass

    return res


def transform_spatial_inertia(
    I_B: npt.NDArray[np.float64],
    X: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Transform spatial inertia between frames.

    Transforms the spatial inertia matrix from frame B to frame A using
    spatial transformation X.

    The transformation formula is:
        I_A = Xᵀ · I_B · X

    This preserves the symmetric positive-definite properties of the
    spatial inertia matrix.

    Args:
        I_B: 6x6 spatial inertia matrix in frame B
        X: 6x6 spatial transformation from B to A

    Returns:
        6x6 spatial inertia matrix in frame A

    References:
        Featherstone, R. (2008). Rigid Body Dynamics Algorithms.
        Chapter 2: Spatial Vector Algebra, Section 2.9

    Example:
        >>> i_b = mcI(1.0, np.zeros(3), 0.01 * np.eye(3))
        >>> x_transform = np.eye(6)  # Identity transform
        >>> i_a = transform_spatial_inertia(i_b, x_transform)
        >>> np.allclose(i_a, i_b)
        True
    """
    I_B = np.asarray(I_B)
    X = np.asarray(X)

    if I_B.shape != (6, 6):
        msg = f"I_B must be 6x6 matrix, got shape {I_B.shape}"
        raise ValueError(msg)
    if X.shape != (6, 6):
        msg = f"X must be 6x6 matrix, got shape {X.shape}"
        raise ValueError(msg)

    return X.T @ I_B @ X
