"""
Spatial inertia matrices and transformations.

Implements functions for constructing and transforming spatial inertia matrices
used in rigid body dynamics.
"""

from __future__ import annotations

import numpy as np

from .spatial_vectors import skew


def mcI(mass: float, com: np.ndarray, I_com: np.ndarray) -> np.ndarray:
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

    assert isinstance(mass, int | float), "mass must be number"
    assert mass > 0, "mass must be positive scalar"
    assert com.shape == (3,), f"com must be 3x1 vector, got shape {com.shape}"
    assert i_com.shape == (3, 3), f"I_com must be 3x3 matrix, got shape {i_com.shape}"

    # Symmetrize inertia tensor (numerical precision)
    i_com = (i_com + i_com.T) / 2

    # Create skew-symmetric matrix for COM vector
    c_skew = skew(com)

    # Parallel axis theorem: transform inertia to reference point
    # I = I_com + m * c× * c×ᵀ
    i_ref = i_com + mass * (c_skew @ c_skew.T)

    # Build the spatial inertia matrix
    return np.block([[i_ref, mass * c_skew], [mass * c_skew.T, mass * np.eye(3)]])


def transform_spatial_inertia(I_B: np.ndarray, X: np.ndarray) -> np.ndarray:
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
    i_b = np.asarray(I_B)
    x_transform = np.asarray(X)

    assert i_b.shape == (6, 6), f"I_B must be 6x6 matrix, got shape {i_b.shape}"
    assert x_transform.shape == (
        6,
        6,
    ), f"X must be 6x6 matrix, got shape {x_transform.shape}"

    # Symmetrize input (numerical precision)
    i_b = (i_b + i_b.T) / 2

    # Transform the inertia matrix
    i_a = x_transform.T @ i_b @ x_transform

    # Ensure result is symmetric
    return (i_a + i_a.T) / 2
