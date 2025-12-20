"""
Spatial inertia matrices and transformations.

Implements functions for constructing and transforming spatial inertia matrices
used in rigid body dynamics.
"""

from __future__ import annotations

import numpy as np

from .spatial_vectors import skew


def mci(mass: float, com: np.ndarray, i_com: np.ndarray) -> np.ndarray:
    """
    Construct spatial inertia matrix from mass, COM, and rotational inertia.

    The spatial inertia matrix has the form:
        I_spatial = [ I_com + m·cx·cxᵀ    m·cx ]
                    [      m·cxᵀ          m·I3 ]

    where c is the COM vector and cx is its skew-symmetric matrix.

    Args:
        mass: Scalar mass of the body (kg)
        com: 3x1 vector from reference point to center of mass (m)
        i_com: 3x3 rotational inertia tensor about COM (kg·m²)

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
        >>> i_spatial = mci(mass, com, i_sphere)
        >>> i_spatial.shape
        (6, 6)
    """
    com = np.asarray(com).ravel()
    i_com = np.asarray(i_com)

    if not (isinstance(mass, int | float) and mass > 0):
        msg = "mass must be positive scalar"
        raise ValueError(msg)
    if com.shape != (3,):
        msg = f"com must be 3x1 vector, got shape {com.shape}"
        raise ValueError(msg)
    if i_com.shape != (3, 3):
        msg = f"i_com must be 3x3 matrix, got shape {i_com.shape}"
        raise ValueError(msg)

    # Symmetrize inertia tensor (numerical precision)
    i_com = (i_com + i_com.T) / 2

    # Create skew-symmetric matrix for COM vector
    c_skew = skew(com)

    # Parallel axis theorem: transform inertia to reference point
    # Apply Parallel Axis Theorem: I_ref = I_com + mass * (c_skew @ c_skew.T)
    i_ref = i_com + mass * (c_skew @ c_skew.T)

    # Build the spatial inertia matrix
    return np.block([[i_ref, mass * c_skew], [mass * c_skew.T, mass * np.eye(3)]])


def transform_spatial_inertia(i_b: np.ndarray, x_transform: np.ndarray) -> np.ndarray:
    """
    Transform spatial inertia between frames.

    Transforms the spatial inertia matrix from frame B to frame A using
    spatial transformation X.

    The transformation formula is:
        I_A = Xᵀ · I_B · X

    This preserves the symmetric positive-definite properties of the
    spatial inertia matrix.

    Args:
        i_b: 6x6 spatial inertia matrix in frame B
        x_transform: 6x6 spatial transformation from B to A

    Returns:
        6x6 spatial inertia matrix in frame A

    References:
        Featherstone, R. (2008). Rigid Body Dynamics Algorithms.
        Chapter 2: Spatial Vector Algebra, Section 2.9

    Example:
        >>> i_b = mci(1.0, np.zeros(3), 0.01 * np.eye(3))
        >>> x_transform = np.eye(6)  # Identity transform
        >>> i_a = transform_spatial_inertia(i_b, x_transform)
        >>> np.allclose(i_a, i_b)
        True
    """
    i_b = np.asarray(i_b)
    x_transform = np.asarray(x_transform)

    if i_b.shape != (6, 6):
        msg = f"I_B must be 6x6 matrix, got shape {i_b.shape}"
        raise ValueError(msg)
    if x_transform.shape != (6, 6):
        msg = f"X must be 6x6 matrix, got shape {x_transform.shape}"
        raise ValueError(msg)

    # Symmetrize input (numerical precision)
    i_b = (i_b + i_b.T) / 2

    # Transform the inertia matrix
    i_a = x_transform.T @ i_b @ x_transform

    # Ensure result is symmetric
    return (i_a + i_a.T) / 2
