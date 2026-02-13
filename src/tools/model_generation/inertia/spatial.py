"""
Spatial inertia utilities for rigid body dynamics.

This module provides functions for working with 6x6 spatial inertia
matrices used in articulated body algorithms.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.core.contracts import precondition


@precondition(lambda mass, com, I_com: mass > 0, "Mass must be positive")
def mcI(
    mass: float,
    com: NDArray[np.float64],
    I_com: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Construct 6x6 spatial inertia matrix from mass, COM, and rotational inertia.

    The spatial inertia has the structure:
        [ I_com + m*c×c×ᵀ    m*c× ]
        [      m*c×ᵀ        m*I₃ ]

    where c× is the skew-symmetric matrix of the COM vector.

    Args:
        mass: Mass in kg
        com: Center of mass position [x, y, z] in local frame
        I_com: 3x3 rotational inertia matrix at COM

    Returns:
        6x6 spatial inertia matrix

    Raises:
        ValueError: If inputs have wrong dimensions
    """
    com = np.asarray(com, dtype=np.float64).flatten()
    I_com = np.asarray(I_com, dtype=np.float64)

    if com.shape != (3,):
        raise ValueError(f"COM must be 3-vector, got shape {com.shape}")
    if I_com.shape != (3, 3):
        raise ValueError(f"I_com must be 3x3, got shape {I_com.shape}")

    # Symmetrize I_com for numerical stability
    I_com = 0.5 * (I_com + I_com.T)

    # Skew-symmetric matrix of COM
    c_skew = _skew(com)

    # Construct spatial inertia
    # Upper left: I + m * c× * c×ᵀ = I - m * c× * c×
    I_rot = I_com - mass * c_skew @ c_skew

    # Upper right / lower left: m * c×
    mc_skew = mass * c_skew

    # Lower right: m * I₃
    m_eye = mass * np.eye(3)

    # Assemble 6x6 matrix
    spatial_inertia = np.zeros((6, 6), dtype=np.float64)
    spatial_inertia[:3, :3] = I_rot
    spatial_inertia[:3, 3:] = mc_skew
    spatial_inertia[3:, :3] = mc_skew.T
    spatial_inertia[3:, 3:] = m_eye

    return spatial_inertia


def transform_spatial_inertia(
    I_A: NDArray[np.float64],
    X_BA: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Transform spatial inertia from frame A to frame B.

    Uses the relationship: I_B = X_BA^T * I_A * X_BA

    Args:
        I_A: 6x6 spatial inertia in frame A
        X_BA: 6x6 spatial transform from A to B

    Returns:
        6x6 spatial inertia in frame B
    """
    I_A = np.asarray(I_A, dtype=np.float64)
    X_BA = np.asarray(X_BA, dtype=np.float64)

    if I_A.shape != (6, 6) or X_BA.shape != (6, 6):
        raise ValueError("Both matrices must be 6x6")

    return X_BA.T @ I_A @ X_BA


def spatial_inertia_to_urdf(
    spatial_inertia: NDArray[np.float64],
) -> dict[str, Any]:
    """
    Extract URDF-compatible inertia parameters from spatial inertia.

    Args:
        spatial_inertia: 6x6 spatial inertia matrix

    Returns:
        Dict with mass, com (xyz), and inertia (ixx, iyy, izz, ixy, ixz, iyz)
    """
    spatial_inertia = np.asarray(spatial_inertia, dtype=np.float64)

    if spatial_inertia.shape != (6, 6):
        raise ValueError("Spatial inertia must be 6x6")

    # Extract mass from lower-right block
    mass = spatial_inertia[3, 3]

    if mass <= 0:
        raise ValueError(f"Invalid mass extracted: {mass}")

    # Extract m*c× from upper-right block
    mc_skew = spatial_inertia[:3, 3:]

    # Recover COM from skew matrix: c× = [0, -cz, cy; cz, 0, -cx; -cy, cx, 0]
    com = (
        np.array(
            [
                mc_skew[2, 1],  # cx = (m*c×)[2,1] / m
                mc_skew[0, 2],  # cy = (m*c×)[0,2] / m
                mc_skew[1, 0],  # cz = (m*c×)[1,0] / m
            ]
        )
        / mass
    )

    # Extract rotational inertia
    # I_rot = I_com - m * c× * c×, so I_com = I_rot + m * c× * c×
    c_skew = _skew(com)
    I_rot = spatial_inertia[:3, :3]
    I_com = I_rot + mass * c_skew @ c_skew

    # Symmetrize
    I_com = 0.5 * (I_com + I_com.T)

    return {
        "mass": float(mass),
        "com": [float(com[0]), float(com[1]), float(com[2])],
        "ixx": float(I_com[0, 0]),
        "iyy": float(I_com[1, 1]),
        "izz": float(I_com[2, 2]),
        "ixy": float(I_com[0, 1]),
        "ixz": float(I_com[0, 2]),
        "iyz": float(I_com[1, 2]),
    }


@precondition(
    lambda mass, com, ixx, iyy, izz, ixy=0.0, ixz=0.0, iyz=0.0: mass > 0,
    "Mass must be positive",
)
def urdf_to_spatial_inertia(
    mass: float,
    com: tuple[float, float, float] | list[float] | NDArray,
    ixx: float,
    iyy: float,
    izz: float,
    ixy: float = 0.0,
    ixz: float = 0.0,
    iyz: float = 0.0,
) -> NDArray[np.float64]:
    """
    Create spatial inertia matrix from URDF parameters.

    Args:
        mass: Mass in kg
        com: Center of mass [x, y, z]
        ixx, iyy, izz: Diagonal inertia elements
        ixy, ixz, iyz: Off-diagonal inertia elements

    Returns:
        6x6 spatial inertia matrix
    """
    I_com = np.array(
        [
            [ixx, ixy, ixz],
            [ixy, iyy, iyz],
            [ixz, iyz, izz],
        ],
        dtype=np.float64,
    )

    return mcI(mass, np.asarray(com), I_com)


def spatial_transform(
    rotation: NDArray[np.float64],
    translation: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Create 6x6 spatial transform matrix.

    The spatial transform has structure:
        [ R    0 ]
        [ t×R  R ]

    where t× is the skew-symmetric matrix of translation.

    Args:
        rotation: 3x3 rotation matrix
        translation: 3-vector translation

    Returns:
        6x6 spatial transform
    """
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64).flatten()

    if rotation.shape != (3, 3):
        raise ValueError(f"Rotation must be 3x3, got {rotation.shape}")
    if translation.shape != (3,):
        raise ValueError(f"Translation must be 3-vector, got {translation.shape}")

    t_skew = _skew(translation)

    X = np.zeros((6, 6), dtype=np.float64)
    X[:3, :3] = rotation
    X[3:, :3] = t_skew @ rotation
    X[3:, 3:] = rotation

    return X


def _skew(v: NDArray[np.float64]) -> NDArray[np.float64]:
    """Create skew-symmetric matrix from 3-vector."""
    v = np.asarray(v, dtype=np.float64).flatten()
    return np.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ],
        dtype=np.float64,
    )


def composite_rigid_body_inertia(
    inertias: list[tuple[NDArray[np.float64], NDArray[np.float64]]],
) -> NDArray[np.float64]:
    """
    Compute composite rigid body inertia from multiple spatial inertias.

    Args:
        inertias: List of (spatial_inertia, transform_to_composite_frame) tuples

    Returns:
        Combined 6x6 spatial inertia in composite frame
    """
    result = np.zeros((6, 6), dtype=np.float64)

    for I_i, X_ci in inertias:
        # Transform inertia to composite frame and add
        result += transform_spatial_inertia(I_i, X_ci)

    return result
