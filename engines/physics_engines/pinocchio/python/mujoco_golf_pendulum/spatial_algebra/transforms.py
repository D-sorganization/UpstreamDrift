"""
Spatial coordinate transformations (Plücker transforms).

Implements spatial transformation matrices for converting spatial vectors
between different coordinate frames.
"""

import numpy as np

from .spatial_vectors import skew


def xrot(e_rot: np.ndarray) -> np.ndarray:
    """
    Spatial coordinate transformation for pure rotation.

    Returns the 6x6 spatial transformation matrix for a pure rotation
    described by the 3x3 rotation matrix E.

    The transformation matrix has the form:
        X = [ E    0 ]
            [ 0    E ]

    Args:
        e_rot: 3x3 rotation matrix from frame A to frame B

    Returns:
        6x6 spatial transformation matrix

    References:
        Featherstone, R. (2008). Rigid Body Dynamics Algorithms.
        Chapter 2: Spatial Vector Algebra

    Example:
        >>> # 90 degree rotation about z-axis
        >>> theta = np.pi / 2
        >>> e_rot = np.array([[np.cos(theta), -np.sin(theta), 0],
        ...               [np.sin(theta),  np.cos(theta), 0],
        ...               [0, 0, 1]])
        >>> x_transform = xrot(e_rot)
        >>> x_transform.shape
        (6, 6)
    """
    e_rot = np.asarray(e_rot)
    assert e_rot.shape == (
        3,
        3,
    ), f"E must be 3x3 rotation matrix, got shape {e_rot.shape}"

    # Verify it's a valid rotation matrix (optional check)
    det_e = np.linalg.det(e_rot)
    if not np.isclose(det_e, 1.0, atol=1e-6):
        msg = f"E may not be a valid rotation matrix (det={det_e})"
        raise ValueError(msg)

    return np.block([[e_rot, np.zeros((3, 3))], [np.zeros((3, 3)), e_rot]])


def xlt(r: np.ndarray) -> np.ndarray:
    """
    Spatial coordinate transformation for pure translation.

    Returns the 6x6 spatial transformation matrix for a pure
    translation by vector r.

    The transformation matrix has the form:
        X = [ I      0   ]
            [ -r×    I   ]

    where r× is the skew-symmetric matrix of r.

    Args:
        r: 3x1 translation vector from A to B, expressed in frame A

    Returns:
        6x6 spatial transformation matrix

    References:
        Featherstone, R. (2008). Rigid Body Dynamics Algorithms.
        Chapter 2: Spatial Vector Algebra

    Example:
        >>> r = np.array([1, 2, 3])
        >>> x_transform = xlt(r)
        >>> x_transform.shape
        (6, 6)
    """
    r = np.asarray(r).flatten()
    assert r.shape == (3,), f"r must be 3x1 vector, got shape {r.shape}"

    r_skew = skew(r)

    return np.block([[np.eye(3), np.zeros((3, 3))], [-r_skew, np.eye(3)]])


def xtrans(e_rot: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    General spatial coordinate transformation (Plücker transform).

    Returns the 6x6 spatial transformation matrix for a general rigid body
    transformation consisting of rotation E and translation r.

    The transformation matrix (Plücker transform) has the form:
        X = [ E        0     ]
            [ -E·r×    E     ]

    where r× is the skew-symmetric matrix of r.

    For motion vectors: v_A = X @ v_B
    For force vectors:  f_B = X.T @ f_A  (note the transpose)

    Args:
        e_rot: 3x3 rotation matrix from frame A to frame B
        r: 3x1 translation vector from A to B origin, expressed in frame A

    Returns:
        6x6 spatial transformation matrix (Plücker transform)

    References:
        Featherstone, R. (2008). Rigid Body Dynamics Algorithms.
        Chapter 2: Spatial Vector Algebra, Section 2.6

    Example:
        >>> e_rot = np.eye(3)
        >>> r = np.array([1, 0, 0])
        >>> x_transform = xtrans(e_rot, r)
        >>> x_transform.shape
        (6, 6)
    """
    e_rot = np.asarray(e_rot)
    r = np.asarray(r).flatten()

    assert e_rot.shape == (3, 3), f"E must be 3x3 matrix, got shape {e_rot.shape}"
    assert r.shape == (3,), f"r must be 3x1 vector, got shape {r.shape}"

    r_skew = skew(r)

    return np.block([[e_rot, np.zeros((3, 3))], [-e_rot @ r_skew, e_rot]])


def inv_xtrans(e_rot: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Inverse of spatial coordinate transformation.

    Returns the inverse of the spatial transformation matrix computed
    by xtrans(E, r). This is more efficient than computing np.linalg.inv(xtrans(E, r)).

    The inverse transformation has the form:
        X⁻¹ = [ Eᵀ       0      ]
              [ r×·Eᵀ    Eᵀ     ]

    Args:
        e_rot: 3x3 rotation matrix from frame A to frame B
        r: 3x1 translation vector from A to B, expressed in frame A

    Returns:
        6x6 inverse spatial transformation matrix

    References:
        Featherstone, R. (2008). Rigid Body Dynamics Algorithms.
        Chapter 2: Spatial Vector Algebra

    Example:
        >>> e_rot = np.eye(3)
        >>> r = np.array([1, 0, 0])
        >>> x_transform = xtrans(e_rot, r)
        >>> x_inv = inv_xtrans(e_rot, r)
        >>> np.allclose(x_transform @ x_inv, np.eye(6))
        True
    """
    e_rot = np.asarray(e_rot)
    r = np.asarray(r).flatten()

    assert e_rot.shape == (3, 3), f"E must be 3x3 matrix, got shape {e_rot.shape}"
    assert r.shape == (3,), f"r must be 3x1 vector, got shape {r.shape}"

    e_t = e_rot.T
    r_skew = skew(r)

    return np.block([[e_t, np.zeros((3, 3))], [r_skew @ e_t, e_t]])
