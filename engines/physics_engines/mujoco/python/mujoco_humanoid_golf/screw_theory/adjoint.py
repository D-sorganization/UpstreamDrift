"""
Adjoint transformations for twists and wrenches.
"""

import numpy as np
from mujoco_humanoid_golf.spatial_algebra.spatial_vectors import skew


def adjoint_transform(T: np.ndarray) -> np.ndarray:
    """
    Compute adjoint transformation matrix.

    Computes the 6x6 adjoint transformation matrix from a 4x4 homogeneous
    transformation matrix T.

    The adjoint transformation maps twists (spatial velocities) from one
    frame to another:
        V_a = Ad_{T_ab} * V_b

    The adjoint matrix has the form:
        Ad = [R,        0    ]
             [p_hat*R,  R    ]

    where R is the 3x3 rotation matrix and p_hat is the skew-symmetric
    matrix of the position vector p.

    This is equivalent to Featherstone's spatial transformation matrix,
    assuming spatial vectors are ordered [angular; linear].

    Args:
        T: 4x4 homogeneous transformation matrix

    Returns:
        6x6 adjoint transformation matrix

    Properties:
        - Ad(T1 @ T2) = Ad(T1) @ Ad(T2)
        - Ad(inv(T)) = inv(Ad(T))
        - For transforming wrenches: use Ad.T

    References:
        Murray, Li, Sastry (1994). A Mathematical Introduction to Robotic Manipulation.
        Lynch, Park (2017). Modern Robotics: Mechanics, Planning, and Control.
        Chapter 3: Rigid-Body Motions, Section 3.3.3

    Example:
        >>> # Create transformation: 90° rotation about z, translate [1;0;0]
        >>> T = np.array([[0, -1, 0, 1],
        ...               [1,  0, 0, 0],
        ...               [0,  0, 1, 0],
        ...               [0,  0, 0, 1]])
        >>> Ad = adjoint_transform(T)
        >>> Ad.shape
        (6, 6)

        >>> # Transform a twist
        >>> V_b = np.array([0, 0, 1, 0, 0, 0])  # Angular velocity about z
        >>> V_a = Ad @ V_b
    """
    t_transform = np.asarray(T)
    assert t_transform.shape == (4, 4), f"T must be 4x4, got shape {t_transform.shape}"

    # Extract rotation and position
    r_rot = t_transform[:3, :3]
    p = t_transform[:3, 3]

    # Create skew-symmetric matrix of position
    p_hat = skew(p)

    # Optimization: Direct assignment is faster than np.block
    # Ad = [[r_rot,      0],
    #       [p_hat @ r_rot, r_rot]]
    res = np.zeros((6, 6), dtype=t_transform.dtype)
    res[:3, :3] = r_rot
    res[3:, 3:] = r_rot
    res[3:, :3] = p_hat @ r_rot

    return res
