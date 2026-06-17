# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Vectorised (batch) inverse-dynamics helpers for the 3-link Lagrangian chain.

These are pure NumPy functions that accept the pre-computed scalar coefficients
from ``LagrangianDynamics`` rather than a ``self`` reference.  Keeping them
here separates the *batch computation* concern from the class that owns the
model parameters.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def batch_inertia_torques(
    qdd: NDArray,
    c01: NDArray,
    c02: NDArray,
    c12: NDArray,
    *,
    M00: float,
    M11: float,
    M22: float,
    a01: float,
    a02: float,
    a12: float,
) -> NDArray:
    """Compute the inertia (mass-matrix) contribution to batch torques.

    Parameters:
        qdd:  Angular accelerations, shape (N, 3).
        c01:  cos(q[:,0] - q[:,1]), shape (N,).
        c02:  cos(q[:,0] - q[:,2]), shape (N,).
        c12:  cos(q[:,1] - q[:,2]), shape (N,).
        M00, M11, M22: Diagonal mass-matrix constants.
        a01, a02, a12: Off-diagonal coupling coefficients.

    Returns:
        Inertia torque contribution, shape (N, 3).

    Complexity:
        O(N) time and O(N) output memory for ``N`` trajectory samples.
    """
    n = qdd.shape[0]
    tau = np.empty((n, 3))
    tau[:, 0] = M00 * qdd[:, 0] + a01 * c01 * qdd[:, 1] + a02 * c02 * qdd[:, 2]
    tau[:, 1] = a01 * c01 * qdd[:, 0] + M11 * qdd[:, 1] + a12 * c12 * qdd[:, 2]
    tau[:, 2] = a02 * c02 * qdd[:, 0] + a12 * c12 * qdd[:, 1] + M22 * qdd[:, 2]
    return tau


def batch_coriolis_torques(
    qd: NDArray,
    s01: NDArray,
    s02: NDArray,
    s12: NDArray,
    *,
    a01: float,
    a02: float,
    a12: float,
) -> NDArray:
    """Compute the centrifugal/Coriolis contribution to batch torques.

    Parameters:
        qd:  Angular velocities, shape (N, 3).
        s01: sin(q[:,0] - q[:,1]), shape (N,).
        s02: sin(q[:,0] - q[:,2]), shape (N,).
        s12: sin(q[:,1] - q[:,2]), shape (N,).
        a01, a02, a12: Off-diagonal coupling coefficients.

    Returns:
        Coriolis torque contribution, shape (N, 3).

    Complexity:
        O(N) time and O(N) output memory for ``N`` trajectory samples.
    """
    n = qd.shape[0]
    # Performance optimization:
    # 1. np.empty avoids zero-initialization overhead since array is fully overwritten.
    # 2. Explicit multiplication (x * x) is significantly faster than exponentiation (x ** 2) in NumPy.
    tau = np.empty((n, 3))
    qd_0 = qd[:, 0]
    qd_1 = qd[:, 1]
    qd_2 = qd[:, 2]
    qd2_0 = qd_0 * qd_0
    qd2_1 = qd_1 * qd_1
    qd2_2 = qd_2 * qd_2
    tau[:, 0] = a01 * s01 * qd2_1 + a02 * s02 * qd2_2
    tau[:, 1] = -a01 * s01 * qd2_0 + a12 * s12 * qd2_2
    tau[:, 2] = -a02 * s02 * qd2_0 - a12 * s12 * qd2_1
    return tau


def batch_gravity_torques(
    q: NDArray,
    *,
    g0: float,
    g1: float,
    g2: float,
    supine: bool = False,
) -> NDArray:
    """Compute the gravity contribution to batch torques.

    Parameters:
        q: Joint angles, shape (N, 3).
        g0, g1, g2: Pre-computed gravity torque coefficients.
        supine: If True, gravity acts perpendicular to the chain axis
            (lifter is lying down) — uses cos(q) instead of sin(q).

    Returns:
        Gravity torque contribution, shape (N, 3).

    Complexity:
        O(N) time and O(N) output memory for ``N`` trajectory samples.
    """
    sq = np.cos(q) if supine else np.sin(q)
    # Performance optimization: Use NumPy array broadcasting instead of manually allocating
    # and populating an intermediate output array via column assignment.
    return sq * np.array([g0, g1, g2])


def numpy_inverse_dynamics_batch(
    q: NDArray,
    qd: NDArray,
    qdd: NDArray,
    *,
    M00: float,
    M11: float,
    M22: float,
    a01: float,
    a02: float,
    a12: float,
    g0: float,
    g1: float,
    g2: float,
    supine: bool = False,
) -> NDArray:
    """NumPy batch inverse dynamics for a 3-link Lagrangian chain.

    Fuses inertia, Coriolis, and gravity contributions into a single
    pre-allocated output array to minimise intermediate allocations.

    Parameters:
        q, qd, qdd: Joint angles/velocities/accelerations, shape (N, 3).
        M00, M11, M22: Diagonal mass-matrix constants.
        a01, a02, a12: Off-diagonal coupling coefficients.
        g0, g1, g2: Gravity torque coefficients.
        supine: If True uses cos(q) for gravity (bench-press orientation).

    Returns:
        torques: shape (N, 3).

    Complexity:
        O(N) time and O(N) output memory for ``N`` trajectory samples.  The
        chain has a fixed 3 DOF, so vectorized column operations scale linearly
        with sample count.
    """
    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]

    d01 = q0 - q1
    d02 = q0 - q2
    d12 = q1 - q2

    tau = np.empty((q.shape[0], 3))

    c01 = np.cos(d01)
    c02 = np.cos(d02)
    c12 = np.cos(d12)

    s01 = np.sin(d01)
    s02 = np.sin(d02)
    s12 = np.sin(d12)

    # Performance optimization: Explicit array multiplication (x * x) is faster than exponentiation (x ** 2).
    qd_0 = qd[:, 0]
    qd_1 = qd[:, 1]
    qd_2 = qd[:, 2]
    qd2_0 = qd_0 * qd_0
    qd2_1 = qd_1 * qd_1
    qd2_2 = qd_2 * qd_2

    qdd_0 = qdd[:, 0]
    qdd_1 = qdd[:, 1]
    qdd_2 = qdd[:, 2]

    a01_c01 = a01 * c01
    a02_c02 = a02 * c02
    a12_c12 = a12 * c12

    a01_s01 = a01 * s01
    a02_s02 = a02 * s02
    a12_s12 = a12 * s12

    sq0 = np.cos(q0) if supine else np.sin(q0)
    sq1 = np.cos(q1) if supine else np.sin(q1)
    sq2 = np.cos(q2) if supine else np.sin(q2)

    tau[:, 0] = (
        M00 * qdd_0
        + a01_c01 * qdd_1
        + a02_c02 * qdd_2
        + a01_s01 * qd2_1
        + a02_s02 * qd2_2
        + g0 * sq0
    )
    tau[:, 1] = (
        a01_c01 * qdd_0
        + M11 * qdd_1
        + a12_c12 * qdd_2
        - a01_s01 * qd2_0
        + a12_s12 * qd2_2
        + g1 * sq1
    )
    tau[:, 2] = (
        a02_c02 * qdd_0
        + a12_c12 * qdd_1
        + M22 * qdd_2
        - a02_s02 * qd2_0
        - a12_s12 * qd2_1
        + g2 * sq2
    )

    return tau
