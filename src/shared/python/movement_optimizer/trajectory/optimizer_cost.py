# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Pure cost-term functions for the trajectory optimiser.

Each function computes one additive term of the total optimisation cost.
They are free functions (no class dependency) so they are trivially testable
in isolation.  The :class:`TrajectoryOptimizer` calls these with its own
weights and time-step.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..constants import TV_RATE_WEIGHT_RATIO
from .tuning import ENDPOINT_ACCEL_WEIGHT_RATIO

__all__ = [
    "compute_balance_cost",
    "compute_endpoint_damping_cost",
    "compute_jerk_cost",
    "compute_torque_cost",
    "compute_torque_rate_cost",
]


def compute_torque_cost(torques: NDArray, dt: float) -> float:
    """Integral of squared joint torques over the trajectory.

    Preconditions:
        torques.ndim == 2
        dt > 0

    Complexity:
        O(N * D) time and O(1) extra memory for ``N`` samples and ``D`` joints.
    """
    # np.vdot is significantly faster than np.sum(x**2)
    return float(np.vdot(torques, torques)) * dt


def compute_jerk_cost(qddd: NDArray, dt: float, weight: float) -> float:
    """Smoothness: integral of squared jerk (third derivative of angle).

    Preconditions:
        qddd.ndim == 2
        dt > 0
        weight >= 0

    Complexity:
        O(N * D) time and O(1) extra memory for ``N`` samples and ``D`` joints.
    """
    # np.vdot is significantly faster than np.sum(x**2)
    return weight * float(np.vdot(qddd, qddd)) * dt


def compute_torque_rate_cost(torques: NDArray, dt: float, weight: float) -> float:
    """Penalise rapid torque changes using L2 + total-variation regularization.

    Preconditions:
        torques.ndim == 2 and torques.shape[0] >= 2
        dt > 0
        weight >= 0

    Complexity:
        O(N * D) time and O(N * D) memory because adjacent torque differences
        are materialized.
    """
    dtau = np.diff(torques, axis=0) / dt
    # np.vdot is significantly faster than np.sum(x**2)
    l2_cost = float(np.vdot(dtau, dtau)) * dt
    tv_cost = float(np.sum(np.abs(dtau))) * dt * TV_RATE_WEIGHT_RATIO
    return weight * (l2_cost + tv_cost)


def compute_endpoint_damping_cost(
    qd: NDArray,
    qdd: NDArray,
    dt: float,
    weight: float,
    n_damp: int,
    damp_weights: NDArray,
) -> float:
    """Extra penalty on motion near trajectory endpoints.

    Preconditions:
        qd.ndim == qdd.ndim == 2
        n_damp >= 1 and n_damp <= qd.shape[0]
        len(damp_weights) == n_damp
        dt > 0
        weight >= 0

    Complexity:
        O(K * D) time and memory for ``K = n_damp`` endpoint samples and ``D``
        joints.
    """
    nd = n_damp
    w = damp_weights
    w_end = w[::-1]

    # np.vdot is significantly faster than np.sum(x**2, axis=1) combined with np.dot
    w_col = w[:, np.newaxis]
    w_end_col = w_end[:, np.newaxis]

    a = ENDPOINT_ACCEL_WEIGHT_RATIO
    cost = (
        np.vdot(w_col * qd[:nd], qd[:nd])
        + np.vdot(w_end_col * qd[-nd:], qd[-nd:])
        + a * np.vdot(w_col * qdd[:nd], qdd[:nd])
        + a * np.vdot(w_end_col * qdd[-nd:], qdd[-nd:])
    )
    return weight * float(cost) * dt


def compute_balance_cost(com_x: NDArray, center: float, dt: float, weight: float) -> float:
    """Soft centering preference — penalise COM deviation from the inner BOS center.

    Preconditions:
        com_x.ndim == 1
        dt > 0
        weight >= 0

    Complexity:
        O(N) time and O(N) memory for the centered COM vector.
    """
    delta = com_x - center
    # np.vdot is significantly faster than np.sum(x**2)
    return weight * float(np.vdot(delta, delta)) * dt
