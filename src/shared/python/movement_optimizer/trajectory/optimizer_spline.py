# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Spline construction and trajectory evaluation helpers.

Separating these from :class:`TrajectoryOptimizer` gives them a single
responsibility (spline math) and makes them independently testable.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

__all__ = [
    "build_splines",
    "eval_trajectory",
]


def build_splines(
    x: NDArray,
    q_start: NDArray,
    q_end: NDArray,
    q_via: NDArray | None,
    t_ctrl: NDArray,
    n_waypoints: int,
    n_dof: int,
) -> CubicSpline:
    """Build a multidimensional clamped cubic spline from the flat optimisation vector *x*.

    Parameters:
        x: Flat waypoint vector of length n_waypoints * n_dof.
        q_start: Start configuration, shape (n_dof,).
        q_end: End configuration, shape (n_dof,).
        q_via: Optional via-point configuration, shape (n_dof,).
        t_ctrl: Control-point time grid, length n_waypoints + 2 [+ 1 if q_via].
        n_waypoints: Number of interior waypoints.
        n_dof: Degrees of freedom.

    Returns:
        CubicSpline evaluating to shape (..., n_dof).

    Preconditions:
        len(x) == n_waypoints * n_dof
        len(t_ctrl) == n_waypoints + 2 (+ 1 if q_via is not None)
    """
    wp = x.reshape(n_waypoints, n_dof)

    if q_via is not None:
        n_half = n_waypoints // 2
        q_all = np.vstack([q_start, wp[:n_half], q_via, wp[n_half:], q_end])
    else:
        q_all = np.vstack([q_start, wp, q_end])

    return CubicSpline(t_ctrl, q_all, bc_type="clamped")


def eval_trajectory(
    splines: CubicSpline,
    t_eval: NDArray,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Evaluate position, velocity, acceleration, and jerk at the eval grid.

    Parameters:
        splines: Multidimensional CubicSpline object.
        t_eval: Evaluation time grid, shape (n_eval,).

    Returns:
        Tuple (q, qd, qdd, qddd), each of shape (n_eval, n_dof).

    Preconditions:
        t_eval.ndim == 1
    """
    q = splines(t_eval)
    qd = splines(t_eval, 1)
    qdd = splines(t_eval, 2)
    qddd = splines(t_eval, 3)
    return q, qd, qdd, qddd
