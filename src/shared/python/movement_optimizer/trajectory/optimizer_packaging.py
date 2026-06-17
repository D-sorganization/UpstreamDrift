# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Result-packaging helpers for the trajectory optimiser.

These free functions handle the post-solve steps: expanding the solver's
flat solution vector into physical trajectories, checking feasibility, and
assembling the final :class:`OptimizationResult` dataclass.

Keeping them separate from the main optimiser class makes them independently
testable and keeps the class focused on orchestration.
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ..backend import PhysicsBackend
from ..constants import COM_FEASIBILITY_TOL_M, JOINT_FEASIBILITY_TOL_RAD
from .result import OptimizationResult


class _OptimizeResult(Protocol):
    """Structural protocol for scipy.optimize.OptimizeResult.

    Scipy's OptimizeResult is a dict-like object; the stubs expose it as
    ``object`` in some contexts.  Declaring a minimal Protocol avoids
    ``type: ignore[attr-defined]`` at every ``res.fun`` and ``res.x``
    access without importing scipy in type-check-only paths.
    """

    @property
    def fun(self) -> float:
        """Objective function value at the solution."""
        ...

    @property
    def x(self) -> NDArray:
        """Solution array."""
        ...


class _SplineCallable(Protocol):
    """Protocol for build_splines callable: x -> CubicSpline-like."""

    def __call__(self, x: NDArray) -> object:
        """Build splines from flat waypoint vector."""
        ...


class _TrajectoryCallable(Protocol):
    """Protocol for eval_trajectory callable: splines -> (q, qd, qdd, qddd)."""

    def __call__(self, splines: object) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Evaluate trajectory from splines."""
        ...


logger = logging.getLogger(__name__)

__all__ = [
    "build_result_object",
    "check_com_feasibility",
    "count_joint_limit_violations",
    "evaluate_solution",
]


def evaluate_solution(
    res: _OptimizeResult,
    dynamics: PhysicsBackend,
    exercise_type: str,
    bar_mass: float,
    build_splines_fn: _SplineCallable,
    eval_trajectory_fn: _TrajectoryCallable,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Expand solver output into physical trajectories and COM / bar paths.

    Parameters:
        res: scipy OptimizeResult (must have .x attribute).
        dynamics: physics backend providing inverse_dynamics_batch, com_x_batch,
            com_position, and bar_position.
        exercise_type: exercise identifier string.
        bar_mass: barbell mass in kg.
        build_splines_fn: callable(x) -> CubicSpline.
        eval_trajectory_fn: callable(splines) -> (q, qd, qdd, qddd).

    Returns:
        Tuple of (q, qd, qdd, torques, power, com_traj, bar_traj, com_x),
        each as float64 NDArray.
    """
    splines = build_splines_fn(res.x)
    q, qd, qdd, _ = eval_trajectory_fn(splines)

    torques = dynamics.inverse_dynamics_batch(q, qd, qdd)
    power = torques * qd

    n_pts = q.shape[0]
    com_x = dynamics.com_x_batch(q, exercise_type, bar_mass)
    com_traj = np.empty((n_pts, 2))
    bar_traj = np.empty((n_pts, 2))
    for n in range(n_pts):
        com_full = dynamics.com_position(q[n], exercise_type, bar_mass)
        com_traj[n, 0] = com_x[n]
        com_traj[n, 1] = com_full[1]
        bar_traj[n] = dynamics.bar_position(q[n], exercise_type)

    return q, qd, qdd, torques, power, com_traj, bar_traj, com_x


def check_com_feasibility(
    cost_finite: bool,
    com_x: NDArray[np.float64],
    exercise_type: str,
    inner_heel: float,
    inner_toe: float,
) -> bool:
    """Return True if the COM trajectory satisfies inner-BOS bounds.

    Bench press is exempt (lifter is lying down, COM constraint is inactive).
    Emits a warning when cost is finite but COM is out of bounds.

    Preconditions:
        com_x.ndim == 1
    """
    if exercise_type == "bench_press":
        return True
    in_bounds = bool(
        np.all(com_x >= inner_heel - COM_FEASIBILITY_TOL_M)
        and np.all(com_x <= inner_toe + COM_FEASIBILITY_TOL_M)
    )
    if cost_finite and not in_bounds:
        logger.warning(
            "Solution found but COM violated inner BOS: min=%.4f max=%.4f (bounds: [%.4f, %.4f])",
            com_x.min(),
            com_x.max(),
            inner_heel,
            inner_toe,
        )
    return in_bounds


def count_joint_limit_violations(
    q: NDArray[np.float64],
    q_bounds: NDArray[np.float64] | None,
) -> int:
    """Count (and warn about) trajectory points that violate joint limits.

    Spline overshoot between control points may push q outside q_bounds.
    A small tolerance (:data:`JOINT_FEASIBILITY_TOL_RAD`) absorbs benign
    numerical overshoot so only material violations are counted.
    Returns 0 when q_bounds is None.

    Preconditions:
        q.ndim == 2 and q.shape[1] == 3
        q_bounds is None or q_bounds.shape == (3, 2)
    """
    if q_bounds is None:
        return 0
    lower = q_bounds[:, 0] - JOINT_FEASIBILITY_TOL_RAD
    upper = q_bounds[:, 1] + JOINT_FEASIBILITY_TOL_RAD
    n_violations = int(np.sum((q < lower) | (q > upper)))
    if n_violations > 0:
        logger.warning(
            "Trajectory has %d point(s) violating joint limits "
            "(spline overshoot between control points).",
            n_violations,
        )
    return n_violations


def build_result_object(
    *,
    t_eval: NDArray[np.float64],
    res: _OptimizeResult,
    q: NDArray[np.float64],
    qd: NDArray[np.float64],
    qdd: NDArray[np.float64],
    torques: NDArray[np.float64],
    power: NDArray[np.float64],
    com_traj: NDArray[np.float64],
    bar_traj: NDArray[np.float64],
    com_x: NDArray[np.float64],
    success: bool,
    n_joint_limit_violations: int,
    elapsed: float,
    n_evals: int,
) -> OptimizationResult:
    """Assemble the final OptimizationResult dataclass from solver outputs.

    Preconditions:
        t_eval.ndim == 1 and len(t_eval) == q.shape[0]
        elapsed >= 0
    """
    cost_val = float(res.fun)
    com_h_range = float((np.max(com_x) - np.min(com_x)) * 100.0)
    return OptimizationResult(
        t=t_eval,
        q=q,
        qd=qd,
        qdd=qdd,
        torques=torques,
        power=power,
        com=com_traj,
        bar=bar_traj,
        success=success,
        cost=cost_val,
        com_horizontal_range_cm=com_h_range,
        elapsed_s=elapsed,
        n_evals=n_evals,
        n_joint_limit_violations=n_joint_limit_violations,
    )
