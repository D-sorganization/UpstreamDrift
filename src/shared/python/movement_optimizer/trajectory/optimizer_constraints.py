# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Constraint-vector builders for the trajectory optimiser.

These free functions build the inequality-constraint vectors consumed by
scipy.optimize.minimize (SLSQP).  Keeping them separate from the main
optimiser class makes them independently testable and keeps the class
focused on orchestration.

All returned arrays must be >= 0 at feasible points (SLSQP convention).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from ..backend import PhysicsBackend
from ..constants import BAR_KNEE_CLEARANCE_M
from ..models import BodyModel

# Type alias for the spline-building callable used throughout the optimizer.
# build_splines takes a flat waypoint vector and returns a CubicSpline object.
_BuildSplinesFn = Callable[[NDArray], object]

__all__ = [
    "bar_knee_clearance",
    "build_constraints",
    "com_constraint_values",
    "joint_limit_constraint_values",
]


def com_constraint_values(
    x: NDArray,
    build_splines_fn: _BuildSplinesFn,
    t_eval: NDArray,
    dynamics: PhysicsBackend,
    exercise_type: str,
    bar_mass: float,
    inner_heel: float,
    inner_toe: float,
) -> NDArray:
    """Return COM constraint violation vector for SLSQP.

    Returns array of length 2*n_eval:
        [0..n_eval-1]   = com_x - inner_heel  (must be >= 0)
        [n_eval..2*n-1] = inner_toe - com_x    (must be >= 0)

    Preconditions:
        x.ndim == 1
        inner_heel <= inner_toe

    Complexity:
        O(W * D + N * D) time and O(N) output memory, where ``W`` is waypoint
        controls, ``N`` evaluation samples, and ``D`` degrees of freedom.
    """
    splines = build_splines_fn(x)
    q = splines(t_eval)  # type: ignore[operator]  # CubicSpline is callable but typed as object; the returned object is always callable at runtime
    com_x = dynamics.com_x_batch(q, exercise_type, bar_mass)
    lower = com_x - inner_heel
    upper = inner_toe - com_x
    return np.concatenate([lower, upper])


def bar_knee_clearance(
    x: NDArray,
    build_splines_fn: _BuildSplinesFn,
    t_eval: NDArray,
    body: BodyModel,
) -> NDArray:
    """Bar must stay in front of the knees during pulling exercises.

    Returns array of length n_eval:
        bar_x - knee_x + margin  (must be >= 0)

    Active for deadlift, clean, and snatch exercises.

    Preconditions:
        x.ndim == 1

    Complexity:
        O(W * D + N * D) time and O(N) output memory for spline evaluation and
        vectorized clearance computation.
    """
    splines = build_splines_fn(x)
    q = splines(t_eval)  # type: ignore[operator]  # CubicSpline is callable but typed as object
    L = body.L
    # Using @ for matrix-vector multiplication is significantly faster than
    # calculating individual segment x-coordinates via element-wise multiplication and summing.
    # We want: bar_x - knee_x = (shoulder_x) - knee_x
    # Since shoulder_x = knee_x + L[1]*sin(q1) + L[2]*sin(q2),
    # bar_x - knee_x = L[1]*sin(q1) + L[2]*sin(q2) = sin(q[:, 1:3]) @ L[1:3]
    bar_minus_knee_x = np.sin(q[:, 1:3]) @ L[1:3]
    return bar_minus_knee_x + BAR_KNEE_CLEARANCE_M


def joint_limit_constraint_values(
    x: NDArray,
    build_splines_fn: _BuildSplinesFn,
    t_eval: NDArray,
    q_bounds: NDArray,
) -> NDArray:
    """Return joint limit constraint violation vector for SLSQP.

    Ensures all evaluated trajectory points stay within joint limits,
    preventing spline overshoot between control points.

    Returns array of length 2*n_eval*n_dof (all must be >= 0).

    Preconditions:
        x.ndim == 1
        q_bounds.shape == (n_dof, 2)

    Complexity:
        O(W * D + N * D) time and O(N * D) output memory.
    """
    splines = build_splines_fn(x)
    q = splines(t_eval)  # type: ignore[operator]  # CubicSpline is callable but typed as object
    lower = q - q_bounds[:, 0]
    upper = q_bounds[:, 1] - q
    return np.concatenate([lower.flatten(), upper.flatten()])


def build_constraints(
    exercise_type: str,
    build_splines_fn: _BuildSplinesFn,
    t_eval: NDArray,
    dynamics: PhysicsBackend,
    bar_mass: float,
    inner_heel: float,
    inner_toe: float,
    q_bounds: NDArray,
    body: BodyModel,
) -> list[dict]:
    """Build the SLSQP inequality constraint list.

    Bench press exercises omit COM/balance constraints because the lifter
    is lying on a bench, not standing.

    Pulling exercises (deadlift, clean, snatch) add bar-knee clearance.

    Preconditions:
        exercise_type is a known exercise string.
        inner_heel <= inner_toe.

    Complexity:
        O(1) time and memory; the returned callables do the O(N * D) work when
        SLSQP evaluates them.
    """
    constraints: list[dict] = [
        {
            "type": "ineq",
            "fun": joint_limit_constraint_values,
            "args": (build_splines_fn, t_eval, q_bounds),
        },
    ]

    if exercise_type != "bench_press":
        constraints.append(
            {
                "type": "ineq",
                "fun": com_constraint_values,
                "args": (
                    build_splines_fn,
                    t_eval,
                    dynamics,
                    exercise_type,
                    bar_mass,
                    inner_heel,
                    inner_toe,
                ),
            }
        )

        pulling_exercises = {"deadlift", "clean", "snatch"}
        if exercise_type in pulling_exercises:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": bar_knee_clearance,
                    "args": (build_splines_fn, t_eval, body),
                }
            )

    return constraints
