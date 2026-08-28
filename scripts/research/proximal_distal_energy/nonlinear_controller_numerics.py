"""Numerical contracts for manufactured nonlinear-controller qualification."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]
Dynamics: TypeAlias = Callable[[FloatArray, FloatArray], FloatArray]


@dataclass(frozen=True, slots=True)
class BoxBounds:
    """Per-control lower and upper bounds enforced during optimization."""

    lower: FloatArray
    upper: FloatArray


@dataclass(frozen=True, slots=True)
class QuadraticTrackingCost:
    """Time-invariant quadratic tracking objective."""

    state_weight: FloatArray
    control_weight: FloatArray
    terminal_weight: FloatArray
    reference_state: FloatArray
    reference_control: FloatArray


@dataclass(frozen=True, slots=True)
class SolverResult:
    """Typed solver outcome with no fabricated trajectory on failure."""

    success: bool
    status: str
    states: FloatArray | None
    controls: FloatArray | None
    cost: float | None
    accepted_costs: tuple[float, ...]
    iterations: int


class DynamicsFailure(RuntimeError):
    """Signal a non-finite or shape-invalid pure-dynamics output."""


def manufactured_step(state: FloatArray, control: FloatArray) -> FloatArray:
    """Advance a damped nonlinear pendulum fixture by one explicit step."""
    state = finite_vector("state", state, 2)
    control = finite_vector("control", control, 1)
    step_s = 0.04
    angle, rate = state
    acceleration = -math.sin(angle) - 0.15 * rate + control[0]
    return np.array([angle + step_s * rate, rate + step_s * acceleration])


def central_dynamics_jacobians(
    dynamics: Dynamics,
    state: npt.ArrayLike,
    control: npt.ArrayLike,
    *,
    state_steps: npt.ArrayLike,
    control_steps: npt.ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """Return reset-safe central Jacobians of a pure discrete dynamics map."""
    state_vector = finite_vector("state", state)
    control_vector = finite_vector("control", control)
    state_delta = positive_steps("state_steps", state_steps, state_vector.size)
    control_delta = positive_steps("control_steps", control_steps, control_vector.size)
    state_map = _central_columns(
        dynamics, state_vector, control_vector, state_delta, True
    )
    control_map = _central_columns(
        dynamics, state_vector, control_vector, control_delta, False
    )
    return state_map, control_map


def validated_problem(
    initial_state: npt.ArrayLike,
    horizon: int,
    cost: QuadraticTrackingCost,
    bounds: BoxBounds,
    initial_controls: npt.ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Validate and normalize one finite-horizon box-constrained problem."""
    state = finite_vector("initial_state", initial_state)
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    controls = np.asarray(initial_controls, dtype=float)
    valid_controls = (
        controls.ndim == 2
        and controls.shape[0] == horizon
        and np.all(np.isfinite(controls))
    )
    if not valid_controls:
        raise ValueError("initial_controls must be finite with one row per step")
    lower = finite_vector("bounds.lower", bounds.lower, controls.shape[1])
    upper = finite_vector("bounds.upper", bounds.upper, controls.shape[1])
    if np.any(lower >= upper) or np.any(controls < lower) or np.any(controls > upper):
        raise ValueError("initial controls and bound ordering must be valid")
    validate_cost(cost, state.size, controls.shape[1])
    return state, controls.copy(), lower, upper


def validate_cost(
    cost: QuadraticTrackingCost, state_size: int, control_size: int
) -> None:
    """Require finite symmetric positive-semidefinite quadratic weights."""
    entries = (
        ("state_weight", cost.state_weight, state_size),
        ("terminal_weight", cost.terminal_weight, state_size),
        ("control_weight", cost.control_weight, control_size),
    )
    for name, value, size in entries:
        matrix = np.asarray(value, dtype=float)
        if matrix.shape != (size, size) or not np.all(np.isfinite(matrix)):
            raise ValueError(f"{name} must be finite with shape {(size, size)}")
        symmetric = np.allclose(matrix, matrix.T)
        positive_semidefinite = np.all(np.linalg.eigvalsh(matrix) >= 0.0)
        if not symmetric or not positive_semidefinite:
            raise ValueError(f"{name} must be symmetric positive semidefinite")
    finite_vector("reference_state", cost.reference_state, state_size)
    finite_vector("reference_control", cost.reference_control, control_size)


def rollout(
    dynamics: Dynamics, initial_state: FloatArray, controls: FloatArray
) -> FloatArray:
    """Roll out pure dynamics and fail if any returned state is invalid."""
    states = np.empty((controls.shape[0] + 1, initial_state.size))
    states[0] = initial_state
    for index, control in enumerate(controls):
        states[index + 1] = checked_step(dynamics, states[index], control)
    return states


def checked_step(
    dynamics: Dynamics, state: FloatArray, control: FloatArray
) -> FloatArray:
    """Evaluate pure dynamics while enforcing shape and finiteness."""
    candidate = np.asarray(dynamics(state.copy(), control.copy()), dtype=float)
    if candidate.shape != state.shape or not np.all(np.isfinite(candidate)):
        raise DynamicsFailure("dynamics output must be finite with state shape")
    return candidate


def trajectory_cost(
    states: FloatArray, controls: FloatArray, cost: QuadraticTrackingCost
) -> float:
    """Evaluate the registered finite-horizon quadratic objective."""
    state_error = states[:-1] - cost.reference_state
    control_error = controls - cost.reference_control
    terminal_error = states[-1] - cost.reference_state
    running_state = np.einsum("ni,ij,nj->", state_error, cost.state_weight, state_error)
    running_control = np.einsum(
        "ni,ij,nj->", control_error, cost.control_weight, control_error
    )
    terminal = terminal_error @ cost.terminal_weight @ terminal_error
    return float(running_state + running_control + terminal)


def monotonic(values: npt.ArrayLike) -> bool:
    """Return whether at least two accepted costs are nonincreasing."""
    array = np.asarray(values, dtype=float)
    return bool(array.size >= 2 and np.all(np.diff(array) <= 1.0e-10))


def finite_vector(
    name: str, value: npt.ArrayLike, size: int | None = None
) -> FloatArray:
    """Normalize a finite one-dimensional vector with optional exact size."""
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.size == 0 or (size is not None and vector.size != size):
        raise ValueError(f"{name} has invalid size")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite")
    return vector


def positive_steps(name: str, value: npt.ArrayLike, size: int) -> FloatArray:
    """Normalize finite-difference steps and require strict positivity."""
    steps = finite_vector(name, value, size)
    if np.any(steps <= 0.0):
        raise ValueError(f"{name} must be positive")
    return steps


def _central_columns(
    dynamics: Dynamics,
    state: FloatArray,
    control: FloatArray,
    steps: FloatArray,
    vary_state: bool,
) -> FloatArray:
    columns = np.empty((state.size, steps.size))
    for index, step in enumerate(steps):
        plus_state, minus_state = state.copy(), state.copy()
        plus_control, minus_control = control.copy(), control.copy()
        if vary_state:
            plus_state[index] += step
            minus_state[index] -= step
        else:
            plus_control[index] += step
            minus_control[index] -= step
        numerator = checked_step(dynamics, plus_state, plus_control)
        numerator -= checked_step(dynamics, minus_state, minus_control)
        columns[:, index] = numerator / (2.0 * step)
    return columns
