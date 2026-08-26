"""Trajectory-varying finite-window control-authority mechanics.

These helpers construct exact-discrete local Jacobians and scaled reachability
Gramians along a declared trajectory. They do not establish global nonlinear
reachability, bounded-control feasibility, human actuator capacity, controller
superiority, passive torque, robustness, or coaching strategy.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from scripts.research.proximal_distal_energy.phase_event_stability import (
    StateScales,
    registered_step,
)
from src.shared.python.simulation_backends import GolfModelParams, make_backend

FloatArray: TypeAlias = npt.NDArray[np.float64]


def _finite_vector(name: str, value: npt.ArrayLike, *, size: int) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {size} finite values")
    return array


def _positive_vector(name: str, value: npt.ArrayLike, *, size: int) -> FloatArray:
    array = _finite_vector(name, value, size=size)
    if np.any(array <= 0.0):
        raise ValueError(f"{name} must contain {size} finite positive values")
    return array


def _readonly(value: npt.ArrayLike) -> FloatArray:
    array = np.asarray(value, dtype=float).copy()
    if not np.all(np.isfinite(array)):
        raise ValueError("result contains non-finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class ControlScales:
    """Positive characteristic scales for shoulder and wrist torque."""

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in self.values)
        if len(values) != 2 or any(
            not math.isfinite(value) or value <= 0.0 for value in values
        ):
            raise ValueError("control scales must contain two finite positive values")
        object.__setattr__(self, "values", values)

    @property
    def array(self) -> FloatArray:
        """Return a defensive NumPy representation."""

        return np.asarray(self.values, dtype=float)


@dataclass(frozen=True, slots=True)
class StepLinearization:
    """Physical and scaled Jacobians of one exact registered RK4 step."""

    state_matrix: FloatArray
    input_matrix: FloatArray
    scaled_state_matrix: FloatArray
    scaled_sample_input_matrix: FloatArray
    scaled_energy_input_matrix: FloatArray

    def __post_init__(self) -> None:
        expected = {
            "state_matrix": (4, 4),
            "input_matrix": (4, 2),
            "scaled_state_matrix": (4, 4),
            "scaled_sample_input_matrix": (4, 2),
            "scaled_energy_input_matrix": (4, 2),
        }
        for name, shape in expected.items():
            array = np.asarray(getattr(self, name), dtype=float)
            if array.shape != shape or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be finite with shape {shape}")
            object.__setattr__(self, name, _readonly(array))


@dataclass(frozen=True, slots=True)
class EventConditionedGramian:
    """Arrival-state projection and explicit event-tangent authority."""

    status: str
    transversality_per_s: float
    projection: FloatArray | None
    tangent_basis: FloatArray | None
    tangent_gramian: FloatArray | None


def _central_jacobian(
    function: Callable[[FloatArray], npt.ArrayLike],
    point: FloatArray,
    steps: FloatArray,
    *,
    output_size: int,
) -> FloatArray:
    jacobian = np.empty((output_size, point.size), dtype=float)
    for column, step in enumerate(steps):
        upper = point.copy()
        lower = point.copy()
        upper[column] += step
        lower[column] -= step
        upper_value = _finite_vector(
            "step output",
            function(upper),
            size=output_size,
        )
        lower_value = _finite_vector(
            "step output",
            function(lower),
            size=output_size,
        )
        jacobian[:, column] = (upper_value - lower_value) / (2.0 * step)
    return jacobian


def step_linearization(
    *,
    params: GolfModelParams,
    state: npt.ArrayLike,
    control: npt.ArrayLike,
    time_s: float,
    dt_s: float,
    state_steps: npt.ArrayLike,
    control_steps: npt.ArrayLike,
    state_scales: StateScales,
    control_scales: ControlScales,
) -> StepLinearization:
    """Differentiate the exact RK4 step in state and input coordinates.

    ``scaled_sample_input_matrix`` maps a dimensionless, piecewise-constant
    control sample to scaled state. ``scaled_energy_input_matrix`` divides that
    matrix by ``sqrt(dt_s)`` so a unit discrete input has the same quadratic
    cost as a unit-energy piecewise-constant continuous input.
    """

    vector = _finite_vector("state", state, size=4).copy()
    command = _finite_vector("control", control, size=2).copy()
    x_steps = _positive_vector("state_steps", state_steps, size=4)
    u_steps = _positive_vector("control_steps", control_steps, size=2)
    if not math.isfinite(time_s):
        raise ValueError("time_s must be finite")
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    backend = make_backend("ode", params, dt=dt_s)

    def state_map(candidate: FloatArray) -> FloatArray:
        return registered_step(
            backend,
            candidate,
            command,
            time_s=time_s,
            dt_s=dt_s,
        )

    def input_map(candidate: FloatArray) -> FloatArray:
        return registered_step(
            backend,
            vector,
            candidate,
            time_s=time_s,
            dt_s=dt_s,
        )

    state_matrix = _central_jacobian(
        state_map, vector, x_steps, output_size=vector.size
    )
    input_matrix = _central_jacobian(
        input_map, command, u_steps, output_size=vector.size
    )
    x_scale = state_scales.array
    u_scale = control_scales.array
    scaled_state = state_matrix * x_scale[np.newaxis, :] / x_scale[:, np.newaxis]
    scaled_sample_input = input_matrix * u_scale[np.newaxis, :] / x_scale[:, np.newaxis]
    return StepLinearization(
        state_matrix=state_matrix,
        input_matrix=input_matrix,
        scaled_state_matrix=scaled_state,
        scaled_sample_input_matrix=scaled_sample_input,
        scaled_energy_input_matrix=scaled_sample_input / math.sqrt(dt_s),
    )


def _trajectory_matrices(
    state_matrices: npt.ArrayLike,
    input_matrices: npt.ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    state = np.asarray(state_matrices, dtype=float)
    control = np.asarray(input_matrices, dtype=float)
    if (
        state.ndim != 3
        or state.shape[0] == 0
        or state.shape[1] == 0
        or state.shape[1] != state.shape[2]
    ):
        raise ValueError("state_matrices must be a nonempty stack of square matrices")
    if control.ndim != 3 or control.shape[0] != state.shape[0]:
        raise ValueError("input_matrices must have the same step count")
    if control.shape[1] != state.shape[1] or control.shape[2] == 0:
        raise ValueError("input_matrices must match the state dimension")
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(control)):
        raise ValueError("state and input matrices must be finite")
    return state, control


def reachability_history(
    state_matrices: npt.ArrayLike,
    input_matrices: npt.ArrayLike,
    *,
    channel_mask: npt.ArrayLike | None = None,
) -> FloatArray:
    """Propagate the discrete LTV reachability Gramian from zero authority."""

    state, control = _trajectory_matrices(state_matrices, input_matrices)
    mask = (
        np.ones(control.shape[2], dtype=float)
        if channel_mask is None
        else _finite_vector("channel_mask", channel_mask, size=control.shape[2])
    )
    if np.any((mask != 0.0) & (mask != 1.0)):
        raise ValueError("channel_mask entries must be zero or one")
    masked = control * mask[np.newaxis, np.newaxis, :]
    history = np.zeros((state.shape[0] + 1, state.shape[1], state.shape[1]))
    for index, (state_matrix, input_matrix) in enumerate(
        zip(state, masked, strict=True)
    ):
        propagated = state_matrix @ history[index] @ state_matrix.T
        injected = input_matrix @ input_matrix.T
        history[index + 1] = 0.5 * (propagated + injected + (propagated + injected).T)
    return _readonly(history)


def frozen_local_gramian(
    state_matrix: npt.ArrayLike,
    input_matrix: npt.ArrayLike,
    *,
    step_count: int,
) -> FloatArray:
    """Repeat one frozen discrete linearization as an explicit countermodel."""

    if (
        isinstance(step_count, bool)
        or not isinstance(step_count, int)
        or step_count < 1
    ):
        raise ValueError("step_count must be a positive integer")
    state = np.asarray(state_matrix, dtype=float)
    control = np.asarray(input_matrix, dtype=float)
    history = reachability_history(
        np.repeat(state[np.newaxis, :, :], step_count, axis=0),
        np.repeat(control[np.newaxis, :, :], step_count, axis=0),
    )
    return _readonly(history[-1])


def event_conditioned_gramian(
    gramian: npt.ArrayLike,
    *,
    event_flow: npt.ArrayLike,
    guard_gradient: npt.ArrayLike,
    transversality_threshold: float,
) -> EventConditionedGramian:
    """Project arrival authority into an orthonormal event-tangent basis."""

    matrix = np.asarray(gramian, dtype=float)
    if (
        matrix.ndim != 2
        or matrix.shape[0] == 0
        or matrix.shape[0] != matrix.shape[1]
        or not np.all(np.isfinite(matrix))
    ):
        raise ValueError("gramian must be a nonempty finite square matrix")
    if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-14):
        raise ValueError("gramian must be symmetric")
    dimension = matrix.shape[0]
    flow = _finite_vector("event_flow", event_flow, size=dimension)
    gradient = _finite_vector("guard_gradient", guard_gradient, size=dimension)
    if np.linalg.norm(gradient) == 0.0:
        raise ValueError("guard_gradient must be nonzero")
    if not math.isfinite(transversality_threshold) or transversality_threshold <= 0.0:
        raise ValueError("transversality_threshold must be finite and positive")
    denominator = float(gradient @ flow)
    if abs(denominator) <= transversality_threshold:
        return EventConditionedGramian(
            status="near_grazing",
            transversality_per_s=denominator,
            projection=None,
            tangent_basis=None,
            tangent_gramian=None,
        )
    projection = np.eye(dimension) - np.outer(flow, gradient) / denominator
    _, _, right = np.linalg.svd(gradient.reshape(1, -1), full_matrices=True)
    tangent_basis = right[1:].T
    projected = projection @ matrix @ projection.T
    tangent = tangent_basis.T @ projected @ tangent_basis
    tangent = 0.5 * (tangent + tangent.T)
    return EventConditionedGramian(
        status="transverse",
        transversality_per_s=denominator,
        projection=_readonly(projection),
        tangent_basis=_readonly(tangent_basis),
        tangent_gramian=_readonly(tangent),
    )


__all__ = [
    "ControlScales",
    "EventConditionedGramian",
    "StepLinearization",
    "event_conditioned_gramian",
    "frozen_local_gramian",
    "reachability_history",
    "step_linearization",
]
