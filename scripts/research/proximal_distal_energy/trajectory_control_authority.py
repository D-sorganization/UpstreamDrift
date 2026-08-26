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
    rollout_state_history,
    state_derivative,
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


@dataclass(frozen=True, slots=True)
class TrajectoryLinearization:
    """Nominal trajectory and exact-discrete Jacobians at every step."""

    time_s: FloatArray
    state: FloatArray
    controls: FloatArray
    state_matrices: FloatArray
    input_matrices: FloatArray
    scaled_state_matrices: FloatArray
    scaled_sample_input_matrices: FloatArray
    scaled_energy_input_matrices: FloatArray

    def __post_init__(self) -> None:
        controls = np.asarray(self.controls, dtype=float)
        step_count = controls.shape[0] if controls.ndim == 2 else -1
        expected = {
            "time_s": (step_count + 1,),
            "state": (step_count + 1, 4),
            "controls": (step_count, 2),
            "state_matrices": (step_count, 4, 4),
            "input_matrices": (step_count, 4, 2),
            "scaled_state_matrices": (step_count, 4, 4),
            "scaled_sample_input_matrices": (step_count, 4, 2),
            "scaled_energy_input_matrices": (step_count, 4, 2),
        }
        if step_count < 1:
            raise ValueError("controls must be a nonempty (N, 2) array")
        for name, shape in expected.items():
            array = np.asarray(getattr(self, name), dtype=float)
            if array.shape != shape or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be finite with shape {shape}")
            object.__setattr__(self, name, _readonly(array))


@dataclass(frozen=True, slots=True)
class RefinedCrossing:
    """Exact-step guard root retained inside one registered crossing bracket."""

    status: str
    time_s: float
    partial_dt_s: float
    state: FloatArray
    guard_residual: float
    transversality_per_s: float

    def __post_init__(self) -> None:
        scalars = (
            self.time_s,
            self.partial_dt_s,
            self.guard_residual,
            self.transversality_per_s,
        )
        if not all(math.isfinite(value) for value in scalars):
            raise ValueError("crossing scalars must be finite")
        if self.partial_dt_s <= 0.0:
            raise ValueError("partial_dt_s must be positive")
        object.__setattr__(
            self, "state", _readonly(_finite_vector("state", self.state, size=4))
        )


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
    scaled_state, scaled_sample_input, scaled_energy_input = scale_step_matrices(
        state_matrix,
        input_matrix,
        dt_s=dt_s,
        state_scales=state_scales,
        control_scales=control_scales,
    )
    return StepLinearization(
        state_matrix=state_matrix,
        input_matrix=input_matrix,
        scaled_state_matrix=scaled_state,
        scaled_sample_input_matrix=scaled_sample_input,
        scaled_energy_input_matrix=scaled_energy_input,
    )


def scale_step_matrices(
    state_matrix: npt.ArrayLike,
    input_matrix: npt.ArrayLike,
    *,
    dt_s: float,
    state_scales: StateScales,
    control_scales: ControlScales,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Scale exact-discrete matrices in sample and energy coordinates."""

    state = np.asarray(state_matrix, dtype=float)
    control = np.asarray(input_matrix, dtype=float)
    if state.shape != (4, 4) or not np.all(np.isfinite(state)):
        raise ValueError("state_matrix must be finite with shape (4, 4)")
    if control.shape != (4, 2) or not np.all(np.isfinite(control)):
        raise ValueError("input_matrix must be finite with shape (4, 2)")
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    x_scale = state_scales.array
    u_scale = control_scales.array
    scaled_state = state * x_scale[np.newaxis, :] / x_scale[:, np.newaxis]
    scaled_sample = control * u_scale[np.newaxis, :] / x_scale[:, np.newaxis]
    return (
        _readonly(scaled_state),
        _readonly(scaled_sample),
        _readonly(scaled_sample / math.sqrt(dt_s)),
    )


def linearize_trajectory(
    *,
    params: GolfModelParams,
    initial_state: npt.ArrayLike,
    controls: npt.ArrayLike,
    dt_s: float,
    state_steps: npt.ArrayLike,
    control_steps: npt.ArrayLike,
    state_scales: StateScales,
    control_scales: ControlScales,
) -> TrajectoryLinearization:
    """Linearize the exact RK4 map along every registered nominal step."""

    initial = _finite_vector("initial_state", initial_state, size=4).copy()
    commands = np.asarray(controls, dtype=float)
    if (
        commands.ndim != 2
        or commands.shape[0] == 0
        or commands.shape[1] != 2
        or not np.all(np.isfinite(commands))
    ):
        raise ValueError("controls must be a nonempty finite (N, 2) array")
    x_steps = _positive_vector("state_steps", state_steps, size=4)
    u_steps = _positive_vector("control_steps", control_steps, size=2)
    time_s, state = rollout_state_history(
        params,
        initial_state=initial,
        controls=commands,
        dt_s=dt_s,
    )
    linearizations = [
        step_linearization(
            params=params,
            state=state[index],
            control=command,
            time_s=float(time_s[index]),
            dt_s=dt_s,
            state_steps=x_steps,
            control_steps=u_steps,
            state_scales=state_scales,
            control_scales=control_scales,
        )
        for index, command in enumerate(commands)
    ]
    return TrajectoryLinearization(
        time_s=time_s,
        state=state,
        controls=commands,
        state_matrices=np.stack([item.state_matrix for item in linearizations]),
        input_matrices=np.stack([item.input_matrix for item in linearizations]),
        scaled_state_matrices=np.stack(
            [item.scaled_state_matrix for item in linearizations]
        ),
        scaled_sample_input_matrices=np.stack(
            [item.scaled_sample_input_matrix for item in linearizations]
        ),
        scaled_energy_input_matrices=np.stack(
            [item.scaled_energy_input_matrix for item in linearizations]
        ),
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


def propagated_terminal_input_sensitivity(
    state_matrices: npt.ArrayLike,
    input_matrices: npt.ArrayLike,
    *,
    pulse_step: int,
    channel_index: int,
) -> FloatArray:
    """Propagate one declared discrete input column to the terminal state."""

    state, control = _trajectory_matrices(state_matrices, input_matrices)
    if (
        isinstance(pulse_step, bool)
        or not isinstance(pulse_step, int)
        or not 0 <= pulse_step < state.shape[0]
    ):
        raise ValueError("pulse_step must index a trajectory step")
    if (
        isinstance(channel_index, bool)
        or not isinstance(channel_index, int)
        or not 0 <= channel_index < control.shape[2]
    ):
        raise ValueError("channel_index must index an input channel")
    sensitivity = control[pulse_step, :, channel_index].copy()
    for step in range(pulse_step + 1, state.shape[0]):
        sensitivity = state[step] @ sensitivity
    return _readonly(sensitivity)


def direct_terminal_pulse_sensitivity(
    *,
    params: GolfModelParams,
    initial_state: npt.ArrayLike,
    controls: npt.ArrayLike,
    dt_s: float,
    state_scales: StateScales,
    control_scales: ControlScales,
    pulse_step: int,
    channel_index: int,
    perturbation_scale: float,
) -> FloatArray:
    """Differentiate a complete rollout for one energy-normalized input pulse."""

    commands = np.asarray(controls, dtype=float)
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    if commands.ndim != 2:
        raise ValueError("controls must be a nonempty finite (N, 2) array")
    return direct_variable_terminal_pulse_sensitivity(
        params=params,
        initial_state=initial_state,
        controls=commands,
        step_durations_s=np.full(commands.shape[0], dt_s),
        state_scales=state_scales,
        control_scales=control_scales,
        pulse_step=pulse_step,
        channel_index=channel_index,
        perturbation_scale=perturbation_scale,
    )


def _variable_final_state(
    params: GolfModelParams,
    initial_state: FloatArray,
    controls: FloatArray,
    step_durations_s: FloatArray,
) -> FloatArray:
    backend = make_backend("ode", params, dt=float(step_durations_s[0]))
    state = initial_state.copy()
    time_s = 0.0
    for command, duration in zip(controls, step_durations_s, strict=True):
        state = registered_step(
            backend,
            state,
            command,
            time_s=time_s,
            dt_s=float(duration),
        )
        time_s += float(duration)
    return state


def direct_variable_terminal_pulse_sensitivity(
    *,
    params: GolfModelParams,
    initial_state: npt.ArrayLike,
    controls: npt.ArrayLike,
    step_durations_s: npt.ArrayLike,
    state_scales: StateScales,
    control_scales: ControlScales,
    pulse_step: int,
    channel_index: int,
    perturbation_scale: float,
) -> FloatArray:
    """Differentiate one pulse on a declared variable-step RK4 trajectory."""

    initial = _finite_vector("initial_state", initial_state, size=4).copy()
    commands = np.asarray(controls, dtype=float)
    if (
        commands.ndim != 2
        or commands.shape[0] == 0
        or commands.shape[1] != 2
        or not np.all(np.isfinite(commands))
    ):
        raise ValueError("controls must be a nonempty finite (N, 2) array")
    durations = _positive_vector(
        "step_durations_s", step_durations_s, size=commands.shape[0]
    )
    if (
        isinstance(pulse_step, bool)
        or not isinstance(pulse_step, int)
        or not 0 <= pulse_step < commands.shape[0]
    ):
        raise ValueError("pulse_step must index a trajectory step")
    if (
        isinstance(channel_index, bool)
        or not isinstance(channel_index, int)
        or not 0 <= channel_index < commands.shape[1]
    ):
        raise ValueError("channel_index must index an input channel")
    if not math.isfinite(perturbation_scale) or perturbation_scale <= 0.0:
        raise ValueError("perturbation_scale must be finite and positive")
    physical_delta = (
        control_scales.array[channel_index]
        * perturbation_scale
        / math.sqrt(float(durations[pulse_step]))
    )
    upper = commands.copy()
    lower = commands.copy()
    upper[pulse_step, channel_index] += physical_delta
    lower[pulse_step, channel_index] -= physical_delta
    upper_state = _variable_final_state(params, initial, upper, durations)
    lower_state = _variable_final_state(params, initial, lower, durations)
    sensitivity = (
        (upper_state - lower_state) / (2.0 * perturbation_scale) / state_scales.array
    )
    return _readonly(sensitivity)


def refine_guard_crossing(
    *,
    params: GolfModelParams,
    state_before: npt.ArrayLike,
    control: npt.ArrayLike,
    time_before_s: float,
    bracket_dt_s: float,
    guard_gradient: npt.ArrayLike,
    guard_offset: float = 0.0,
    guard_tolerance: float = 1e-10,
    time_tolerance_s: float = 1e-12,
    transversality_threshold: float = 1e-8,
    max_iterations: int = 80,
) -> RefinedCrossing:
    """Refine one negative-to-nonnegative guard bracket using exact RK4 steps.

    Unique-crossing classification remains a trajectory-level caller gate. This
    helper only refines a bracket already shown to contain the declared crossing.
    """

    state = _finite_vector("state_before", state_before, size=4).copy()
    command = _finite_vector("control", control, size=2).copy()
    gradient = _finite_vector("guard_gradient", guard_gradient, size=4)
    if np.linalg.norm(gradient) == 0.0:
        raise ValueError("guard_gradient must be nonzero")
    scalar_values = {
        "time_before_s": time_before_s,
        "bracket_dt_s": bracket_dt_s,
        "guard_offset": guard_offset,
        "guard_tolerance": guard_tolerance,
        "time_tolerance_s": time_tolerance_s,
        "transversality_threshold": transversality_threshold,
    }
    if not all(math.isfinite(value) for value in scalar_values.values()):
        raise ValueError("crossing controls must be finite")
    for name in (
        "bracket_dt_s",
        "guard_tolerance",
        "time_tolerance_s",
        "transversality_threshold",
    ):
        if scalar_values[name] <= 0.0:
            raise ValueError(f"{name} must be positive")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer")

    backend = make_backend("ode", params, dt=bracket_dt_s)

    def guard_at(partial_dt_s: float) -> tuple[float, FloatArray]:
        candidate = registered_step(
            backend,
            state,
            command,
            time_s=time_before_s,
            dt_s=partial_dt_s,
        )
        return float(gradient @ candidate - guard_offset), candidate

    lower = 0.0
    upper = float(bracket_dt_s)
    lower_value = float(gradient @ state - guard_offset)
    upper_value, event_state = guard_at(upper)
    if lower_value >= 0.0 or upper_value < 0.0:
        raise ValueError("registered guard bracket must be negative-to-nonnegative")
    residual = upper_value
    partial = upper
    for _ in range(max_iterations):
        partial = 0.5 * (lower + upper)
        residual, event_state = guard_at(partial)
        if abs(residual) <= guard_tolerance:
            break
        if residual >= 0.0:
            upper = partial
        else:
            lower = partial
        if upper - lower <= time_tolerance_s:
            break
    if abs(residual) > guard_tolerance:
        raise ValueError("guard root did not meet the registered residual tolerance")
    flow = state_derivative(params, event_state, command)
    transversality = float(gradient @ flow)
    status = (
        "near_grazing"
        if abs(transversality) <= transversality_threshold
        else "transverse_candidate"
    )
    return RefinedCrossing(
        status=status,
        time_s=float(time_before_s + partial),
        partial_dt_s=partial,
        state=event_state,
        guard_residual=residual,
        transversality_per_s=transversality,
    )


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
    "RefinedCrossing",
    "StepLinearization",
    "TrajectoryLinearization",
    "direct_terminal_pulse_sensitivity",
    "direct_variable_terminal_pulse_sensitivity",
    "event_conditioned_gramian",
    "frozen_local_gramian",
    "linearize_trajectory",
    "propagated_terminal_input_sensitivity",
    "reachability_history",
    "refine_guard_crossing",
    "scale_step_matrices",
    "step_linearization",
]
