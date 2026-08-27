"""Finite-time transition and event-sensitivity mechanics.

The registered downswing is a finite, nonperiodic trajectory.  These helpers
therefore report local finite-window amplification and transverse event-time
sensitivity.  They do not infer asymptotic stability, a basin of attraction,
human robustness, neural timing demand, or coaching strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from src.shared.python.simulation_backends import GolfModelParams, make_backend
from src.shared.python.simulation_backends.protocol import SimState, SimulationBackend

FloatArray: TypeAlias = npt.NDArray[np.float64]


def _vector(name: str, value: npt.ArrayLike, *, size: int | None = None) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector")
    if size is not None and array.size != size:
        raise ValueError(f"{name} must contain {size} entries")
    return array


def _square_matrix(name: str, value: npt.ArrayLike) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if (
        array.ndim != 2
        or array.shape[0] == 0
        or array.shape[0] != array.shape[1]
        or not np.all(np.isfinite(array))
    ):
        raise ValueError(f"{name} must be a nonempty finite square matrix")
    return array


@dataclass(frozen=True, slots=True)
class StateScales:
    """Positive scales for the four analytical state coordinates."""

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in self.values)
        if len(values) != 4 or any(
            not math.isfinite(value) or value <= 0.0 for value in values
        ):
            raise ValueError("state scales must contain four finite positive values")
        object.__setattr__(self, "values", values)

    @property
    def array(self) -> FloatArray:
        """Return a defensive NumPy representation."""

        return np.asarray(self.values, dtype=float)


@dataclass(frozen=True, slots=True)
class FiniteTimeSpectra:
    """Singular values and finite-time exponents for transition maps."""

    singular_values: FloatArray
    exponents_per_s: FloatArray


@dataclass(frozen=True, slots=True)
class EventSensitivity:
    """Implicit event-time derivative or a typed near-grazing outcome."""

    status: str
    transversality_per_s: float
    derivative_s_per_scaled_state: FloatArray | None


@dataclass(frozen=True, slots=True)
class PeriodicityGate:
    """Finite-state closure gate that controls Floquet eligibility."""

    normalized_residual: float
    tolerance: float
    periodic: bool
    floquet_eligible: bool


@dataclass(frozen=True, slots=True)
class Crossing:
    """First positive guard crossing and ambiguity classification."""

    status: str
    crossing_count: int
    sample_index: int | None
    fraction: float | None
    time_s: float | None


@dataclass(frozen=True, slots=True)
class TransitionRollout:
    """Nominal state history and physical state-transition matrices."""

    time_s: FloatArray
    state: FloatArray
    transition: FloatArray


@dataclass(frozen=True, slots=True)
class DirectEventSensitivity:
    """Finite-difference event-time derivative with typed crossing status."""

    status: str
    derivative_s_per_scaled_state: FloatArray | None
    crossing_statuses: tuple[str, ...]


def normalized_transition(
    transition: npt.ArrayLike, state_scales: StateScales
) -> FloatArray:
    """Map a dimensional state transition into declared scaled coordinates."""

    matrix = _square_matrix("transition", transition)
    scales = state_scales.array
    if matrix.shape != (scales.size, scales.size):
        raise ValueError("transition dimension must match state scales")
    return matrix * scales[np.newaxis, :] / scales[:, np.newaxis]


def _step_state(
    backend: SimulationBackend,
    state: FloatArray,
    control: FloatArray,
    *,
    time_s: float,
    dt_s: float,
) -> FloatArray:
    """Advance one state through the backend's registered RK4 map."""

    backend.reset(SimState(q=state[:2].copy(), v=state[2:].copy(), time=float(time_s)))
    backend.set_control(control.copy())
    backend.step(dt_s)
    result = backend.get_state()
    return np.concatenate((result.q, result.v)).astype(float, copy=False)


def _step_jacobian(
    backend: SimulationBackend,
    state: FloatArray,
    control: FloatArray,
    *,
    time_s: float,
    dt_s: float,
    state_steps: FloatArray,
) -> FloatArray:
    """Central-difference the exact discrete RK4 step map."""

    jacobian = np.empty((state.size, state.size), dtype=float)
    for column, step in enumerate(state_steps):
        upper = state.copy()
        lower = state.copy()
        upper[column] += step
        lower[column] -= step
        upper_state = _step_state(backend, upper, control, time_s=time_s, dt_s=dt_s)
        lower_state = _step_state(backend, lower, control, time_s=time_s, dt_s=dt_s)
        jacobian[:, column] = (upper_state - lower_state) / (2.0 * step)
    if not np.all(np.isfinite(jacobian)):
        raise ValueError("step Jacobian contains non-finite values")
    return jacobian


def propagate_state_transition(
    params: GolfModelParams,
    *,
    initial_state: npt.ArrayLike,
    controls: npt.ArrayLike,
    dt_s: float,
    state_steps: npt.ArrayLike,
) -> TransitionRollout:
    """Roll out the analytical model and its discrete variational map."""

    initial = _vector("initial_state", initial_state, size=4)
    commands = np.asarray(controls, dtype=float)
    steps = _vector("state_steps", state_steps, size=4)
    if (
        commands.ndim != 2
        or commands.shape[0] == 0
        or commands.shape[1] != 2
        or not np.all(np.isfinite(commands))
    ):
        raise ValueError("controls must be a nonempty finite (N, 2) array")
    if np.any(steps <= 0.0):
        raise ValueError("state_steps must be positive")
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")

    nominal_backend = make_backend("ode", params, dt=dt_s)
    nominal_backend.reset(
        SimState(q=initial[:2].copy(), v=initial[2:].copy(), time=0.0)
    )
    trace = nominal_backend.rollout(commands, horizon=commands.shape[0], dt=dt_s)
    states = np.column_stack((trace.q, trace.v))
    transitions = np.empty((states.shape[0], 4, 4), dtype=float)
    transitions[0] = np.eye(4)
    jacobian_backend = make_backend("ode", params, dt=dt_s)
    for index, control in enumerate(commands):
        step_map = _step_jacobian(
            jacobian_backend,
            states[index],
            control,
            time_s=float(trace.t[index]),
            dt_s=dt_s,
            state_steps=steps,
        )
        transitions[index + 1] = step_map @ transitions[index]
    return TransitionRollout(
        time_s=np.asarray(trace.t, dtype=float),
        state=np.asarray(states, dtype=float),
        transition=transitions,
    )


def _rollout_final_state(
    params: GolfModelParams,
    initial_state: FloatArray,
    controls: FloatArray,
    dt_s: float,
) -> FloatArray:
    backend = make_backend("ode", params, dt=dt_s)
    backend.reset(SimState(q=initial_state[:2], v=initial_state[2:], time=0.0))
    trace = backend.rollout(controls, horizon=controls.shape[0], dt=dt_s)
    return np.concatenate((trace.q[-1], trace.v[-1]))


def rollout_state_history(
    params: GolfModelParams,
    *,
    initial_state: npt.ArrayLike,
    controls: npt.ArrayLike,
    dt_s: float,
) -> tuple[FloatArray, FloatArray]:
    """Return time and four-state histories from an arbitrary initial state."""

    initial = _vector("initial_state", initial_state, size=4)
    commands = np.asarray(controls, dtype=float)
    if (
        commands.ndim != 2
        or commands.shape[0] == 0
        or commands.shape[1] != 2
        or not np.all(np.isfinite(commands))
    ):
        raise ValueError("controls must be a nonempty finite (N, 2) array")
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    backend = make_backend("ode", params, dt=dt_s)
    backend.reset(SimState(q=initial[:2], v=initial[2:], time=0.0))
    trace = backend.rollout(commands, horizon=commands.shape[0], dt=dt_s)
    return np.asarray(trace.t, dtype=float), np.column_stack((trace.q, trace.v))


def direct_transition_control(
    params: GolfModelParams,
    *,
    initial_state: npt.ArrayLike,
    controls: npt.ArrayLike,
    dt_s: float,
    state_scales: StateScales,
    perturbation_scale: float,
) -> FloatArray:
    """Central-difference the complete rollout in scaled coordinates."""

    initial = _vector("initial_state", initial_state, size=4)
    commands = np.asarray(controls, dtype=float)
    if commands.ndim != 2 or commands.shape[1] != 2 or commands.shape[0] == 0:
        raise ValueError("controls must be a nonempty (N, 2) array")
    if not np.all(np.isfinite(commands)):
        raise ValueError("controls must be finite")
    if not math.isfinite(perturbation_scale) or perturbation_scale <= 0.0:
        raise ValueError("perturbation_scale must be finite and positive")
    scales = state_scales.array
    result = np.empty((4, 4), dtype=float)
    for column, scale in enumerate(scales):
        delta = perturbation_scale * scale
        upper = initial.copy()
        lower = initial.copy()
        upper[column] += delta
        lower[column] -= delta
        upper_final = _rollout_final_state(params, upper, commands, dt_s)
        lower_final = _rollout_final_state(params, lower, commands, dt_s)
        result[:, column] = (
            (upper_final - lower_final) / (2.0 * perturbation_scale) / scales
        )
    return result


def direct_event_time_control(
    params: GolfModelParams,
    *,
    initial_state: npt.ArrayLike,
    controls: npt.ArrayLike,
    dt_s: float,
    state_scales: StateScales,
    perturbation_scale: float,
    guard_gradient: npt.ArrayLike,
    guard_offset: float = 0.0,
) -> DirectEventSensitivity:
    """Differentiate a linear state guard using complete perturbed rollouts."""

    initial = _vector("initial_state", initial_state, size=4)
    commands = np.asarray(controls, dtype=float)
    gradient = _vector("guard_gradient", guard_gradient, size=4)
    if not math.isfinite(guard_offset):
        raise ValueError("guard_offset must be finite")
    if not math.isfinite(perturbation_scale) or perturbation_scale <= 0.0:
        raise ValueError("perturbation_scale must be finite and positive")
    scales = state_scales.array
    derivative = np.empty(4, dtype=float)
    statuses: list[str] = []
    for column, scale in enumerate(scales):
        delta = perturbation_scale * scale
        upper = initial.copy()
        lower = initial.copy()
        upper[column] += delta
        lower[column] -= delta
        upper_time, upper_state = rollout_state_history(
            params,
            initial_state=upper,
            controls=commands,
            dt_s=dt_s,
        )
        lower_time, lower_state = rollout_state_history(
            params,
            initial_state=lower,
            controls=commands,
            dt_s=dt_s,
        )
        upper_crossing = first_positive_crossing(
            upper_time, upper_state @ gradient - guard_offset
        )
        lower_crossing = first_positive_crossing(
            lower_time, lower_state @ gradient - guard_offset
        )
        statuses.extend((upper_crossing.status, lower_crossing.status))
        if (
            upper_crossing.crossing_count != 1
            or lower_crossing.crossing_count != 1
            or upper_crossing.time_s is None
            or lower_crossing.time_s is None
        ):
            return DirectEventSensitivity(
                status="unavailable_nonunique_or_absent_crossing",
                derivative_s_per_scaled_state=None,
                crossing_statuses=tuple(statuses),
            )
        derivative[column] = (upper_crossing.time_s - lower_crossing.time_s) / (
            2.0 * perturbation_scale
        )
    return DirectEventSensitivity(
        status="available_transverse_candidates",
        derivative_s_per_scaled_state=derivative,
        crossing_statuses=tuple(statuses),
    )


def state_derivative(
    params: GolfModelParams, state: npt.ArrayLike, control: npt.ArrayLike
) -> FloatArray:
    """Evaluate the canonical analytical four-state vector field."""

    vector = _vector("state", state, size=4)
    command = _vector("control", control, size=2)
    backend = make_backend("ode", params)
    derivative = np.empty(4, dtype=float)
    derivative[:2] = vector[2:]
    derivative[2:] = backend.forward_dynamics(vector[:2], vector[2:], command)
    return derivative


def first_positive_crossing(
    time_s: npt.ArrayLike, guard_value: npt.ArrayLike
) -> Crossing:
    """Return the first negative-to-nonnegative crossing with interpolation."""

    times = _vector("time_s", time_s)
    values = _vector("guard_value", guard_value)
    if times.size != values.size or times.size < 2 or np.any(np.diff(times) <= 0.0):
        raise ValueError("time and guard values must share a strictly increasing grid")
    indices = np.flatnonzero((values[:-1] < 0.0) & (values[1:] >= 0.0))
    if indices.size == 0:
        return Crossing("absent", 0, None, None, None)
    index = int(indices[0])
    denominator = values[index + 1] - values[index]
    fraction = 0.0 if denominator == 0.0 else float(-values[index] / denominator)
    event_time = float(times[index] + fraction * (times[index + 1] - times[index]))
    status = "transverse_candidate" if indices.size == 1 else "multiple"
    return Crossing(status, int(indices.size), index, fraction, event_time)


def finite_time_spectra(
    transitions: npt.ArrayLike, elapsed_time_s: npt.ArrayLike
) -> FiniteTimeSpectra:
    """Compute singular spectra and local finite-window exponents.

    The first exponent row is ``NaN`` because a zero-duration Lyapunov rate is
    undefined.  Singular values remain available at the initial identity map.
    """

    maps = np.asarray(transitions, dtype=float)
    times = _vector("elapsed_time_s", elapsed_time_s)
    if (
        maps.ndim != 3
        or maps.shape[0] != times.size
        or maps.shape[1] == 0
        or maps.shape[1] != maps.shape[2]
        or not np.all(np.isfinite(maps))
    ):
        raise ValueError("transitions must be finite square maps on the time grid")
    if times.size < 2 or times[0] != 0.0 or np.any(np.diff(times) <= 0.0):
        raise ValueError("elapsed time must start at zero and increase strictly")
    singular = np.asarray(
        [np.linalg.svd(matrix, compute_uv=False) for matrix in maps], dtype=float
    )
    exponents = np.full_like(singular, np.nan)
    exponents[1:] = np.log(singular[1:]) / times[1:, np.newaxis]
    return FiniteTimeSpectra(singular_values=singular, exponents_per_s=exponents)


def event_time_sensitivity(
    transition_to_event: npt.ArrayLike,
    *,
    event_flow: npt.ArrayLike,
    guard_gradient: npt.ArrayLike,
    state_scales: StateScales,
    transversality_threshold: float,
) -> EventSensitivity:
    """Differentiate a state guard's event time with respect to scaled state.

    For ``h(x)=0``, the implicit derivative is
    ``dt/dz0 = -(dh/dx Phi S)/(dh/dx f)``.  A derivative is deliberately not
    returned when the crossing is near grazing.
    """

    transition = _square_matrix("transition_to_event", transition_to_event)
    dimension = transition.shape[0]
    flow = _vector("event_flow", event_flow, size=dimension)
    gradient = _vector("guard_gradient", guard_gradient, size=dimension)
    scales = state_scales.array
    if dimension != scales.size:
        raise ValueError("transition dimension must match state scales")
    if not math.isfinite(transversality_threshold) or transversality_threshold <= 0.0:
        raise ValueError("transversality_threshold must be finite and positive")
    denominator = float(gradient @ flow)
    if abs(denominator) <= transversality_threshold:
        return EventSensitivity(
            status="near_grazing",
            transversality_per_s=denominator,
            derivative_s_per_scaled_state=None,
        )
    derivative = -(gradient @ transition) * scales / denominator
    return EventSensitivity(
        status="transverse",
        transversality_per_s=denominator,
        derivative_s_per_scaled_state=np.asarray(derivative, dtype=float),
    )


def saltation_matrix(
    *,
    reset_jacobian: npt.ArrayLike,
    flow_before: npt.ArrayLike,
    flow_after: npt.ArrayLike,
    guard_gradient: npt.ArrayLike,
    guard_time_derivative: float,
    reset_time_derivative: npt.ArrayLike,
    transversality_threshold: float = 1e-12,
) -> FloatArray:
    """Return the first-order saltation map for a time-dependent guard."""

    reset = _square_matrix("reset_jacobian", reset_jacobian)
    dimension = reset.shape[0]
    before = _vector("flow_before", flow_before, size=dimension)
    after = _vector("flow_after", flow_after, size=dimension)
    gradient = _vector("guard_gradient", guard_gradient, size=dimension)
    reset_time = _vector("reset_time_derivative", reset_time_derivative, size=dimension)
    scalars = (float(guard_time_derivative), float(transversality_threshold))
    if not all(math.isfinite(value) for value in scalars) or scalars[1] <= 0.0:
        raise ValueError("guard derivative and threshold must be finite")
    denominator = scalars[0] + float(gradient @ before)
    if abs(denominator) <= scalars[1]:
        raise ValueError("saltation requires a transverse guard")
    jump = after - reset @ before - reset_time
    return reset + np.outer(jump, gradient) / denominator


def periodicity_gate(
    *,
    initial_state: npt.ArrayLike,
    final_state: npt.ArrayLike,
    state_scales: StateScales,
    tolerance: float,
) -> PeriodicityGate:
    """Require scaled state closure before any Floquet interpretation."""

    initial = _vector("initial_state", initial_state, size=4)
    final = _vector("final_state", final_state, size=4)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    residual = float(np.linalg.norm((final - initial) / state_scales.array))
    periodic = residual <= tolerance
    return PeriodicityGate(
        normalized_residual=residual,
        tolerance=float(tolerance),
        periodic=periodic,
        floquet_eligible=periodic,
    )


__all__ = [
    "Crossing",
    "DirectEventSensitivity",
    "EventSensitivity",
    "FiniteTimeSpectra",
    "PeriodicityGate",
    "StateScales",
    "TransitionRollout",
    "direct_transition_control",
    "direct_event_time_control",
    "event_time_sensitivity",
    "finite_time_spectra",
    "first_positive_crossing",
    "normalized_transition",
    "periodicity_gate",
    "propagate_state_transition",
    "rollout_state_history",
    "saltation_matrix",
    "state_derivative",
]
