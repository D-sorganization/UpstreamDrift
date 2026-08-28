"""Deterministic bounded multiple shooting for a declared guard target.

The solver uses state nodes and exact-RK4 continuity constraints over a fixed
nominal crossing bracket.  The final partial-step duration is a decision
variable, so the arrival time may change within that bracket.  Every retained
candidate is independently replayed by :mod:`bounded_event_reachability`;
solver-internal state nodes are never accepted as scientific evidence alone.

This is a local model-scenario feasibility baseline.  It does not establish
global reachability, human torque/rate capacity, controller superiority,
passive torque, robustness, or coaching guidance.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from scripts.research.proximal_distal_energy.bounded_event_reachability import (
    BoundedEventReachabilityProblem,
    BoundedReachabilityOutcome,
    EventReplayStatus,
    FeasibilityStatus,
    evaluate_bounded_candidate,
    replay_guard_event,
)
from scripts.research.proximal_distal_energy.phase_event_stability import (
    registered_step,
)
from src.shared.python.simulation_backends import make_backend

FloatArray: TypeAlias = npt.NDArray[np.float64]


def _readonly(value: npt.ArrayLike) -> FloatArray:
    array = np.asarray(value, dtype=float).copy()
    if not np.all(np.isfinite(array)):
        raise ValueError("multiple-shooting result contains non-finite values")
    array.setflags(write=False)
    return array


class MultipleShootingStatus(str, Enum):
    """Solver outcome after independent nonlinear replay."""

    CONVERGED = "converged"
    INFEASIBLE = "infeasible"
    ITERATION_LIMIT = "iteration_limit"
    NUMERICAL_FAILURE = "numerical_failure"
    REPLAY_REJECTED = "replay_rejected"


@dataclass(frozen=True, slots=True)
class MultipleShootingConfig:
    """Deterministic discretization and SLSQP qualification contract."""

    segment_count: int
    max_iterations: int = 300
    constraint_tolerance: float = 1e-7
    objective_tolerance: float = 1e-10
    seed: int = 0
    initial_control_fraction: float = 0.0

    def __post_init__(self) -> None:
        for name in ("segment_count", "max_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("constraint_tolerance", "objective_tolerance"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer")
        if (
            not math.isfinite(self.initial_control_fraction)
            or not 0.0 <= self.initial_control_fraction <= 1.0
        ):
            raise ValueError("initial_control_fraction must be finite in [0, 1]")


@dataclass(frozen=True, slots=True)
class MultipleShootingResult:
    """Qualified solver record plus independent direct replay."""

    status: MultipleShootingStatus
    solver_success: bool
    message: str
    iterations: int
    objective: float
    event_time_s: float | None
    maximum_continuity_residual: float
    maximum_target_residual: float
    segment_perturbations: FloatArray
    state_nodes: FloatArray
    perturbations: FloatArray
    replay: BoundedReachabilityOutcome | None

    def __post_init__(self) -> None:
        for name in (
            "objective",
            "maximum_continuity_residual",
            "maximum_target_residual",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.event_time_s is not None and not math.isfinite(self.event_time_s):
            raise ValueError("event_time_s must be finite when available")
        if self.iterations < 0:
            raise ValueError("iterations must be nonnegative")
        segment = np.asarray(self.segment_perturbations, dtype=float)
        nodes = np.asarray(self.state_nodes, dtype=float)
        perturbations = np.asarray(self.perturbations, dtype=float)
        if segment.ndim != 2 or segment.shape[1] != 2:
            raise ValueError("segment_perturbations must have shape (M, 2)")
        if nodes.shape != (segment.shape[0], 4):
            raise ValueError("state_nodes must have shape (M, 4)")
        if perturbations.ndim != 2 or perturbations.shape[1] != 2:
            raise ValueError("perturbations must have shape (N, 2)")
        object.__setattr__(self, "segment_perturbations", _readonly(segment))
        object.__setattr__(self, "state_nodes", _readonly(nodes))
        object.__setattr__(self, "perturbations", _readonly(perturbations))


@dataclass(frozen=True, slots=True)
class _ShootingLayout:
    segment_indices: tuple[FloatArray, ...]
    prefix_step_count: int
    nominal_partial_dt_s: float


def _crossing_layout(
    problem: BoundedEventReachabilityProblem,
    segment_count: int,
) -> _ShootingLayout:
    replay = replay_guard_event(
        params=problem.params,
        initial_state=problem.initial_state,
        controls=problem.nominal_controls,
        dt_s=problem.dt_s,
        guard=problem.guard,
    )
    if replay.status is not EventReplayStatus.TRANSVERSE or replay.time_s is None:
        raise ValueError("nominal problem must have one transverse guard crossing")
    full_steps = int(math.floor(replay.time_s / problem.dt_s))
    partial = float(replay.time_s - full_steps * problem.dt_s)
    if partial <= problem.guard.time_tolerance_s:
        full_steps -= 1
        partial = problem.dt_s
    prefix_count = full_steps + 1
    if prefix_count > problem.nominal_controls.shape[0]:
        raise ValueError("nominal event lies outside the control horizon")
    if segment_count > prefix_count:
        raise ValueError("segment_count cannot exceed the pre-event step count")
    partitions = tuple(
        np.asarray(indices, dtype=np.int64)
        for indices in np.array_split(np.arange(prefix_count), segment_count)
    )
    if any(indices.size == 0 for indices in partitions):
        raise ValueError("every shooting segment must contain at least one step")
    return _ShootingLayout(partitions, prefix_count, partial)


def _unpack(
    decision: FloatArray,
    *,
    segment_count: int,
) -> tuple[FloatArray, FloatArray, float]:
    control_end = 2 * segment_count
    state_end = control_end + 4 * segment_count
    segment_controls = decision[:control_end].reshape(segment_count, 2)
    state_nodes = decision[control_end:state_end].reshape(segment_count, 4)
    partial_dt_s = float(decision[state_end])
    return segment_controls, state_nodes, partial_dt_s


def _step_durations(
    layout: _ShootingLayout,
    *,
    dt_s: float,
    partial_dt_s: float,
) -> FloatArray:
    durations = np.full(layout.prefix_step_count, dt_s, dtype=float)
    durations[-1] = partial_dt_s
    return durations


def _integrate_segment(
    problem: BoundedEventReachabilityProblem,
    *,
    start_state: FloatArray,
    perturbation: FloatArray,
    indices: FloatArray,
    durations: FloatArray,
) -> FloatArray:
    backend = make_backend("ode", problem.params, dt=problem.dt_s)
    state = start_state.copy()
    time_s = float(np.sum(durations[: int(indices[0])]))
    for index_value in indices:
        index = int(index_value)
        duration = float(durations[index])
        state = registered_step(
            backend,
            state,
            problem.nominal_controls[index] + perturbation,
            time_s=time_s,
            dt_s=duration,
        )
        time_s += duration
    return state


def _shooting_residual(
    decision: FloatArray,
    *,
    problem: BoundedEventReachabilityProblem,
    layout: _ShootingLayout,
) -> FloatArray:
    segment_controls, state_nodes, partial_dt_s = _unpack(
        decision, segment_count=len(layout.segment_indices)
    )
    durations = _step_durations(
        layout,
        dt_s=problem.dt_s,
        partial_dt_s=partial_dt_s,
    )
    state_scale = problem.state_scales.array
    residuals: list[FloatArray] = []
    start = np.asarray(problem.initial_state, dtype=float)
    for perturbation, node, indices in zip(
        segment_controls,
        state_nodes,
        layout.segment_indices,
        strict=True,
    ):
        predicted = _integrate_segment(
            problem,
            start_state=start,
            perturbation=perturbation,
            indices=indices,
            durations=durations,
        )
        residuals.append((predicted - node) / state_scale)
        start = node
    target = np.asarray(problem.target_event_state, dtype=float)
    residuals.append((state_nodes[-1] - target) / state_scale)
    return np.concatenate(residuals)


def _segment_rates(segment_controls: FloatArray, *, dt_s: float) -> FloatArray:
    return np.diff(np.vstack((np.zeros((1, 2)), segment_controls)), axis=0) / dt_s


def _expand_perturbations(
    segment_controls: FloatArray,
    layout: _ShootingLayout,
) -> FloatArray:
    perturbations = np.empty((layout.prefix_step_count, 2), dtype=float)
    for perturbation, indices in zip(
        segment_controls, layout.segment_indices, strict=True
    ):
        perturbations[indices.astype(int)] = perturbation
    return perturbations


def _seeded_segment_controls(
    problem: BoundedEventReachabilityProblem,
    config: MultipleShootingConfig,
) -> FloatArray:
    """Generate a reproducible feasible bounded-walk initialization."""

    fraction = config.initial_control_fraction
    controls = np.zeros((config.segment_count, 2), dtype=float)
    if fraction == 0.0:
        return controls
    lower = fraction * problem.bounds.lower_array
    upper = fraction * problem.bounds.upper_array
    maximum_step = fraction * problem.bounds.rate_array * problem.dt_s
    generator = np.random.default_rng(config.seed)
    previous = np.zeros(2, dtype=float)
    for index in range(config.segment_count):
        feasible_lower = np.maximum(lower, previous - maximum_step)
        feasible_upper = np.minimum(upper, previous + maximum_step)
        if np.any(feasible_lower > feasible_upper):
            raise ValueError("seeded control initialization has no feasible interval")
        controls[index] = generator.uniform(feasible_lower, feasible_upper)
        previous = controls[index]
    return controls


def _initial_decision(
    problem: BoundedEventReachabilityProblem,
    layout: _ShootingLayout,
    config: MultipleShootingConfig,
) -> FloatArray:
    segment_count = len(layout.segment_indices)
    if segment_count != config.segment_count:
        raise ValueError("layout and config segment counts must match")
    controls = _seeded_segment_controls(problem, config)
    nodes = np.empty((segment_count, 4), dtype=float)
    durations = _step_durations(
        layout,
        dt_s=problem.dt_s,
        partial_dt_s=layout.nominal_partial_dt_s,
    )
    state = np.asarray(problem.initial_state, dtype=float)
    for segment, indices in enumerate(layout.segment_indices):
        state = _integrate_segment(
            problem,
            start_state=state,
            perturbation=controls[segment],
            indices=indices,
            durations=durations,
        )
        nodes[segment] = state
    return np.concatenate(
        (controls.ravel(), nodes.ravel(), [layout.nominal_partial_dt_s])
    )


def _objective(
    decision: FloatArray,
    *,
    problem: BoundedEventReachabilityProblem,
    layout: _ShootingLayout,
) -> float:
    segment_controls, _, partial_dt_s = _unpack(
        decision, segment_count=len(layout.segment_indices)
    )
    durations = _step_durations(
        layout,
        dt_s=problem.dt_s,
        partial_dt_s=partial_dt_s,
    )
    energy = 0.0
    control_scale = problem.control_scales.array
    for perturbation, indices in zip(
        segment_controls, layout.segment_indices, strict=True
    ):
        energy += float(
            np.sum(durations[indices.astype(int)])
            * np.sum(np.square(perturbation / control_scale))
        )
    return energy


def _decision_bounds(
    problem: BoundedEventReachabilityProblem,
    layout: _ShootingLayout,
) -> list[tuple[float | None, float | None]]:
    bounds: list[tuple[float | None, float | None]] = []
    for _ in layout.segment_indices:
        bounds.extend(
            zip(
                problem.bounds.lower_nm,
                problem.bounds.upper_nm,
                strict=True,
            )
        )
    bounds.extend([(None, None)] * (4 * len(layout.segment_indices)))
    bounds.append((problem.guard.time_tolerance_s, problem.dt_s))
    return bounds


def _rate_margin(
    decision: FloatArray,
    *,
    problem: BoundedEventReachabilityProblem,
    segment_count: int,
) -> FloatArray:
    segment_controls, _, _ = _unpack(decision, segment_count=segment_count)
    rates = _segment_rates(segment_controls, dt_s=problem.dt_s)
    return (problem.bounds.rate_array[np.newaxis, :] - np.abs(rates)).ravel()


def _qualified_result(
    *,
    problem: BoundedEventReachabilityProblem,
    config: MultipleShootingConfig,
    layout: _ShootingLayout,
    decision: FloatArray,
    solver_success: bool,
    message: str,
    iterations: int,
) -> MultipleShootingResult:
    segment_controls, state_nodes, partial_dt_s = _unpack(
        decision, segment_count=config.segment_count
    )
    residual = _shooting_residual(decision, problem=problem, layout=layout)
    continuity = residual[: 4 * config.segment_count]
    target = residual[4 * config.segment_count :]
    maximum_continuity = float(np.max(np.abs(continuity)))
    maximum_target = float(np.max(np.abs(target)))
    perturbations = _expand_perturbations(segment_controls, layout)
    replay_problem = replace(
        problem,
        nominal_controls=problem.nominal_controls[: layout.prefix_step_count],
    )
    replay = evaluate_bounded_candidate(replay_problem, perturbations)
    replay_ok = replay.feasibility_status is FeasibilityStatus.FEASIBLE
    constraints_ok = (
        maximum_continuity <= config.constraint_tolerance
        and maximum_target <= config.constraint_tolerance
    )
    if solver_success and constraints_ok and replay_ok:
        status = MultipleShootingStatus.CONVERGED
    elif solver_success and not replay_ok:
        status = MultipleShootingStatus.REPLAY_REJECTED
    elif "iteration" in message.lower():
        status = MultipleShootingStatus.ITERATION_LIMIT
    else:
        status = MultipleShootingStatus.INFEASIBLE
    event_time = (
        replay.event.time_s
        if replay.event is not None and replay.event.time_s is not None
        else (layout.prefix_step_count - 1) * problem.dt_s + partial_dt_s
    )
    return MultipleShootingResult(
        status=status,
        solver_success=solver_success,
        message=message,
        iterations=iterations,
        objective=_objective(decision, problem=problem, layout=layout),
        event_time_s=event_time,
        maximum_continuity_residual=maximum_continuity,
        maximum_target_residual=maximum_target,
        segment_perturbations=segment_controls,
        state_nodes=state_nodes,
        perturbations=perturbations,
        replay=replay,
    )


def solve_bounded_event_multiple_shooting(
    problem: BoundedEventReachabilityProblem,
    config: MultipleShootingConfig,
) -> MultipleShootingResult:
    """Solve one local event target and require independent exact replay."""

    layout = _crossing_layout(problem, config.segment_count)
    initial = _initial_decision(problem, layout, config)
    if problem.bounds.is_zero_authority:
        result = _qualified_result(
            problem=problem,
            config=config,
            layout=layout,
            decision=initial,
            solver_success=False,
            message="zero incremental authority; direct replay only",
            iterations=0,
        )
        if result.replay is not None and (
            result.replay.feasibility_status is FeasibilityStatus.FEASIBLE
        ):
            return replace(
                result,
                status=MultipleShootingStatus.CONVERGED,
                solver_success=True,
            )
        return replace(result, status=MultipleShootingStatus.INFEASIBLE)

    equality = {
        "type": "eq",
        "fun": lambda decision: _shooting_residual(
            np.asarray(decision, dtype=float),
            problem=problem,
            layout=layout,
        ),
    }
    rate = {
        "type": "ineq",
        "fun": lambda decision: _rate_margin(
            np.asarray(decision, dtype=float),
            problem=problem,
            segment_count=config.segment_count,
        ),
    }
    try:
        solved = minimize(
            lambda decision: _objective(
                np.asarray(decision, dtype=float),
                problem=problem,
                layout=layout,
            ),
            initial,
            method="SLSQP",
            bounds=_decision_bounds(problem, layout),
            constraints=(equality, rate),
            options={
                "maxiter": config.max_iterations,
                "ftol": config.objective_tolerance,
                "disp": False,
            },
        )
    except (ArithmeticError, RuntimeError, ValueError) as exc:
        segment_controls, state_nodes, _ = _unpack(
            initial, segment_count=config.segment_count
        )
        residual = _shooting_residual(initial, problem=problem, layout=layout)
        continuity = residual[: 4 * config.segment_count]
        target = residual[4 * config.segment_count :]
        return MultipleShootingResult(
            status=MultipleShootingStatus.NUMERICAL_FAILURE,
            solver_success=False,
            message=str(exc),
            iterations=0,
            objective=_objective(initial, problem=problem, layout=layout),
            event_time_s=None,
            maximum_continuity_residual=float(np.max(np.abs(continuity))),
            maximum_target_residual=float(np.max(np.abs(target))),
            segment_perturbations=segment_controls,
            state_nodes=state_nodes,
            perturbations=_expand_perturbations(segment_controls, layout),
            replay=None,
        )
    decision = np.asarray(solved.x, dtype=float)
    return _qualified_result(
        problem=problem,
        config=config,
        layout=layout,
        decision=decision,
        solver_success=bool(solved.success),
        message=str(solved.message),
        iterations=int(getattr(solved, "nit", 0)),
    )


__all__ = [
    "MultipleShootingConfig",
    "MultipleShootingResult",
    "MultipleShootingStatus",
    "solve_bounded_event_multiple_shooting",
]
