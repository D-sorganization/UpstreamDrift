"""Bounded nonlinear replay at a declared delivery guard.

This module is the independent replay boundary for issue #9124.  It evaluates
finite control perturbations about a registered nominal torque history through
the exact analytical-double-pendulum RK4 operator.  Bounds are model-scenario
constraints, not human strength evidence.  The module does not establish
global reachability, controller optimality, physiological feasibility, passive
torque, or a coaching strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from scripts.research.proximal_distal_energy.phase_event_stability import (
    StateScales,
    first_positive_crossing,
    rollout_state_history,
)
from scripts.research.proximal_distal_energy.trajectory_control_authority import (
    ControlScales,
    GuardCrossingConfig,
    refine_guard_crossing,
)
from src.shared.python.simulation_backends import GolfModelParams

FloatArray: TypeAlias = npt.NDArray[np.float64]


def _finite_vector(name: str, value: npt.ArrayLike, *, size: int) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {size} finite values")
    return array


def _control_history(name: str, value: npt.ArrayLike) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if (
        array.ndim != 2
        or array.shape[0] < 1
        or array.shape[1] != 2
        or not np.all(np.isfinite(array))
    ):
        raise ValueError(f"{name} must be a nonempty finite (N, 2) array")
    return array


def _readonly(value: npt.ArrayLike) -> FloatArray:
    array = np.asarray(value, dtype=float).copy()
    if not np.all(np.isfinite(array)):
        raise ValueError("result contains non-finite values")
    array.setflags(write=False)
    return array


class EventReplayStatus(str, Enum):
    """Topology/numerics classification from independent direct replay."""

    TRANSVERSE = "transverse"
    GRAZING = "grazing"
    ABSENT = "absent"
    MULTIPLE = "multiple"
    NUMERICAL_FAILURE = "numerical_failure"


class FeasibilityStatus(str, Enum):
    """Target feasibility classification without conflating failure modes."""

    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    WRONG_CROSSING = "wrong_crossing"
    GRAZING = "grazing"
    NUMERICAL_FAILURE = "numerical_failure"


class ConstraintStatus(str, Enum):
    """Independent perturbation-bound classification."""

    INTERIOR = "interior"
    BOUND_SATURATED = "bound_saturated"
    BOUND_VIOLATION = "bound_violation"


class AuthorityStatus(str, Enum):
    """Whether the declared bounds permit any incremental control."""

    AVAILABLE = "available"
    ZERO_INCREMENTAL_AUTHORITY = "zero_incremental_authority"


@dataclass(frozen=True, slots=True)
class ControlPerturbationBounds:
    """Per-channel amplitude and slew limits for incremental torque.

    Rate is evaluated from a declared zero perturbation immediately before the
    horizon.  Therefore the first sample is part of the slew-rate contract.
    """

    lower_nm: tuple[float, ...]
    upper_nm: tuple[float, ...]
    max_rate_nm_per_s: tuple[float, ...]
    activity_tolerance: float = 1e-9
    violation_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        lower = tuple(float(value) for value in self.lower_nm)
        upper = tuple(float(value) for value in self.upper_nm)
        rate = tuple(float(value) for value in self.max_rate_nm_per_s)
        for name, values in (
            ("lower_nm", lower),
            ("upper_nm", upper),
            ("max_rate_nm_per_s", rate),
        ):
            if len(values) != 2 or not all(math.isfinite(value) for value in values):
                raise ValueError(f"{name} must contain two finite values")
        if any(low > 0.0 or high < 0.0 for low, high in zip(lower, upper, strict=True)):
            raise ValueError("perturbation bounds must contain zero")
        if any(low > high for low, high in zip(lower, upper, strict=True)):
            raise ValueError("lower perturbation bounds cannot exceed upper bounds")
        if any(value < 0.0 for value in rate):
            raise ValueError("maximum perturbation rate must be nonnegative")
        for name in ("activity_tolerance", "violation_tolerance"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        object.__setattr__(self, "lower_nm", lower)
        object.__setattr__(self, "upper_nm", upper)
        object.__setattr__(self, "max_rate_nm_per_s", rate)

    @classmethod
    def zero(cls) -> ControlPerturbationBounds:
        """Return an explicit zero-incremental-authority countermodel."""

        return cls(
            lower_nm=(0.0, 0.0),
            upper_nm=(0.0, 0.0),
            max_rate_nm_per_s=(0.0, 0.0),
        )

    @property
    def lower_array(self) -> FloatArray:
        return np.asarray(self.lower_nm, dtype=float)

    @property
    def upper_array(self) -> FloatArray:
        return np.asarray(self.upper_nm, dtype=float)

    @property
    def rate_array(self) -> FloatArray:
        return np.asarray(self.max_rate_nm_per_s, dtype=float)

    @property
    def is_zero_authority(self) -> bool:
        return self.lower_nm == (0.0, 0.0) and self.upper_nm == (0.0, 0.0)


@dataclass(frozen=True, slots=True)
class EventReplay:
    """Exact direct-replay event record, including unavailable outcomes."""

    status: EventReplayStatus
    crossing_count: int
    time_s: float | None
    state: FloatArray | None
    guard_residual: float | None
    transversality_per_s: float | None
    message: str = ""

    def __post_init__(self) -> None:
        if self.crossing_count < 0:
            raise ValueError("crossing_count must be nonnegative")
        if self.state is not None:
            object.__setattr__(
                self,
                "state",
                _readonly(_finite_vector("event state", self.state, size=4)),
            )
        for name in ("time_s", "guard_residual", "transversality_per_s"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite when available")


@dataclass(frozen=True, slots=True)
class BoundedEventReachabilityProblem:
    """Immutable matched contract for one bounded event target."""

    params: GolfModelParams
    initial_state: tuple[float, ...]
    nominal_controls: FloatArray
    dt_s: float
    state_scales: StateScales
    control_scales: ControlScales
    bounds: ControlPerturbationBounds
    guard: GuardCrossingConfig
    target_event_state: tuple[float, ...]
    tangent_tolerance: float

    def __post_init__(self) -> None:
        initial = tuple(float(value) for value in self.initial_state)
        target = tuple(float(value) for value in self.target_event_state)
        _finite_vector("initial_state", initial, size=4)
        target_array = _finite_vector("target_event_state", target, size=4)
        controls = _control_history("nominal_controls", self.nominal_controls)
        if not math.isfinite(self.dt_s) or self.dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        if not math.isfinite(self.tangent_tolerance) or self.tangent_tolerance < 0.0:
            raise ValueError("tangent_tolerance must be finite and nonnegative")
        guard_residual = float(
            self.guard.gradient_array @ target_array - self.guard.guard_offset
        )
        if abs(guard_residual) > self.guard.guard_tolerance:
            raise ValueError("target_event_state must lie on the guard")
        object.__setattr__(self, "initial_state", initial)
        object.__setattr__(self, "target_event_state", target)
        object.__setattr__(self, "nominal_controls", _readonly(controls))


@dataclass(frozen=True, slots=True)
class BoundedReachabilityOutcome:
    """Reviewer-facing result of one independently replayed candidate."""

    feasibility_status: FeasibilityStatus
    constraint_status: ConstraintStatus
    authority_status: AuthorityStatus
    event: EventReplay | None
    event_tangent_residual: float | None
    full_state_residual: float | None
    scaled_control_energy: float
    peak_scaled_control: float
    amplitude_bound_active: bool
    rate_bound_active: bool
    maximum_amplitude_violation_nm: float
    maximum_rate_violation_nm_per_s: float


@dataclass(frozen=True, slots=True)
class _BoundDiagnostics:
    status: ConstraintStatus
    amplitude_active: bool
    rate_active: bool
    maximum_amplitude_violation_nm: float
    maximum_rate_violation_nm_per_s: float


def replay_guard_event(
    *,
    params: GolfModelParams,
    initial_state: npt.ArrayLike,
    controls: npt.ArrayLike,
    dt_s: float,
    guard: GuardCrossingConfig,
) -> EventReplay:
    """Replay controls and refine the unique declared positive crossing."""

    initial = _finite_vector("initial_state", initial_state, size=4).copy()
    commands = _control_history("controls", controls).copy()
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    try:
        time_s, state = rollout_state_history(
            params,
            initial_state=initial,
            controls=commands,
            dt_s=dt_s,
        )
        crossing = first_positive_crossing(
            time_s,
            state @ guard.gradient_array - guard.guard_offset,
        )
        if crossing.crossing_count == 0:
            return EventReplay(EventReplayStatus.ABSENT, 0, None, None, None, None)
        if crossing.crossing_count != 1 or crossing.sample_index is None:
            return EventReplay(
                EventReplayStatus.MULTIPLE,
                crossing.crossing_count,
                crossing.time_s,
                None,
                None,
                None,
            )
        index = crossing.sample_index
        refined = refine_guard_crossing(
            params=params,
            state_before=state[index],
            control=commands[index],
            time_before_s=float(time_s[index]),
            bracket_dt_s=dt_s,
            config=guard,
        )
    except (ArithmeticError, RuntimeError, ValueError) as exc:
        return EventReplay(
            EventReplayStatus.NUMERICAL_FAILURE,
            0,
            None,
            None,
            None,
            None,
            message=str(exc),
        )
    status = (
        EventReplayStatus.GRAZING
        if refined.status == "near_grazing"
        else EventReplayStatus.TRANSVERSE
    )
    return EventReplay(
        status=status,
        crossing_count=1,
        time_s=refined.time_s,
        state=refined.state,
        guard_residual=refined.guard_residual,
        transversality_per_s=refined.transversality_per_s,
    )


def _bound_diagnostics(
    perturbations: FloatArray,
    *,
    dt_s: float,
    bounds: ControlPerturbationBounds,
) -> _BoundDiagnostics:
    lower = bounds.lower_array
    upper = bounds.upper_array
    rate_limit = bounds.rate_array
    amplitude_excess = np.maximum(
        np.maximum(lower[np.newaxis, :] - perturbations, 0.0),
        np.maximum(perturbations - upper[np.newaxis, :], 0.0),
    )
    rates = np.diff(np.vstack((np.zeros((1, 2)), perturbations)), axis=0) / dt_s
    rate_excess = np.maximum(np.abs(rates) - rate_limit[np.newaxis, :], 0.0)
    amplitude_violation = float(np.max(amplitude_excess))
    rate_violation = float(np.max(rate_excess))
    violation = (
        amplitude_violation > bounds.violation_tolerance
        or rate_violation > bounds.violation_tolerance
    )
    amplitude_active = bool(
        np.any(
            np.isclose(
                perturbations,
                lower[np.newaxis, :],
                rtol=0.0,
                atol=bounds.activity_tolerance,
            )
            | np.isclose(
                perturbations,
                upper[np.newaxis, :],
                rtol=0.0,
                atol=bounds.activity_tolerance,
            )
        )
    )
    rate_active = bool(
        np.any(
            np.isclose(
                np.abs(rates),
                rate_limit[np.newaxis, :],
                rtol=0.0,
                atol=bounds.activity_tolerance,
            )
        )
    )
    status = (
        ConstraintStatus.BOUND_VIOLATION
        if violation
        else (
            ConstraintStatus.BOUND_SATURATED
            if amplitude_active or rate_active
            else ConstraintStatus.INTERIOR
        )
    )
    return _BoundDiagnostics(
        status=status,
        amplitude_active=amplitude_active,
        rate_active=rate_active,
        maximum_amplitude_violation_nm=amplitude_violation,
        maximum_rate_violation_nm_per_s=rate_violation,
    )


def _event_residuals(
    problem: BoundedEventReachabilityProblem,
    event_state: FloatArray,
) -> tuple[float, float]:
    scales = problem.state_scales.array
    scaled_difference = (
        event_state - np.asarray(problem.target_event_state, dtype=float)
    ) / scales
    scaled_gradient = problem.guard.gradient_array * scales
    _, _, right = np.linalg.svd(scaled_gradient.reshape(1, -1), full_matrices=True)
    tangent_basis = right[1:].T
    tangent_residual = float(np.linalg.norm(tangent_basis.T @ scaled_difference))
    return tangent_residual, float(np.linalg.norm(scaled_difference))


def evaluate_bounded_candidate(
    problem: BoundedEventReachabilityProblem,
    perturbations: npt.ArrayLike,
) -> BoundedReachabilityOutcome:
    """Classify one bounded finite perturbation through independent replay."""

    delta = _control_history("perturbations", perturbations).copy()
    if delta.shape != problem.nominal_controls.shape:
        raise ValueError("perturbations must match nominal_controls shape")
    diagnostics = _bound_diagnostics(delta, dt_s=problem.dt_s, bounds=problem.bounds)
    scaled_delta = delta / problem.control_scales.array[np.newaxis, :]
    energy = float(problem.dt_s * np.sum(np.square(scaled_delta)))
    peak = float(np.max(np.abs(scaled_delta)))
    authority = (
        AuthorityStatus.ZERO_INCREMENTAL_AUTHORITY
        if problem.bounds.is_zero_authority
        else AuthorityStatus.AVAILABLE
    )

    def outcome(
        feasibility: FeasibilityStatus,
        *,
        event: EventReplay | None = None,
        tangent_residual: float | None = None,
        full_residual: float | None = None,
    ) -> BoundedReachabilityOutcome:
        return BoundedReachabilityOutcome(
            feasibility_status=feasibility,
            constraint_status=diagnostics.status,
            authority_status=authority,
            event=event,
            event_tangent_residual=tangent_residual,
            full_state_residual=full_residual,
            scaled_control_energy=energy,
            peak_scaled_control=peak,
            amplitude_bound_active=diagnostics.amplitude_active,
            rate_bound_active=diagnostics.rate_active,
            maximum_amplitude_violation_nm=(diagnostics.maximum_amplitude_violation_nm),
            maximum_rate_violation_nm_per_s=(
                diagnostics.maximum_rate_violation_nm_per_s
            ),
        )

    if diagnostics.status is ConstraintStatus.BOUND_VIOLATION:
        return outcome(FeasibilityStatus.INFEASIBLE)
    event = replay_guard_event(
        params=problem.params,
        initial_state=problem.initial_state,
        controls=problem.nominal_controls + delta,
        dt_s=problem.dt_s,
        guard=problem.guard,
    )
    if event.status is EventReplayStatus.NUMERICAL_FAILURE:
        return outcome(FeasibilityStatus.NUMERICAL_FAILURE, event=event)
    if event.status in (EventReplayStatus.ABSENT, EventReplayStatus.MULTIPLE):
        return outcome(FeasibilityStatus.WRONG_CROSSING, event=event)
    if event.status is EventReplayStatus.GRAZING:
        return outcome(FeasibilityStatus.GRAZING, event=event)
    if event.state is None:
        raise AssertionError("transverse replay must provide an event state")
    tangent_residual, full_residual = _event_residuals(problem, event.state)
    feasibility = (
        FeasibilityStatus.FEASIBLE
        if tangent_residual <= problem.tangent_tolerance
        else FeasibilityStatus.INFEASIBLE
    )
    return outcome(
        feasibility,
        event=event,
        tangent_residual=tangent_residual,
        full_residual=full_residual,
    )


__all__ = [
    "AuthorityStatus",
    "BoundedEventReachabilityProblem",
    "BoundedReachabilityOutcome",
    "ConstraintStatus",
    "ControlPerturbationBounds",
    "EventReplay",
    "EventReplayStatus",
    "FeasibilityStatus",
    "evaluate_bounded_candidate",
    "replay_guard_event",
]
