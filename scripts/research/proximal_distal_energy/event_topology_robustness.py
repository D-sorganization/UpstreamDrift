"""Global event-topology contracts for proximal-distal robustness studies.

The #9124 solver intentionally qualified one positive crossing inside one
registered bracket.  This module removes that local assumption before delay or
noise is introduced: every crossing on a declared rollout is retained with its
direction and transversality.  Results remain synthetic model evidence, not
human motor-variability or coaching evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from scripts.research.proximal_distal_energy.phase_event_stability import (
    rollout_state_history,
)
from scripts.research.proximal_distal_energy.trajectory_control_authority import (
    GuardCrossingConfig,
    refine_guard_crossing,
)
from src.shared.python.simulation_backends import GolfModelParams

FloatArray: TypeAlias = npt.NDArray[np.float64]


def _finite_vector(name: str, value: npt.ArrayLike) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector")
    return array


class CrossingDirection(str, Enum):
    """Orientation of a guard crossing in the declared guard coordinates."""

    POSITIVE = "negative_to_nonnegative"
    NEGATIVE = "positive_to_nonpositive"


class EventTopologyStatus(str, Enum):
    """Fail-closed global event classification."""

    ABSENT = "absent"
    UNIQUE_TRANSVERSE = "unique_transverse"
    MULTIPLE = "multiple"
    GRAZING = "grazing"
    INITIAL_ON_GUARD = "initial_on_guard"
    NUMERICAL_FAILURE = "numerical_failure"


class DelayInterpolation(str, Enum):
    """Declared interpretation of the sampled open-loop command history."""

    ZERO_ORDER_HOLD = "zero_order_hold"
    LINEAR_NODAL = "linear_nodal"


@dataclass(frozen=True, slots=True)
class CommandDelayConfig:
    """Causal command-delay policy for a predetermined sampled control.

    ``prehistory_control`` applies at every negative source time. Linear-nodal
    interpolation is only used at nonnegative source times; it therefore does
    not invent an interpolation ramp across the start of the registered
    horizon.
    """

    delay_s: float
    interpolation: DelayInterpolation = DelayInterpolation.ZERO_ORDER_HOLD
    prehistory_control: tuple[float, ...] = (0.0, 0.0)
    posthistory_control: tuple[float, ...] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if not math.isfinite(self.delay_s) or self.delay_s < 0.0:
            raise ValueError("delay_s must be finite and nonnegative")
        if not isinstance(self.interpolation, DelayInterpolation):
            raise ValueError("interpolation must be a DelayInterpolation")
        for name in ("prehistory_control", "posthistory_control"):
            history = tuple(float(value) for value in getattr(self, name))
            if len(history) != 2 or not all(math.isfinite(value) for value in history):
                raise ValueError(f"{name} must contain two finite values")
            object.__setattr__(self, name, history)


@dataclass(frozen=True, slots=True)
class CrossingBracket:
    """One sampled sign-change bracket and its declared direction."""

    sample_index: int
    direction: CrossingDirection

    def __post_init__(self) -> None:
        if self.sample_index < 0:
            raise ValueError("sample_index must be nonnegative")


@dataclass(frozen=True, slots=True)
class CrossingBracketEnumeration:
    """All sampled crossing brackets, including an initial-guard ambiguity."""

    brackets: tuple[CrossingBracket, ...]
    initial_on_guard: bool


@dataclass(frozen=True, slots=True)
class GlobalGuardEvent:
    """One exact-step refined event on the global rollout."""

    direction: CrossingDirection
    sample_index: int
    time_s: float
    state: FloatArray
    guard_residual: float
    transversality_per_s: float
    near_grazing: bool

    def __post_init__(self) -> None:
        if self.sample_index < 0:
            raise ValueError("sample_index must be nonnegative")
        scalars = (self.time_s, self.guard_residual, self.transversality_per_s)
        if not all(math.isfinite(value) for value in scalars):
            raise ValueError("event scalars must be finite")
        state = _finite_vector("state", self.state).copy()
        if state.shape != (4,):
            raise ValueError("state must contain four values")
        state.setflags(write=False)
        object.__setattr__(self, "state", state)


@dataclass(frozen=True, slots=True)
class GlobalEventTopology:
    """Complete event topology over one declared global rollout horizon."""

    status: EventTopologyStatus
    events: tuple[GlobalGuardEvent, ...]
    message: str = ""

    @property
    def crossing_count(self) -> int:
        return len(self.events)


@dataclass(frozen=True, slots=True)
class DelayContinuationConfig:
    """Immutable physical-delay schedule on one common global horizon."""

    delays_s: tuple[float, ...]
    common_horizon_s: float
    interpolation: DelayInterpolation = DelayInterpolation.LINEAR_NODAL
    prehistory_control: tuple[float, ...] = (0.0, 0.0)
    posthistory_control: tuple[float, ...] = (0.0, 0.0)
    grid_tolerance_s: float = 1e-12

    def __post_init__(self) -> None:
        delays = tuple(float(value) for value in self.delays_s)
        if (
            not delays
            or delays[0] != 0.0
            or not all(math.isfinite(value) and value >= 0.0 for value in delays)
            or any(
                right <= left
                for left, right in zip(delays[:-1], delays[1:], strict=True)
            )
        ):
            raise ValueError("delays_s must start at zero and strictly increase")
        if not math.isfinite(self.common_horizon_s) or self.common_horizon_s <= 0.0:
            raise ValueError("common_horizon_s must be finite and positive")
        if not math.isfinite(self.grid_tolerance_s) or self.grid_tolerance_s <= 0.0:
            raise ValueError("grid_tolerance_s must be finite and positive")
        history_contract = CommandDelayConfig(
            delay_s=0.0,
            interpolation=self.interpolation,
            prehistory_control=self.prehistory_control,
            posthistory_control=self.posthistory_control,
        )
        object.__setattr__(self, "delays_s", delays)
        object.__setattr__(
            self, "prehistory_control", history_contract.prehistory_control
        )
        object.__setattr__(
            self, "posthistory_control", history_contract.posthistory_control
        )


@dataclass(frozen=True, slots=True)
class DelayContinuationOutcome:
    """Global topology retained at one declared physical command delay."""

    delay_s: float
    topology: GlobalEventTopology


@dataclass(frozen=True, slots=True)
class DelayContinuationResult:
    """Matched delay outcomes and the independent zero-delay replay gate."""

    outcomes: tuple[DelayContinuationOutcome, ...]
    output_sample_count: int
    zero_delay_control_residual: float


def apply_command_delay(
    controls: npt.ArrayLike,
    *,
    dt_s: float,
    config: CommandDelayConfig,
    output_sample_count: int | None = None,
) -> FloatArray:
    """Delay a sampled open-loop command using the declared interpolation.

    The input samples are interpreted at ``t = k * dt_s``. The output at that
    same node evaluates the declared command at ``t - delay_s``. Source times
    before zero use the explicit prehistory vector. Source times at or beyond
    the end of the final sample interval use the explicit posthistory vector.
    """

    commands = np.asarray(controls, dtype=float)
    if (
        commands.ndim != 2
        or commands.shape[0] < 1
        or commands.shape[1] != 2
        or not np.all(np.isfinite(commands))
    ):
        raise ValueError("controls must be a nonempty finite (N, 2) array")
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    output_count = (
        commands.shape[0] if output_sample_count is None else output_sample_count
    )
    if (
        isinstance(output_count, bool)
        or not isinstance(output_count, int)
        or output_count < 1
    ):
        raise ValueError("output_sample_count must be a positive integer")
    delayed = np.empty((output_count, 2), dtype=float)
    prehistory = np.asarray(config.prehistory_control, dtype=float)
    posthistory = np.asarray(config.posthistory_control, dtype=float)
    ratio = config.delay_s / dt_s
    tolerance = 16.0 * np.finfo(float).eps * max(1.0, ratio, output_count)
    for output_index in range(output_count):
        source_position = output_index - ratio
        if source_position < -tolerance:
            delayed[output_index] = prehistory
            continue
        source_position = max(0.0, source_position)
        if source_position >= commands.shape[0] - tolerance:
            delayed[output_index] = posthistory
            continue
        lower = int(math.floor(source_position + tolerance))
        if config.interpolation is DelayInterpolation.ZERO_ORDER_HOLD:
            delayed[output_index] = commands[lower]
            continue
        upper = lower + 1
        fraction = min(max(source_position - lower, 0.0), 1.0)
        upper_value = commands[upper] if upper < commands.shape[0] else posthistory
        delayed[output_index] = (1.0 - fraction) * commands[
            lower
        ] + fraction * upper_value
    delayed.setflags(write=False)
    return delayed


def evaluate_delay_continuation(
    *,
    params: GolfModelParams,
    initial_state: npt.ArrayLike,
    controls: npt.ArrayLike,
    dt_s: float,
    guard: GuardCrossingConfig,
    config: DelayContinuationConfig,
) -> DelayContinuationResult:
    """Evaluate a preregistered delay schedule on an identical global horizon."""

    commands = np.asarray(controls, dtype=float)
    if (
        commands.ndim != 2
        or commands.shape[0] < 1
        or commands.shape[1] != 2
        or not np.all(np.isfinite(commands))
    ):
        raise ValueError("controls must be a nonempty finite (N, 2) array")
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    sample_count_float = config.common_horizon_s / dt_s
    output_count = int(round(sample_count_float))
    if abs(output_count * dt_s - config.common_horizon_s) > config.grid_tolerance_s:
        raise ValueError("common_horizon_s must lie on the integration grid")
    minimum_horizon = commands.shape[0] * dt_s + config.delays_s[-1]
    if config.common_horizon_s + config.grid_tolerance_s < minimum_horizon:
        raise ValueError(
            "common horizon must retain the complete delayed command program"
        )

    reference = np.repeat(
        np.asarray(config.posthistory_control, dtype=float)[np.newaxis, :],
        output_count,
        axis=0,
    )
    reference[: commands.shape[0]] = commands
    outcomes: list[DelayContinuationOutcome] = []
    zero_residual = math.inf
    for delay_s in config.delays_s:
        delayed = apply_command_delay(
            commands,
            dt_s=dt_s,
            output_sample_count=output_count,
            config=CommandDelayConfig(
                delay_s=delay_s,
                interpolation=config.interpolation,
                prehistory_control=config.prehistory_control,
                posthistory_control=config.posthistory_control,
            ),
        )
        if delay_s == 0.0:
            zero_residual = float(np.max(np.abs(delayed - reference)))
        topology = replay_global_event_topology(
            params=params,
            initial_state=initial_state,
            controls=delayed,
            dt_s=dt_s,
            guard=guard,
        )
        outcomes.append(DelayContinuationOutcome(delay_s, topology))
    return DelayContinuationResult(tuple(outcomes), output_count, zero_residual)


def enumerate_crossing_brackets(
    time_s: npt.ArrayLike,
    guard_values: npt.ArrayLike,
    *,
    zero_tolerance: float,
) -> CrossingBracketEnumeration:
    """Enumerate all directed sampled sign changes without selecting an event.

    An exact zero sample is assigned to the arriving bracket once.  A rollout
    beginning on the guard is instead ambiguous because no arrival direction
    exists; callers must handle it explicitly rather than silently skipping it.
    """

    times = _finite_vector("time_s", time_s)
    values = _finite_vector("guard_values", guard_values)
    if times.size != values.size or times.size < 2:
        raise ValueError("time_s and guard_values must share at least two samples")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("time_s must be strictly increasing")
    if not math.isfinite(zero_tolerance) or zero_tolerance <= 0.0:
        raise ValueError("zero_tolerance must be finite and positive")
    if abs(float(values[0])) <= zero_tolerance:
        return CrossingBracketEnumeration((), True)

    brackets: list[CrossingBracket] = []
    for index, (left, right) in enumerate(zip(values[:-1], values[1:], strict=True)):
        if left < 0.0 <= right:
            direction = CrossingDirection.POSITIVE
        elif left > 0.0 >= right:
            direction = CrossingDirection.NEGATIVE
        else:
            continue
        brackets.append(CrossingBracket(index, direction))
    return CrossingBracketEnumeration(tuple(brackets), False)


def _oriented_guard(
    guard: GuardCrossingConfig, direction: CrossingDirection
) -> GuardCrossingConfig:
    if direction is CrossingDirection.POSITIVE:
        return guard
    return GuardCrossingConfig(
        guard_gradient=tuple(-value for value in guard.guard_gradient),
        guard_offset=-guard.guard_offset,
        guard_tolerance=guard.guard_tolerance,
        time_tolerance_s=guard.time_tolerance_s,
        transversality_threshold=guard.transversality_threshold,
        max_iterations=guard.max_iterations,
    )


def replay_global_event_topology(
    *,
    params: GolfModelParams,
    initial_state: npt.ArrayLike,
    controls: npt.ArrayLike,
    dt_s: float,
    guard: GuardCrossingConfig,
) -> GlobalEventTopology:
    """Retain and refine every directed guard crossing on a rollout."""

    initial = _finite_vector("initial_state", initial_state)
    commands = np.asarray(controls, dtype=float)
    if initial.shape != (4,):
        raise ValueError("initial_state must contain four values")
    if (
        commands.ndim != 2
        or commands.shape[0] < 1
        or commands.shape[1] != 2
        or not np.all(np.isfinite(commands))
    ):
        raise ValueError("controls must be a nonempty finite (N, 2) array")
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")

    try:
        times, states = rollout_state_history(
            params,
            initial_state=initial,
            controls=commands,
            dt_s=dt_s,
        )
        guard_values = states @ guard.gradient_array - guard.guard_offset
        enumeration = enumerate_crossing_brackets(
            times,
            guard_values,
            zero_tolerance=guard.guard_tolerance,
        )
        if enumeration.initial_on_guard:
            return GlobalEventTopology(EventTopologyStatus.INITIAL_ON_GUARD, ())

        events: list[GlobalGuardEvent] = []
        for bracket in enumeration.brackets:
            oriented = _oriented_guard(guard, bracket.direction)
            refined = refine_guard_crossing(
                params=params,
                state_before=states[bracket.sample_index],
                control=commands[bracket.sample_index],
                time_before_s=float(times[bracket.sample_index]),
                bracket_dt_s=dt_s,
                config=oriented,
            )
            sign = 1.0 if bracket.direction is CrossingDirection.POSITIVE else -1.0
            events.append(
                GlobalGuardEvent(
                    direction=bracket.direction,
                    sample_index=bracket.sample_index,
                    time_s=refined.time_s,
                    state=refined.state,
                    guard_residual=sign * refined.guard_residual,
                    transversality_per_s=(sign * refined.transversality_per_s),
                    near_grazing=refined.status == "near_grazing",
                )
            )
    except (ArithmeticError, RuntimeError, ValueError) as exc:
        return GlobalEventTopology(
            EventTopologyStatus.NUMERICAL_FAILURE,
            (),
            message=str(exc),
        )

    retained = tuple(events)
    if not retained:
        status = EventTopologyStatus.ABSENT
    elif len(retained) > 1:
        status = EventTopologyStatus.MULTIPLE
    elif retained[0].near_grazing:
        status = EventTopologyStatus.GRAZING
    else:
        status = EventTopologyStatus.UNIQUE_TRANSVERSE
    return GlobalEventTopology(status, retained)


__all__ = [
    "CommandDelayConfig",
    "CrossingBracket",
    "CrossingBracketEnumeration",
    "CrossingDirection",
    "DelayContinuationConfig",
    "DelayContinuationOutcome",
    "DelayContinuationResult",
    "DelayInterpolation",
    "EventTopologyStatus",
    "GlobalEventTopology",
    "GlobalGuardEvent",
    "apply_command_delay",
    "enumerate_crossing_brackets",
    "evaluate_delay_continuation",
    "replay_global_event_topology",
]
