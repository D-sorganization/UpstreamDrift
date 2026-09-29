"""Locate articulated contact events on a declared discrete state path."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class ContactEventKind(str, Enum):
    """Supported active-set transitions for a tension interface."""

    OPENING = "opening"
    REATTACHMENT = "reattachment"


@dataclass(frozen=True, slots=True)
class ContactEventLocationConfig:
    """Numerical and trace-consistency contract for root location."""

    gap_tolerance_m: float = 1.0e-10
    time_tolerance_s: float = 1.0e-12
    max_iterations: int = 80
    validate_active_gap_consistency: bool = True

    def __post_init__(self) -> None:
        if not np.isfinite(self.gap_tolerance_m) or self.gap_tolerance_m <= 0.0:
            raise ValueError("gap_tolerance_m must be finite and positive")
        if not np.isfinite(self.time_tolerance_s) or self.time_tolerance_s <= 0.0:
            raise ValueError("time_tolerance_s must be finite and positive")
        if not isinstance(self.max_iterations, int) or self.max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer")
        if not isinstance(self.validate_active_gap_consistency, bool):
            raise TypeError("validate_active_gap_consistency must be a bool")


@dataclass(frozen=True, slots=True)
class ContactEventRecord:
    """One root located on the registered linear state interpolant."""

    kind: ContactEventKind
    time_s: float
    left_index: int
    right_index: int
    hand_index: int
    station_index: int
    position: FloatArray
    velocity: FloatArray
    gap_residual_m: float
    final_bracket_width_s: float
    path_model: str = "linear_state_interpolant"


@dataclass(frozen=True, slots=True)
class EventAlignedStateTrace:
    """State samples with one duplicate-time boundary per event group."""

    time_s: FloatArray
    positions: FloatArray
    velocities: FloatArray
    segment_ids: NDArray[np.int64]
    event_record_offsets: NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class _ValidatedEventTrace:
    time: FloatArray
    positions: FloatArray
    velocities: FloatArray
    gaps: FloatArray
    active: NDArray[np.bool_]


@dataclass(slots=True)
class _AlignmentBuffers:
    time: list[float]
    positions: list[FloatArray]
    velocities: list[FloatArray]
    segments: list[int]
    event_offsets: list[int]
    segment: int = 0


def _finite(name: str, value: object, *, ndim: int) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    return array


def _validate_event_trace(
    time_s: FloatArray,
    positions: FloatArray,
    velocities: FloatArray,
    station_signed_gap_m: FloatArray,
    station_active: NDArray[np.bool_],
    gap_evaluator: Callable[[FloatArray], FloatArray],
    config: ContactEventLocationConfig,
) -> _ValidatedEventTrace:
    if not isinstance(config, ContactEventLocationConfig):
        raise TypeError("config must be a ContactEventLocationConfig")
    time = _finite("time_s", time_s, ndim=1)
    q = _finite("positions", positions, ndim=2)
    qd = _finite("velocities", velocities, ndim=2)
    gaps = _finite("station_signed_gap_m", station_signed_gap_m, ndim=3)
    active = np.asarray(station_active, dtype=bool)
    if time.size < 2 or np.any(np.diff(time) <= 0.0):
        raise ValueError("time_s must contain at least two strictly increasing samples")
    if q.shape != qd.shape or q.shape[0] != time.size:
        raise ValueError(
            "positions and velocities must share shape (samples, coordinates)"
        )
    if gaps.shape[0] != time.size or active.shape != gaps.shape:
        raise ValueError(
            "gap and active arrays must share shape (samples, hands, stations)"
        )
    if not callable(gap_evaluator):
        raise TypeError("gap_evaluator must be callable")
    if config.validate_active_gap_consistency and not np.array_equal(
        active, gaps > 0.0
    ):
        raise ValueError("active state must equal the positive signed-gap state")
    return _ValidatedEventTrace(time, q, qd, gaps, active)


def _evaluate_station_gap(
    gap_evaluator: Callable[[FloatArray], FloatArray],
    position: FloatArray,
    expected_shape: tuple[int, int],
    hand_index: int,
    station_index: int,
) -> float:
    evaluated = np.asarray(gap_evaluator(position), dtype=np.float64)
    if evaluated.shape != expected_shape or not np.all(np.isfinite(evaluated)):
        raise ValueError("gap_evaluator returned an invalid station-gap array")
    return float(evaluated[hand_index, station_index])


def _root_fraction(
    evaluate: Callable[[float], float],
    left_gap: float,
    right_gap: float,
    config: ContactEventLocationConfig,
    fraction_tolerance: float,
) -> tuple[float, float, float]:
    if abs(left_gap) <= config.gap_tolerance_m:
        return 0.0, left_gap, 0.0
    if abs(right_gap) <= config.gap_tolerance_m:
        return 1.0, right_gap, 0.0
    if np.signbit(left_gap) == np.signbit(right_gap):
        raise ValueError("active-set transition does not bracket a signed-gap root")
    lower, upper = 0.0, 1.0
    f_lower = left_gap
    midpoint = 0.5
    f_midpoint = evaluate(midpoint)
    for _ in range(config.max_iterations):
        midpoint = 0.5 * (lower + upper)
        f_midpoint = evaluate(midpoint)
        if (
            abs(f_midpoint) <= config.gap_tolerance_m
            and upper - lower <= fraction_tolerance
        ):
            return midpoint, f_midpoint, upper - lower
        if np.signbit(f_midpoint) == np.signbit(f_lower):
            lower, f_lower = midpoint, f_midpoint
        else:
            upper = midpoint
    raise ValueError("contact-event root location did not converge")


def _locate_transition(
    trace: _ValidatedEventTrace,
    gap_evaluator: Callable[[FloatArray], FloatArray],
    indices: tuple[int, int, int],
    config: ContactEventLocationConfig,
) -> ContactEventRecord:
    left_index, hand_index, station_index = indices
    right_index = left_index + 1
    q_left, q_right = trace.positions[left_index], trace.positions[right_index]
    qd_left, qd_right = trace.velocities[left_index], trace.velocities[right_index]
    delta_time = float(trace.time[right_index] - trace.time[left_index])
    expected_shape = (trace.gaps.shape[1], trace.gaps.shape[2])

    def evaluate_fraction(fraction: float) -> float:
        position = q_left + fraction * (q_right - q_left)
        return _evaluate_station_gap(
            gap_evaluator,
            position,
            expected_shape,
            hand_index,
            station_index,
        )

    left_gap = evaluate_fraction(0.0)
    right_gap = evaluate_fraction(1.0)
    retained = trace.gaps[:, hand_index, station_index]
    endpoints_match = np.isclose(
        left_gap,
        retained[left_index],
        rtol=0.0,
        atol=config.gap_tolerance_m,
    ) and np.isclose(
        right_gap,
        retained[right_index],
        rtol=0.0,
        atol=config.gap_tolerance_m,
    )
    if not endpoints_match:
        raise ValueError("gap_evaluator endpoints disagree with the retained trace")
    fraction, residual, fraction_width = _root_fraction(
        evaluate_fraction,
        left_gap,
        right_gap,
        config,
        config.time_tolerance_s / delta_time,
    )
    return ContactEventRecord(
        kind=(
            ContactEventKind.OPENING
            if trace.active[left_index, hand_index, station_index]
            else ContactEventKind.REATTACHMENT
        ),
        time_s=float(trace.time[left_index] + fraction * delta_time),
        left_index=left_index,
        right_index=right_index,
        hand_index=hand_index,
        station_index=station_index,
        position=q_left + fraction * (q_right - q_left),
        velocity=qd_left + fraction * (qd_right - qd_left),
        gap_residual_m=residual,
        final_bracket_width_s=fraction_width * delta_time,
    )


def locate_contact_events(
    *,
    time_s: FloatArray,
    positions: FloatArray,
    velocities: FloatArray,
    station_signed_gap_m: FloatArray,
    station_active: NDArray[np.bool_],
    gap_evaluator: Callable[[FloatArray], FloatArray],
    config: ContactEventLocationConfig = ContactEventLocationConfig(),
) -> tuple[ContactEventRecord, ...]:
    """Locate sampled opening/reattachment roots on a linear state path.

    This qualifies event time on the declared interpolation of the retained
    discrete trajectory. It does not claim to recover the continuous
    integrator's exact event solution.
    """

    trace = _validate_event_trace(
        time_s,
        positions,
        velocities,
        station_signed_gap_m,
        station_active,
        gap_evaluator,
        config,
    )
    transition_indices = np.argwhere(trace.active[1:] != trace.active[:-1])
    records: list[ContactEventRecord] = []
    for indices in transition_indices:
        transition = (int(indices[0]), int(indices[1]), int(indices[2]))
        records.append(_locate_transition(trace, gap_evaluator, transition, config))
    return tuple(
        sorted(
            records,
            key=lambda record: (
                record.time_s,
                record.hand_index,
                record.station_index,
            ),
        )
    )


def _validate_alignment_inputs(
    time_s: FloatArray,
    positions: FloatArray,
    velocities: FloatArray,
    events: tuple[ContactEventRecord, ...],
    equality_tolerance: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    time = _finite("time_s", time_s, ndim=1)
    q = _finite("positions", positions, ndim=2)
    qd = _finite("velocities", velocities, ndim=2)
    if time.size < 2 or np.any(np.diff(time) <= 0.0):
        raise ValueError("time_s must contain at least two strictly increasing samples")
    if q.shape != qd.shape or q.shape[0] != time.size:
        raise ValueError(
            "positions and velocities must share shape (samples, coordinates)"
        )
    if not np.isfinite(equality_tolerance) or equality_tolerance <= 0.0:
        raise ValueError("equality_tolerance must be finite and positive")
    if any(not isinstance(event, ContactEventRecord) for event in events):
        raise TypeError("events must contain only ContactEventRecord values")
    ordered = tuple(
        sorted(
            events,
            key=lambda event: (
                event.left_index,
                event.time_s,
                event.hand_index,
                event.station_index,
            ),
        )
    )
    if ordered != events:
        raise ValueError("events must be sorted in path order")
    return time, q, qd


def _group_simultaneous_events(
    interval_events: list[tuple[int, ContactEventRecord]], tolerance: float
) -> tuple[tuple[int, tuple[ContactEventRecord, ...]], ...]:
    groups: list[tuple[int, tuple[ContactEventRecord, ...]]] = []
    cursor = 0
    while cursor < len(interval_events):
        offset, first = interval_events[cursor]
        members = [first]
        cursor += 1
        while cursor < len(interval_events) and np.isclose(
            interval_events[cursor][1].time_s,
            first.time_s,
            rtol=0.0,
            atol=tolerance,
        ):
            members.append(interval_events[cursor][1])
            cursor += 1
        groups.append((offset, tuple(members)))
    return tuple(groups)


def _append_event_group(
    buffers: _AlignmentBuffers,
    offset: int,
    group: tuple[ContactEventRecord, ...],
    tolerance: float,
) -> None:
    first = group[0]
    for member in group[1:]:
        states_agree = np.allclose(
            member.position, first.position, rtol=0.0, atol=tolerance
        ) and np.allclose(member.velocity, first.velocity, rtol=0.0, atol=tolerance)
        if not states_agree:
            raise ValueError("simultaneous event records disagree on state")
    if not np.isclose(buffers.time[-1], first.time_s, rtol=0.0, atol=tolerance):
        buffers.time.append(first.time_s)
        buffers.positions.append(first.position.copy())
        buffers.velocities.append(first.velocity.copy())
        buffers.segments.append(buffers.segment)
    elif not np.allclose(
        buffers.positions[-1], first.position, rtol=0.0, atol=tolerance
    ):
        raise ValueError("event state disagrees with coincident trace sample")
    buffers.segment += 1
    buffers.time.append(first.time_s)
    buffers.positions.append(first.position.copy())
    buffers.velocities.append(first.velocity.copy())
    buffers.segments.append(buffers.segment)
    buffers.event_offsets.append(offset)


def _append_trace_sample(
    buffers: _AlignmentBuffers,
    time_s: float,
    position: FloatArray,
    velocity: FloatArray,
    tolerance: float,
) -> None:
    coincident = np.isclose(buffers.time[-1], time_s, rtol=0.0, atol=tolerance)
    same_position = np.allclose(
        buffers.positions[-1], position, rtol=0.0, atol=tolerance
    )
    if not (coincident and same_position):
        buffers.time.append(float(time_s))
        buffers.positions.append(position.copy())
        buffers.velocities.append(velocity.copy())
        buffers.segments.append(buffers.segment)


def align_state_trace_to_events(
    *,
    time_s: FloatArray,
    positions: FloatArray,
    velocities: FloatArray,
    events: tuple[ContactEventRecord, ...],
    equality_tolerance: float = 1.0e-12,
) -> EventAlignedStateTrace:
    """Insert duplicate pre/post samples so quadrature never crosses an event."""

    time, q, qd = _validate_alignment_inputs(
        time_s, positions, velocities, events, equality_tolerance
    )
    buffers = _AlignmentBuffers(
        time=[float(time[0])],
        positions=[q[0].copy()],
        velocities=[qd[0].copy()],
        segments=[0],
        event_offsets=[],
    )
    event_cursor = 0
    for left_index in range(time.size - 1):
        interval_events: list[tuple[int, ContactEventRecord]] = []
        while (
            event_cursor < len(events) and events[event_cursor].left_index == left_index
        ):
            event = events[event_cursor]
            if event.right_index != left_index + 1:
                raise ValueError("event indices must name adjacent trace samples")
            if not time[left_index] <= event.time_s <= time[left_index + 1]:
                raise ValueError("event time lies outside its retained bracket")
            interval_events.append((event_cursor, event))
            event_cursor += 1
        for offset, group in _group_simultaneous_events(
            interval_events, equality_tolerance
        ):
            _append_event_group(buffers, offset, group, equality_tolerance)
        _append_trace_sample(
            buffers,
            float(time[left_index + 1]),
            q[left_index + 1],
            qd[left_index + 1],
            equality_tolerance,
        )
    if event_cursor != len(events):
        raise ValueError("an event does not belong to a retained trace interval")
    return EventAlignedStateTrace(
        time_s=np.asarray(buffers.time, dtype=np.float64),
        positions=np.asarray(buffers.positions, dtype=np.float64),
        velocities=np.asarray(buffers.velocities, dtype=np.float64),
        segment_ids=np.asarray(buffers.segments, dtype=np.int64),
        event_record_offsets=np.asarray(buffers.event_offsets, dtype=np.int64),
    )


__all__ = [
    "ContactEventKind",
    "ContactEventLocationConfig",
    "ContactEventRecord",
    "EventAlignedStateTrace",
    "align_state_trace_to_events",
    "locate_contact_events",
]
