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


class FrictionEventKind(str, Enum):
    """Transitions into or out of the regularized Coulomb force limit."""

    LIMIT_ENTRY = "friction_limit_entry"
    LIMIT_EXIT = "friction_limit_exit"


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
class FrictionEventRecord:
    """One Coulomb-limit root; this is not evidence of static sticking."""

    kind: FrictionEventKind
    time_s: float
    left_index: int
    right_index: int
    hand_index: int
    station_index: int
    position: FloatArray
    velocity: FloatArray
    friction_margin_residual_n: float
    final_bracket_width_s: float
    path_model: str = "linear_state_interpolant"
    static_stick_modeled: bool = False


EventRecord = ContactEventRecord | FrictionEventRecord


@dataclass(frozen=True, slots=True)
class EventAlignedStateTrace:
    """State samples with one duplicate-time boundary per event group."""

    time_s: FloatArray
    positions: FloatArray
    velocities: FloatArray
    segment_ids: NDArray[np.int64]
    event_record_offsets: NDArray[np.int64]


def _finite(name: str, value: object, *, ndim: int) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    return array


def _root_fraction(
    evaluate: Callable[[float], float],
    left_gap: float,
    right_gap: float,
    *,
    gap_tolerance_m: float,
    fraction_tolerance: float,
    max_iterations: int,
) -> tuple[float, float, float]:
    if abs(left_gap) <= gap_tolerance_m:
        return 0.0, left_gap, 0.0
    if abs(right_gap) <= gap_tolerance_m:
        return 1.0, right_gap, 0.0
    if np.signbit(left_gap) == np.signbit(right_gap):
        raise ValueError("active-set transition does not bracket a scalar root")
    lower, upper = 0.0, 1.0
    f_lower = left_gap
    midpoint = 0.5
    f_midpoint = evaluate(midpoint)
    for _ in range(max_iterations):
        midpoint = 0.5 * (lower + upper)
        f_midpoint = evaluate(midpoint)
        if abs(f_midpoint) <= gap_tolerance_m and upper - lower <= fraction_tolerance:
            return midpoint, f_midpoint, upper - lower
        if np.signbit(f_midpoint) == np.signbit(f_lower):
            lower, f_lower = midpoint, f_midpoint
        else:
            upper = midpoint
    raise ValueError("contact-event root location did not converge")


def locate_contact_events(
    *,
    time_s: FloatArray,
    positions: FloatArray,
    velocities: FloatArray,
    station_signed_gap_m: FloatArray,
    station_active: NDArray[np.bool_],
    gap_evaluator: Callable[[FloatArray], FloatArray],
    gap_tolerance_m: float = 1.0e-10,
    time_tolerance_s: float = 1.0e-12,
    max_iterations: int = 80,
    validate_active_gap_consistency: bool = True,
) -> tuple[ContactEventRecord, ...]:
    """Locate every sampled opening/reattachment on a linear state path.

    This qualifies event time on the declared interpolation of the existing
    discrete trajectory. It does not claim to be the continuous integrator's
    exact event solution, and the path model remains explicit in every record.
    """

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
    if not np.isfinite(gap_tolerance_m) or gap_tolerance_m <= 0.0:
        raise ValueError("gap_tolerance_m must be finite and positive")
    if not np.isfinite(time_tolerance_s) or time_tolerance_s <= 0.0:
        raise ValueError("time_tolerance_s must be finite and positive")
    if not isinstance(max_iterations, int) or max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer")
    if validate_active_gap_consistency and not np.array_equal(active, gaps > 0.0):
        raise ValueError("active state must equal the positive signed-gap state")

    transitions = active[1:] != active[:-1]
    records: list[ContactEventRecord] = []
    for left_index, hand_index, station_index in np.argwhere(transitions):
        right_index = int(left_index + 1)
        left_index = int(left_index)
        hand_index = int(hand_index)
        station_index = int(station_index)
        q_left, q_right = q[left_index], q[right_index]
        qd_left, qd_right = qd[left_index], qd[right_index]
        delta_time = float(time[right_index] - time[left_index])

        def evaluate_fraction(
            fraction: float,
            q_start: FloatArray = q_left,
            q_end: FloatArray = q_right,
            hand: int = hand_index,
            station: int = station_index,
        ) -> float:
            position = q_start + fraction * (q_end - q_start)
            evaluated = np.asarray(gap_evaluator(position), dtype=np.float64)
            if evaluated.shape != gaps.shape[1:] or not np.all(np.isfinite(evaluated)):
                raise ValueError("gap_evaluator returned an invalid station-gap array")
            return float(evaluated[hand, station])

        left_gap = evaluate_fraction(0.0)
        right_gap = evaluate_fraction(1.0)
        if not np.isclose(
            left_gap,
            gaps[left_index, hand_index, station_index],
            rtol=0.0,
            atol=gap_tolerance_m,
        ) or not np.isclose(
            right_gap,
            gaps[right_index, hand_index, station_index],
            rtol=0.0,
            atol=gap_tolerance_m,
        ):
            raise ValueError("gap_evaluator endpoints disagree with the retained trace")
        fraction, residual, fraction_width = _root_fraction(
            evaluate_fraction,
            left_gap,
            right_gap,
            gap_tolerance_m=gap_tolerance_m,
            fraction_tolerance=time_tolerance_s / delta_time,
            max_iterations=max_iterations,
        )
        records.append(
            ContactEventRecord(
                kind=(
                    ContactEventKind.OPENING
                    if active[left_index, hand_index, station_index]
                    else ContactEventKind.REATTACHMENT
                ),
                time_s=float(time[left_index] + fraction * delta_time),
                left_index=left_index,
                right_index=right_index,
                hand_index=hand_index,
                station_index=station_index,
                position=q_left + fraction * (q_right - q_left),
                velocity=qd_left + fraction * (qd_right - qd_left),
                gap_residual_m=residual,
                final_bracket_width_s=fraction_width * delta_time,
            )
        )
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


def locate_friction_limit_events(
    *,
    time_s: FloatArray,
    positions: FloatArray,
    velocities: FloatArray,
    station_friction_margin_n: FloatArray,
    station_active: NDArray[np.bool_],
    station_friction_limited: NDArray[np.bool_],
    margin_evaluator: Callable[[FloatArray, FloatArray], FloatArray],
    force_tolerance_n: float = 1.0e-10,
    time_tolerance_s: float = 1.0e-12,
    max_iterations: int = 80,
) -> tuple[FrictionEventRecord, ...]:
    """Locate regularized Coulomb-limit transitions on a linear state path.

    Only intervals active at both endpoints are eligible. Opening and
    reattachment remain contact events. The underlying law has no tangential
    displacement state, so limit exit must not be interpreted as static stick.
    """

    time = _finite("time_s", time_s, ndim=1)
    q = _finite("positions", positions, ndim=2)
    qd = _finite("velocities", velocities, ndim=2)
    margins = _finite("station_friction_margin_n", station_friction_margin_n, ndim=3)
    active = np.asarray(station_active, dtype=bool)
    limited = np.asarray(station_friction_limited, dtype=bool)
    if time.size < 2 or np.any(np.diff(time) <= 0.0):
        raise ValueError("time_s must contain at least two strictly increasing samples")
    if q.shape != qd.shape or q.shape[0] != time.size:
        raise ValueError("positions and velocities must share shape")
    if margins.shape[0] != time.size or active.shape != margins.shape:
        raise ValueError("friction margin and active arrays must share trace shape")
    if limited.shape != margins.shape:
        raise ValueError("friction-limited state must share the margin shape")
    if np.any(limited & ~active):
        raise ValueError("an inactive station cannot be friction limited")
    if not callable(margin_evaluator):
        raise TypeError("margin_evaluator must be callable")
    if not np.isfinite(force_tolerance_n) or force_tolerance_n <= 0.0:
        raise ValueError("force_tolerance_n must be finite and positive")
    if not np.isfinite(time_tolerance_s) or time_tolerance_s <= 0.0:
        raise ValueError("time_tolerance_s must be finite and positive")

    transitions = limited[1:] != limited[:-1]
    transitions &= active[1:] & active[:-1]
    records: list[FrictionEventRecord] = []
    for left_index, hand_index, station_index in np.argwhere(transitions):
        left_index = int(left_index)
        right_index = left_index + 1
        hand_index = int(hand_index)
        station_index = int(station_index)
        q_left, q_right = q[left_index], q[right_index]
        qd_left, qd_right = qd[left_index], qd[right_index]
        delta_time = float(time[right_index] - time[left_index])

        def evaluate_fraction(
            fraction: float,
            q_start: FloatArray = q_left,
            q_end: FloatArray = q_right,
            qd_start: FloatArray = qd_left,
            qd_end: FloatArray = qd_right,
            hand: int = hand_index,
            station: int = station_index,
        ) -> float:
            position = q_start + fraction * (q_end - q_start)
            velocity = qd_start + fraction * (qd_end - qd_start)
            evaluated = np.asarray(
                margin_evaluator(position, velocity), dtype=np.float64
            )
            if evaluated.shape != margins.shape[1:] or not np.all(
                np.isfinite(evaluated)
            ):
                raise ValueError("margin_evaluator returned an invalid array")
            return float(evaluated[hand, station])

        left_margin = evaluate_fraction(0.0)
        right_margin = evaluate_fraction(1.0)
        if not np.isclose(
            left_margin,
            margins[left_index, hand_index, station_index],
            rtol=0.0,
            atol=force_tolerance_n,
        ) or not np.isclose(
            right_margin,
            margins[right_index, hand_index, station_index],
            rtol=0.0,
            atol=force_tolerance_n,
        ):
            raise ValueError("margin evaluator endpoints disagree with the trace")
        fraction, residual, width = _root_fraction(
            evaluate_fraction,
            left_margin,
            right_margin,
            gap_tolerance_m=force_tolerance_n,
            fraction_tolerance=time_tolerance_s / delta_time,
            max_iterations=max_iterations,
        )
        records.append(
            FrictionEventRecord(
                kind=(
                    FrictionEventKind.LIMIT_ENTRY
                    if limited[right_index, hand_index, station_index]
                    else FrictionEventKind.LIMIT_EXIT
                ),
                time_s=float(time[left_index] + fraction * delta_time),
                left_index=left_index,
                right_index=right_index,
                hand_index=hand_index,
                station_index=station_index,
                position=q_left + fraction * (q_right - q_left),
                velocity=qd_left + fraction * (qd_right - qd_left),
                friction_margin_residual_n=residual,
                final_bracket_width_s=width * delta_time,
            )
        )
    return tuple(
        sorted(
            records, key=lambda item: (item.time_s, item.hand_index, item.station_index)
        )
    )


def align_state_trace_to_events(
    *,
    time_s: FloatArray,
    positions: FloatArray,
    velocities: FloatArray,
    events: tuple[EventRecord, ...],
    equality_tolerance: float = 1.0e-12,
) -> EventAlignedStateTrace:
    """Insert duplicate pre/post samples so quadrature never crosses an event."""

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
    if any(
        not isinstance(event, (ContactEventRecord, FrictionEventRecord))
        for event in events
    ):
        raise TypeError("events must contain only registered event records")
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

    output_time: list[float] = [float(time[0])]
    output_q: list[FloatArray] = [q[0].copy()]
    output_qd: list[FloatArray] = [qd[0].copy()]
    output_segments: list[int] = [0]
    event_offsets: list[int] = []
    segment = 0
    event_cursor = 0
    for left_index in range(time.size - 1):
        interval_events: list[tuple[int, EventRecord]] = []
        while (
            event_cursor < len(ordered)
            and ordered[event_cursor].left_index == left_index
        ):
            event = ordered[event_cursor]
            if event.right_index != left_index + 1:
                raise ValueError("event indices must name adjacent trace samples")
            if not time[left_index] <= event.time_s <= time[left_index + 1]:
                raise ValueError("event time lies outside its retained bracket")
            interval_events.append((event_cursor, event))
            event_cursor += 1

        group_index = 0
        while group_index < len(interval_events):
            offset, first = interval_events[group_index]
            group = [first]
            group_index += 1
            while group_index < len(interval_events) and np.isclose(
                interval_events[group_index][1].time_s,
                first.time_s,
                rtol=0.0,
                atol=equality_tolerance,
            ):
                group.append(interval_events[group_index][1])
                group_index += 1
            if any(
                not np.allclose(
                    member.position,
                    first.position,
                    rtol=0.0,
                    atol=equality_tolerance,
                )
                or not np.allclose(
                    member.velocity,
                    first.velocity,
                    rtol=0.0,
                    atol=equality_tolerance,
                )
                for member in group[1:]
            ):
                raise ValueError("simultaneous event records disagree on state")
            if not np.isclose(
                output_time[-1], first.time_s, rtol=0.0, atol=equality_tolerance
            ):
                output_time.append(first.time_s)
                output_q.append(first.position.copy())
                output_qd.append(first.velocity.copy())
                output_segments.append(segment)
            elif not np.allclose(
                output_q[-1], first.position, rtol=0.0, atol=equality_tolerance
            ):
                raise ValueError("event state disagrees with coincident trace sample")
            segment += 1
            output_time.append(first.time_s)
            output_q.append(first.position.copy())
            output_qd.append(first.velocity.copy())
            output_segments.append(segment)
            event_offsets.append(offset)

        if not (
            np.isclose(
                output_time[-1], time[left_index + 1], rtol=0.0, atol=equality_tolerance
            )
            and np.allclose(
                output_q[-1], q[left_index + 1], rtol=0.0, atol=equality_tolerance
            )
        ):
            output_time.append(float(time[left_index + 1]))
            output_q.append(q[left_index + 1].copy())
            output_qd.append(qd[left_index + 1].copy())
            output_segments.append(segment)
    if event_cursor != len(ordered):
        raise ValueError("an event does not belong to a retained trace interval")
    return EventAlignedStateTrace(
        time_s=np.asarray(output_time, dtype=np.float64),
        positions=np.asarray(output_q, dtype=np.float64),
        velocities=np.asarray(output_qd, dtype=np.float64),
        segment_ids=np.asarray(output_segments, dtype=np.int64),
        event_record_offsets=np.asarray(event_offsets, dtype=np.int64),
    )


__all__ = [
    "ContactEventKind",
    "ContactEventRecord",
    "EventRecord",
    "EventAlignedStateTrace",
    "align_state_trace_to_events",
    "locate_contact_events",
    "FrictionEventKind",
    "FrictionEventRecord",
    "locate_friction_limit_events",
]
