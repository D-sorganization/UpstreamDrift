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
        raise ValueError("active-set transition does not bracket a signed-gap root")
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


__all__ = ["ContactEventKind", "ContactEventRecord", "locate_contact_events"]
