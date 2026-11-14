"""Event-relative time normalisation (101-point resampling) (#14427).

Provides 101-point resampling (0% to 100% in 1% increments) between named swing
events (e.g. address -> top -> impact or ordered Chu et al. 2010 events).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import CubicSpline


@dataclass(frozen=True)
class TimeNormalisedResult:
    """Result of 101-point event-relative time normalisation.

    Attributes:
        grid_percent: 1-D array of shape (101,) containing 0.0, 1.0, ..., 100.0 (%).
        values: Resampled signal values of shape (101,) or (101, D).
        target_times: Physical timestamps corresponding to each point on the grid.
        event_percentages: Mapping of each event in the sequence to its grid percentage.
    """

    grid_percent: np.ndarray
    values: np.ndarray
    target_times: np.ndarray
    event_percentages: dict[str, float]


def _validate_series_inputs(
    time: ArrayLike, signal: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    t_arr = np.asarray(time, dtype=float)
    if t_arr.ndim != 1 or t_arr.size < 2:
        raise ValueError("time must be a 1-D array of length >= 2")
    if not np.all(np.isfinite(t_arr)):
        raise ValueError("time array must contain only finite values")
    if np.any(np.diff(t_arr) <= 0.0):
        raise ValueError("time array must be strictly monotonically increasing")

    s_arr = np.asarray(signal, dtype=float)
    if s_arr.size == 0 or s_arr.shape[0] != t_arr.size:
        raise ValueError("signal leading dimension must match time array length")
    if not np.all(np.isfinite(s_arr)):
        raise ValueError("signal array must contain only finite values")
    return t_arr, s_arr


def _event_times_from_sequence(
    events: Mapping[str, float], event_sequence: Sequence[str]
) -> tuple[tuple[str, ...], list[float]]:
    seq = tuple(str(e) for e in event_sequence)
    if len(seq) < 2:
        raise ValueError("event_sequence must contain at least 2 named events")

    event_times: list[float] = []
    for ev in seq:
        if ev not in events:
            raise ValueError(f"Missing required event '{ev}' in events mapping")
        t_val = float(events[ev])
        if not np.isfinite(t_val):
            raise ValueError(f"Event timestamp for '{ev}' must be finite")
        event_times.append(t_val)

    for i in range(len(event_times) - 1):
        if event_times[i] >= event_times[i + 1]:
            raise ValueError(
                f"Event times must be strictly increasing: {seq[i]} ({event_times[i]}) "
                f">= {seq[i + 1]} ({event_times[i + 1]})"
            )
    return seq, event_times


def _interpolate_at_targets(
    method: Literal["linear", "cubic"],
    t_arr: np.ndarray,
    s_arr: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    if method == "linear":
        if s_arr.ndim == 1:
            return np.interp(target_times, t_arr, s_arr)
        cols = [
            np.interp(target_times, t_arr, s_arr[:, d]) for d in range(s_arr.shape[1])
        ]
        return np.column_stack(cols)
    if method == "cubic":
        return CubicSpline(t_arr, s_arr, axis=0)(target_times)
    raise ValueError(
        f"Unsupported interpolation method: '{method}' (choose 'linear' or 'cubic')"
    )


def resample_event_relative_101(
    time: ArrayLike,
    signal: ArrayLike,
    events: Mapping[str, float],
    *,
    event_sequence: Sequence[str] = ("address", "impact"),
    phase_points: Sequence[int] | None = None,
    method: Literal["linear", "cubic"] = "linear",
) -> TimeNormalisedResult:
    """Resample a time-series signal to 101 points (0% to 100%) between named events (#14427).

    Preconditions:
        - time is 1-D, non-empty, finite, and strictly monotonically increasing.
        - signal is non-empty, finite, with matching leading dimension to time.
        - event_sequence contains at least 2 distinct named events.
        - All named events exist in events with finite timestamps.
        - Event timestamps are strictly monotonically increasing.
        - Event timestamps fall within the bounds of time.
    """
    t_arr, s_arr = _validate_series_inputs(time, signal)
    seq, event_times = _event_times_from_sequence(events, event_sequence)

    t_min = float(t_arr[0])
    t_max = float(t_arr[-1])
    if event_times[0] < t_min or event_times[-1] > t_max:
        raise ValueError(
            f"Event times out of bounds: [{event_times[0]}, {event_times[-1]}] "
            f"outside time span [{t_min}, {t_max}]"
        )

    n_total = 101
    grid_percent = np.linspace(0.0, 100.0, n_total)
    event_percentages: dict[str, float] = {}

    if phase_points is None:
        t_start = event_times[0]
        t_end = event_times[-1]
        t_span = t_end - t_start
        target_times = t_start + (grid_percent / 100.0) * t_span
        for name, t_ev in zip(seq, event_times, strict=True):
            pct = 100.0 * (t_ev - t_start) / t_span
            event_percentages[name] = float(pct)
        event_percentages[seq[0]] = 0.0
        event_percentages[seq[-1]] = 100.0
    else:
        p_points = tuple(int(p) for p in phase_points)
        num_phases = len(seq) - 1
        if len(p_points) != num_phases:
            raise ValueError(
                f"phase_points length ({len(p_points)}) must equal number of phases ({num_phases})"
            )
        expected_total = sum(p_points) - (num_phases - 1)
        if expected_total != n_total:
            raise ValueError(
                f"Sum of phase points with shared boundaries must equal {n_total}; got {expected_total}"
            )

        phase_time_chunks: list[np.ndarray] = []
        accum_pct = 0.0
        pct_step = 100.0 / (n_total - 1)

        for p_idx in range(num_phases):
            n_p = p_points[p_idx]
            t_p0 = event_times[p_idx]
            t_p1 = event_times[p_idx + 1]
            t_sub = np.linspace(t_p0, t_p1, n_p)
            g_sub = np.linspace(accum_pct, accum_pct + (n_p - 1) * pct_step, n_p)
            accum_pct += (n_p - 1) * pct_step
            event_percentages[seq[p_idx]] = float(g_sub[0])
            if p_idx == 0:
                phase_time_chunks.append(t_sub)
            else:
                phase_time_chunks.append(t_sub[1:])
        event_percentages[seq[-1]] = 100.0
        target_times = np.concatenate(phase_time_chunks)

    values = _interpolate_at_targets(method, t_arr, s_arr, target_times)

    return TimeNormalisedResult(
        grid_percent=grid_percent,
        values=values,
        target_times=target_times,
        event_percentages=event_percentages,
    )
