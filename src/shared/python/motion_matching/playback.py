"""Physical time playback engine and continuous trajectory interpolation (MV-04 #10480).

Establishes physical time as the authority for playback across Qt, React,
and native viewers, reusing the shared transport math from Tools
(:mod:`rate_of_closure.simulation.playback_transport`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from rate_of_closure.simulation.playback_transport import (
    DEFAULT_SPEED,
    PLAYBACK_SPEEDS,
    SCRUB_STEPS,
    PlaybackAdvance,
    advance_playback,
    clamp_time,
    scrub_value,
    time_at_scrub,
)
from src.shared.python.math_utils.quaternion import _canonicalize_sign, slerp


@dataclass(frozen=True)
class InterpolatedPlaybackState:
    """Continuous playback state evaluated at physical time ``time_s``."""

    time_s: float
    q: np.ndarray
    model_markers: np.ndarray | None
    target_markers: np.ndarray | None
    forces: np.ndarray | None
    lower_index: int
    fraction: float
    is_solver_state: bool


class PhysicalTimePlayback:
    """Physical time playback engine driving continuous kinematic evaluation.

    Operates on physical seconds rather than GUI render frames. Trajectory
    coordinates are interpolated continuously across nonuniform timestamps:
    quaternion rotations use spherical linear interpolation (SLERP) with
    antipodal sign continuity, and Euclidean coordinates use linear
    interpolation (LERP).
    """

    def __init__(
        self,
        *,
        times_s: np.ndarray,
        q: np.ndarray,
        model_markers: np.ndarray | None = None,
        target_markers: np.ndarray | None = None,
        forces: np.ndarray | None = None,
        quat_indices: Sequence[tuple[int, int]] = (),
        event_indices: dict[str, int] | None = None,
    ) -> None:
        if len(times_s) == 0:
            raise ValueError("times_s must contain at least one timestamp")
        if len(times_s) != len(q):
            raise ValueError("times_s and q length mismatch")

        self._times_s = np.asarray(times_s, dtype=np.float64)
        self._q = np.asarray(q, dtype=np.float64)
        self._model_markers = (
            np.asarray(model_markers, dtype=np.float64)
            if model_markers is not None
            else None
        )
        self._target_markers = (
            np.asarray(target_markers, dtype=np.float64)
            if target_markers is not None
            else None
        )
        self._forces = (
            np.asarray(forces, dtype=np.float64) if forces is not None else None
        )
        self._quat_indices = tuple(quat_indices)

        self._duration_s = float(self._times_s[-1] - self._times_s[0])
        if self._duration_s < 0.0:
            raise ValueError("times_s must be monotonically non-decreasing")

        self._event_times: dict[str, float] = {}
        if event_indices:
            for name, idx in event_indices.items():
                clamped_idx = max(0, min(idx, len(self._times_s) - 1))
                self._event_times[name] = float(self._times_s[clamped_idx])

    @property
    def duration_s(self) -> float:
        """Physical duration of the timeline in seconds."""
        return self._duration_s

    @property
    def times_s(self) -> np.ndarray:
        """Original solver timestamps."""
        return self._times_s

    @property
    def event_times(self) -> dict[str, float]:
        """Physical timestamps of named events."""
        return dict(self._event_times)

    def advance(
        self, current_time_s: float, elapsed_wall_s: float, speed: float
    ) -> PlaybackAdvance:
        """Advance physical time using monotonic elapsed wall-clock interval."""
        return advance_playback(
            time_s=current_time_s,
            elapsed_s=elapsed_wall_s,
            speed=speed,
            duration_s=self._duration_s,
        )

    def scrub_value(self, time_s: float) -> int:
        """Quantize physical time to integer slider value in [0, SCRUB_STEPS]."""
        return scrub_value(time_s, self._duration_s)

    def time_at_scrub(self, value: int) -> float:
        """Physical time corresponding to integer slider value."""
        return time_at_scrub(value, self._duration_s)

    def step_time(self, current_time_s: float, direction: int) -> float:
        """Step to adjacent discrete solver timestamp."""
        if direction == 0:
            raise ValueError("direction must be non-zero (+1 or -1)")

        t_clamped = clamp_time(current_time_s, self._duration_s)
        idx = int(np.searchsorted(self._times_s, t_clamped))

        if direction > 0:
            if idx < len(self._times_s) - 1:
                # If current_time_s is at or past times_s[idx], advance to next
                if self._times_s[idx] <= t_clamped:
                    idx = min(idx + 1, len(self._times_s) - 1)
            return float(self._times_s[idx])

        # Step backward (direction < 0)
        if idx > 0:
            if self._times_s[idx] >= t_clamped and idx > 0:
                idx -= 1
        return float(self._times_s[max(0, idx)])

    def interpolate(self, time_s: float) -> InterpolatedPlaybackState:
        """Interpolate trajectory state continuously at physical time ``time_s``."""
        t_query = clamp_time(time_s, self._duration_s)
        n = len(self._times_s)

        if n == 1 or t_query <= self._times_s[0]:
            return self._exact_state(0, t_query, is_solver=True)
        if t_query >= self._times_s[-1]:
            return self._exact_state(n - 1, t_query, is_solver=True)

        upper_idx = int(np.searchsorted(self._times_s, t_query))
        lower_idx = max(0, upper_idx - 1)

        t_lower = float(self._times_s[lower_idx])
        t_upper = float(self._times_s[upper_idx])
        dt = t_upper - t_lower

        if abs(dt) < 1e-12 or abs(t_query - t_lower) < 1e-9:
            return self._exact_state(lower_idx, t_query, is_solver=True)
        if abs(t_query - t_upper) < 1e-9:
            return self._exact_state(upper_idx, t_query, is_solver=True)

        fraction = float((t_query - t_lower) / dt)

        # 1. Interpolate coordinates q
        q_interp = self._interpolate_q(lower_idx, upper_idx, fraction)

        # 2. Interpolate model markers (LERP)
        mm_interp = self._lerp_array(
            self._model_markers, lower_idx, upper_idx, fraction
        )

        # 3. Interpolate target markers (LERP)
        tm_interp = self._lerp_array(
            self._target_markers, lower_idx, upper_idx, fraction
        )

        # 4. Interpolate forces (LERP)
        forces_interp = self._lerp_array(self._forces, lower_idx, upper_idx, fraction)

        return InterpolatedPlaybackState(
            time_s=t_query,
            q=q_interp,
            model_markers=mm_interp,
            target_markers=tm_interp,
            forces=forces_interp,
            lower_index=lower_idx,
            fraction=fraction,
            is_solver_state=False,
        )

    def _exact_state(
        self, index: int, time_s: float, is_solver: bool
    ) -> InterpolatedPlaybackState:
        mm = self._model_markers[index] if self._model_markers is not None else None
        tm = self._target_markers[index] if self._target_markers is not None else None
        forces = self._forces[index] if self._forces is not None else None
        return InterpolatedPlaybackState(
            time_s=time_s,
            q=self._q[index].copy(),
            model_markers=mm.copy() if mm is not None else None,
            target_markers=tm.copy() if tm is not None else None,
            forces=forces.copy() if forces is not None else None,
            lower_index=index,
            fraction=0.0,
            is_solver_state=is_solver,
        )

    def _interpolate_q(
        self, lower_idx: int, upper_idx: int, fraction: float
    ) -> np.ndarray:
        q_lower = self._q[lower_idx]
        q_upper = self._q[upper_idx]
        q_interp = (1.0 - fraction) * q_lower + fraction * q_upper

        # Apply SLERP to designated quaternion blocks
        for start, end in self._quat_indices:
            if end - start == 4:
                quat_a = _canonicalize_sign(q_lower[start:end])
                quat_b = _canonicalize_sign(q_upper[start:end])
                q_interp[start:end] = slerp(quat_a, quat_b, fraction)

        return q_interp

    @staticmethod
    def _lerp_array(
        arr: np.ndarray | None, lower_idx: int, upper_idx: int, fraction: float
    ) -> np.ndarray | None:
        if arr is None:
            return None
        return (1.0 - fraction) * arr[lower_idx] + fraction * arr[upper_idx]


__all__ = [
    "DEFAULT_SPEED",
    "PLAYBACK_SPEEDS",
    "SCRUB_STEPS",
    "InterpolatedPlaybackState",
    "PhysicalTimePlayback",
    "PlaybackAdvance",
]
