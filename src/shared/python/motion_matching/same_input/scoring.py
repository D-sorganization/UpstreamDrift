"""Score an engine's replay of a same-input bundle against the reference (#11607).

Open-loop replay of a full swing cannot stay within round-off for the whole
horizon: without feedback the standing body is unstable (the linearisation
has real eigenvalues near +30 1/s), so any 1e-9 relative engine difference
grows exponentially.  Two measures follow from that:

* full-horizon replay: the error history, its fitted growth exponent and the
  horizon over which the replay stays inside the acceptance bounds;
* segmented replay: the bundle is re-anchored to the reference state at the
  start of each segment, so the end-of-segment error measures engine parity
  over a fixed horizon independent of where in the swing it occurs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.same_input.bundle import InputBundle
from src.shared.python.motion_matching.same_input.integrator import (
    DynamicsPlant,
    Rollout,
    open_loop,
)

Array = NDArray[np.float64]

# Epic #11605 level L2 bounds.
COORDINATE_BOUND_RAD = 1e-6
FRAME_BOUND_M = 1e-3


class FramePlant(DynamicsPlant, Protocol):
    """Plant that can also place the spec frames (4x4 poses) at ``q``."""

    def kinematic_frames(self, q: Array) -> dict[str, Array]: ...


@dataclass(frozen=True)
class ReplayScore:
    """Error history of a replay; frame errors use the replay engine's FK."""

    time_s: Array
    coordinate_error: Array
    frame_time_s: Array
    frame_error_m: Array
    growth_rate_per_s: float
    horizon_s: float
    failure: str | None = None

    def summary(self) -> dict[str, float | str | None]:
        """Strict-JSON scalars; an undefined growth rate becomes ``None``."""
        growth = self.growth_rate_per_s
        return {
            "failure": self.failure,
            "duration_s": float(self.time_s[-1]),
            "max_coordinate_error_rad": float(self.coordinate_error.max()),
            "max_frame_error_m": float(self.frame_error_m.max()),
            "growth_rate_per_s": growth if np.isfinite(growth) else None,
            "horizon_s": self.horizon_s,
        }


def _frame_positions(plant: FramePlant, q: Array) -> Array:
    return np.array([pose[:3, 3] for pose in plant.kinematic_frames(q).values()])


def growth_rate(time_s: Array, error: Array, low: float, high: float) -> float:
    """Least-squares slope of ``log(error)`` while ``low <= error <= high``.

    Returns NaN when fewer than three samples fall inside the band.
    """
    mask = (error >= low) & (error <= high)
    if int(mask.sum()) < 3:
        return float("nan")
    slope = np.polyfit(time_s[mask], np.log(error[mask]), 1)[0]
    return float(slope)


def score_replay(
    plant: FramePlant, bundle: InputBundle, rollout: Rollout, *, frame_stride: int = 10
) -> ReplayScore:
    """Compare ``rollout`` with the bundle reference over the steps it reached."""
    reached = rollout.q.shape[0]
    if reached > bundle.steps + 1 or rollout.q.shape[1] != bundle.q0.size:
        raise ValueError("rollout must not exceed the bundle")
    if frame_stride < 1:
        raise ValueError("frame_stride must be positive")
    coordinate_error = np.abs(rollout.q - bundle.reference_q[:reached]).max(axis=1)
    rows = np.unique(np.r_[np.arange(0, reached, frame_stride), reached - 1])
    frame_error = np.array(
        [
            np.linalg.norm(
                _frame_positions(plant, rollout.q[k])
                - _frame_positions(plant, bundle.reference_q[k]),
                axis=1,
            ).max()
            for k in rows
        ]
    )
    frame_time = rows * bundle.dt_s
    inside = (coordinate_error <= COORDINATE_BOUND_RAD)[rows] & (
        frame_error <= FRAME_BOUND_M
    )
    first_out = np.flatnonzero(~inside)
    horizon = float(frame_time[-1] if first_out.size == 0 else frame_time[first_out[0]])
    return ReplayScore(
        time_s=rollout.time_s,
        coordinate_error=coordinate_error,
        frame_time_s=frame_time,
        frame_error_m=frame_error,
        growth_rate_per_s=growth_rate(rollout.time_s, coordinate_error, 1e-12, 1e-3),
        horizon_s=horizon,
        failure=rollout.failure,
    )


def segmented_replay(
    plant: FramePlant, bundle: InputBundle, *, segment_steps: int
) -> list[dict[str, float]]:
    """Replay each ``segment_steps`` window from its reference start state.

    Returns one record per segment with its start time and end-of-segment
    coordinate and frame errors.  The bundle's recorded ball force is applied
    on each segment's own clock.
    """
    if segment_steps < 1:
        raise ValueError("segment_steps must be positive")
    segments = []
    impact = bundle.ball_impact()
    for start in range(0, bundle.steps, segment_steps):
        stop = min(start + segment_steps, bundle.steps)
        rollout = open_loop(
            plant,
            bundle.reference_q[start],
            bundle.reference_v[start],
            bundle.efforts[start:stop],
            dt_s=bundle.dt_s,
            impact=None if impact is None else impact.shifted(-start * bundle.dt_s),
        )
        end_q, ref_q = rollout.q[-1], bundle.reference_q[stop]
        segments.append(
            {
                "start_s": start * bundle.dt_s,
                "steps": stop - start,
                "coordinate_error_rad": float(np.abs(end_q - ref_q).max()),
                "frame_error_m": float(
                    np.linalg.norm(
                        _frame_positions(plant, end_q) - _frame_positions(plant, ref_q),
                        axis=1,
                    ).max()
                ),
            }
        )
    return segments
