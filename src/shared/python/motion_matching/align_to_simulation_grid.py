"""Top-level resampling / alignment helper.

Public Python mirror of ``private/align_to_simulation_grid.m``. Built on
top of the existing private ``loaders._align`` helpers so both the dispatch
loader and any engine-specific consumer can call exactly one function.

Public API:
    align_to_simulation_grid -- linear/SLERP resample of raw traces onto the
                                simulation timegrid; returns positions,
                                quaternion series, and the 1-based impact
                                index on the new grid.
    detect_impact_index      -- argmax-clubhead-speed impact estimator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .loaders._align import detect_impact_index, resample_target
from .target import AlignOptions

__all__ = [
    "AlignedTrajectory",
    "align_to_simulation_grid",
    "detect_impact_index",
]


@dataclass(frozen=True)
class AlignedTrajectory:
    """Output of :func:`align_to_simulation_grid`.

    Attributes:
        time:       ``(N,)`` simulation timegrid in seconds, starts at 0.
        butt:       ``(N, 3)`` butt position in metres.
        clubhead:   ``(N, 3)`` clubhead position in metres.
        club_quat:  ``(N, 4)`` unit quaternions ``[w, x, y, z]``.
        impact_idx: 1-based index of the impact frame on ``time``.
    """

    time: NDArray[np.float64]
    butt: NDArray[np.float64]
    clubhead: NDArray[np.float64]
    club_quat: NDArray[np.float64]
    impact_idx: int


def align_to_simulation_grid(
    raw_time: NDArray[np.float64],
    raw_butt: NDArray[np.float64],
    raw_clubhead: NDArray[np.float64],
    raw_quat: NDArray[np.float64],
    opts: AlignOptions | None = None,
    *,
    impact_idx_raw: int | None = None,
) -> AlignedTrajectory:
    """Resample raw mocap onto the simulation timegrid.

    Args:
        raw_time:    ``(M,)`` strictly increasing source timestamps (s).
        raw_butt:    ``(M, 3)`` raw butt position (m).
        raw_clubhead:``(M, 3)`` raw clubhead position (m).
        raw_quat:    ``(M, 4)`` unit quaternions (w-first).
        opts:        Alignment options. Defaults to :class:`AlignOptions`.
        impact_idx_raw: Optional pre-detected raw impact index. Defaults to
                        the speed-argmax estimator
                        :func:`detect_impact_index`.

    Returns:
        :class:`AlignedTrajectory` on the simulation grid.

    Raises:
        ValueError: For shape mismatches or non-monotonic time.
    """
    options = opts if opts is not None else AlignOptions()
    raw_time = np.asarray(raw_time, dtype=np.float64).reshape(-1)
    if raw_time.shape[0] < 2:
        raise ValueError("raw_time must have at least 2 samples")
    if np.any(np.diff(raw_time) <= 0):
        raise ValueError("raw_time must be strictly increasing")

    impact = (
        int(impact_idx_raw)
        if impact_idx_raw is not None
        else detect_impact_index(raw_time, raw_clubhead)
    )

    sim_time, butt, clubhead, quat, impact_out = resample_target(
        raw_time, raw_butt, raw_clubhead, raw_quat, impact, options
    )
    return AlignedTrajectory(
        time=sim_time,
        butt=butt,
        clubhead=clubhead,
        club_quat=quat,
        impact_idx=int(impact_out),
    )
