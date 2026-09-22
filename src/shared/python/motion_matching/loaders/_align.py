"""Impact detection, resampling, and alignment helpers (private).

Used by both the Excel and C3D loaders so the alignment/resampling logic lives
in exactly one place.
"""

from __future__ import annotations

import logging

import math

import numpy as np

from .._series_interp import interp_xyz_series
from ..club_target import AlignOptions
from ._quaternion import slerp_series

logger = logging.getLogger(__name__)


def detect_impact_index(time: np.ndarray, clubhead: np.ndarray) -> int:
    """Index of the frame with the maximum clubhead speed.

    Uses a 5-point central difference where it fits, falling back to lower-order
    differences at the edges.
    """
    if time.shape[0] != clubhead.shape[0]:
        raise ValueError("time and clubhead must share leading dim")
    n = time.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 samples to detect impact")
    speeds = np.zeros(n, dtype=np.float64)
    if n >= 5:
        for i in range(2, n - 2):
            dt = time[i + 1] - time[i - 1]
            if dt <= 0:
                continue
            v = (clubhead[i + 1] - clubhead[i - 1]) / dt
            # ⚡ Bolt: Using math.sqrt(np.vdot) avoids dispatch overhead and is ~1.5x faster than np.linalg.norm
            speeds[i] = float(math.sqrt(np.vdot(v, v)))
        speeds[0] = speeds[2]
        speeds[1] = speeds[2]
        speeds[-1] = speeds[-3]
        speeds[-2] = speeds[-3]
    else:
        for i in range(n - 1):
            dt = time[i + 1] - time[i]
            if dt <= 0:
                continue
            v = (clubhead[i + 1] - clubhead[i]) / dt
            # ⚡ Bolt: Using math.sqrt(np.vdot) avoids dispatch overhead and is ~1.5x faster than np.linalg.norm
            speeds[i] = float(math.sqrt(np.vdot(v, v)))
        speeds[-1] = speeds[-2]
    return int(np.argmax(speeds))


def resample_target(
    raw_time: np.ndarray,
    raw_butt: np.ndarray,
    raw_clubhead: np.ndarray,
    raw_quat: np.ndarray,
    impact_idx_raw: int,
    opts: AlignOptions,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Resample raw arrays onto a uniform sim grid; return aligned arrays.

    The output time vector is ``0 : 1/fs : T`` with ``fs = opts.sample_rate_hz``
    and ``T = opts.simulation_time_s``. Impact alignment shifts the raw time
    vector so the measured impact lands on ``opts.impact_target_t_s``.

    Returns:
        ``(time, butt, clubhead, quat, impact_idx_1based)``
    """
    if opts.sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be > 0")
    if opts.simulation_time_s <= 0:
        raise ValueError("simulation_time_s must be > 0")

    sim_dt = 1.0 / float(opts.sample_rate_hz)
    n_out = int(round(opts.simulation_time_s * opts.sample_rate_hz)) + 1
    sim_time = np.arange(n_out, dtype=np.float64) * sim_dt

    raw_time = np.asarray(raw_time, dtype=np.float64).copy()
    if opts.time_alignment == "impact":
        offset = float(raw_time[impact_idx_raw]) - float(opts.impact_target_t_s)
        raw_time -= offset
    elif opts.time_alignment == "address" or opts.time_alignment == "none":
        raw_time -= float(raw_time[0])
    else:
        raise ValueError(f"Unknown time_alignment {opts.time_alignment!r}")

    butt = interp_xyz_series(sim_time, raw_time, raw_butt)
    clubhead = interp_xyz_series(sim_time, raw_time, raw_clubhead)
    quat = slerp_series(sim_time, raw_time, raw_quat)
    if opts.time_alignment == "impact":
        impact_idx_out = int(np.argmin(np.abs(sim_time - opts.impact_target_t_s))) + 1
    else:
        impact_t = float(raw_time[impact_idx_raw])
        impact_idx_out = int(np.argmin(np.abs(sim_time - impact_t))) + 1
    return sim_time, butt, clubhead, quat, impact_idx_out
