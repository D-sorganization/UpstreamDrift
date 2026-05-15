"""Impact detection, resampling, and alignment helpers (private).

Used by both the Excel and C3D loaders so the alignment/resampling logic lives
in exactly one place.
"""

from __future__ import annotations

import logging

import numpy as np

from ..club_target import AlignOptions
from ._quaternion import slerp

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
            speeds[i] = float(np.linalg.norm(v))
        speeds[0] = speeds[2]
        speeds[1] = speeds[2]
        speeds[-1] = speeds[-3]
        speeds[-2] = speeds[-3]
    else:
        for i in range(n - 1):
            dt = time[i + 1] - time[i]
            if dt <= 0:
                continue
            speeds[i] = float(np.linalg.norm((clubhead[i + 1] - clubhead[i]) / dt))
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

    butt = _interp_xyz(sim_time, raw_time, raw_butt)
    clubhead = _interp_xyz(sim_time, raw_time, raw_clubhead)
    quat = _slerp_series(sim_time, raw_time, raw_quat)
    impact_idx_out = int(np.argmin(np.abs(sim_time - opts.impact_target_t_s))) + 1
    return sim_time, butt, clubhead, quat, impact_idx_out


def _interp_xyz(
    sim_t: np.ndarray, raw_t: np.ndarray, raw_xyz: np.ndarray
) -> np.ndarray:
    """Linear interpolation of an ``(N, 3)`` series, clamping at endpoints."""
    out = np.empty((sim_t.shape[0], 3), dtype=np.float64)
    for k in range(3):
        out[:, k] = np.interp(sim_t, raw_t, raw_xyz[:, k])
    return out


def _slerp_series(
    sim_t: np.ndarray, raw_t: np.ndarray, raw_q: np.ndarray
) -> np.ndarray:
    """SLERP an ``(N, 4)`` quaternion series onto ``sim_t``."""
    out = np.empty((sim_t.shape[0], 4), dtype=np.float64)
    last = raw_t.shape[0] - 1
    for i, t in enumerate(sim_t):
        if t <= raw_t[0]:
            out[i] = raw_q[0]
            continue
        if t >= raw_t[last]:
            out[i] = raw_q[last]
            continue
        j = int(np.searchsorted(raw_t, t)) - 1
        j = max(0, min(j, last - 1))
        span = raw_t[j + 1] - raw_t[j]
        alpha = 0.0 if span == 0.0 else (t - raw_t[j]) / span
        out[i] = slerp(raw_q[j], raw_q[j + 1], float(alpha))
    norms = np.sqrt(np.einsum("ij,ij->i", out, out))[:, np.newaxis]
    norms[norms == 0.0] = 1.0
    return out / norms
