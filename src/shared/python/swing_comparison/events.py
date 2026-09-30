"""Swing events detection for golf swing comparison (Issue #11164).

Defines:
- SwingEvents: Immutable dataclass for address, top of backswing, impact, and finish.
- detect_events: Pure function with DbC detecting the key events.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from src.shared.python.contracts import ensure, require

if TYPE_CHECKING:
    from src.shared.python.swing_comparison.motion import (
        SwingMotion,
    )


@dataclass(frozen=True)
class SwingEvents:
    """Frame indices and timestamps of key golf swing events.

    Attributes:
        address_idx: Frame index for address (start of motion).
        address_time: Timestamp for address in seconds.
        top_idx: Frame index for top of backswing (reversal / minimum path speed).
        top_time: Timestamp for top of backswing in seconds.
        impact_idx: Frame index for impact (maximum club-head speed near lowest Z).
        impact_time: Timestamp for impact in seconds.
        finish_idx: Frame index for finish (follow-through completion).
        finish_time: Timestamp for finish in seconds.

    Invariants:
        0 <= address_idx <= top_idx <= impact_idx <= finish_idx
        address_time <= top_time <= impact_time <= finish_time
    """

    address_idx: int
    address_time: float
    top_idx: int
    top_time: float
    impact_idx: int
    impact_time: float
    finish_idx: int
    finish_time: float

    def __post_init__(self) -> None:
        valid_indices = (
            0 <= self.address_idx <= self.top_idx <= self.impact_idx <= self.finish_idx
        )
        valid_times = (
            self.address_time <= self.top_time <= self.impact_time <= self.finish_time
        )
        msg = (
            f"Invalid swing events order: indices=({self.address_idx}, {self.top_idx}, "
            f"{self.impact_idx}, {self.finish_idx}), times=({self.address_time:.3f}, "
            f"{self.top_time:.3f}, {self.impact_time:.3f}, {self.finish_time:.3f})"
        )
        ensure(valid_indices and valid_times, msg)
        if not (valid_indices and valid_times):
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Convert events to JSON-serializable dictionary."""
        return {
            "address_idx": self.address_idx,
            "address_time": float(self.address_time),
            "top_idx": self.top_idx,
            "top_time": float(self.top_time),
            "impact_idx": self.impact_idx,
            "impact_time": float(self.impact_time),
            "finish_idx": self.finish_idx,
            "finish_time": float(self.finish_time),
        }


def _fill_nans_1d(arr: np.ndarray) -> np.ndarray:
    """Linearly interpolate internal NaNs in a 1D array."""
    out = arr.copy()
    nans = np.isnan(out)
    if not np.any(nans):
        return out
    valid_idx = np.where(~nans)[0]
    if valid_idx.size == 0:
        return np.zeros_like(out)
    out[nans] = np.interp(np.where(nans)[0], valid_idx, out[valid_idx])
    return out


def _fill_nans_3d(arr: np.ndarray) -> np.ndarray:
    """Linearly interpolate internal NaNs across all 3 coordinates."""
    out = np.empty_like(arr, dtype=np.float64)
    for col in range(arr.shape[1]):
        out[:, col] = _fill_nans_1d(arr[:, col])
    return out


def _resolve_tracking_position(motion: SwingMotion) -> np.ndarray:
    """Return the best-available club-head/marker trajectory, NaN-filled."""
    pos: np.ndarray | None = None
    if motion.club_head is not None:
        pos = motion.club_head
    else:
        for candidate in (
            "Marker_2:2:1",
            "club_head",
            "CH",
            "HEAD",
            "LWristTop",
            "grip",
        ):
            if candidate in motion.markers:
                pos = motion.markers[candidate]
                break
        if pos is None and motion.grip is not None:
            pos = motion.grip
        elif pos is None and motion.markers:
            # Pick marker with largest variance in displacement
            best_var = -1.0
            for arr in motion.markers.values():
                v = float(np.nanvar(arr))
                if v > best_var:
                    best_var = v
                    pos = arr

    require(pos is not None, "No tracking trajectory available for event detection")
    return _fill_nans_3d(np.asarray(pos, dtype=np.float64))


def _compute_speed_profile(pos_clean: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Return per-frame speed magnitude from a NaN-filled position trajectory."""
    vel = np.gradient(pos_clean, t, axis=0)
    speed = np.linalg.norm(vel, axis=-1)
    if not np.all(np.isfinite(speed)):
        speed = _fill_nans_1d(speed)
    return speed


def _detect_impact_index(pos_clean: np.ndarray, speed: np.ndarray, n: int) -> int:
    """Detect impact as the maximum speed frame near the lowest club-head Z."""
    start_search = max(1, int(0.20 * n))
    z = pos_clean[:, 2]
    z_sub = z[start_search:]
    z_min = float(np.nanmin(z_sub)) if z_sub.size > 0 else float(np.nanmin(z))
    z_max = float(np.nanmax(z_sub)) if z_sub.size > 0 else float(np.nanmax(z))
    z_span = z_max - z_min

    # Low Z threshold: points within lowest 35% of height span in the candidate window
    z_thresh = z_min + 0.35 * z_span if z_span > 1e-4 else z_min + 0.1
    candidate_mask = (np.arange(n) >= start_search) & (z <= z_thresh)
    candidates = np.where(candidate_mask)[0]

    if candidates.size > 0:
        return int(candidates[np.nanargmax(speed[candidates])])
    return int(start_search + np.nanargmax(speed[start_search:]))


def _detect_top_of_backswing_index(
    motion: SwingMotion,
    speed: np.ndarray,
    impact_idx: int,
    dt: float,
    max_downswing_s: float,
) -> int:
    """Detect top-of-backswing as the speed minimum before impact, refined by grip."""
    max_ds_frames = min(impact_idx - 1, max(2, int(round(max_downswing_s / dt))))
    min_ds_frames = max(1, int(round(0.08 / dt)))
    start_tob = max(1, impact_idx - max_ds_frames)
    end_tob = max(start_tob + 1, impact_idx - min_ds_frames)

    tob_window = speed[start_tob:end_tob]
    top_idx = (
        int(start_tob + np.nanargmin(tob_window)) if tob_window.size > 0 else start_tob
    )

    # If grip is available, check if lead-hand minimum along X occurs near top
    if motion.grip is not None:
        grip_clean = _fill_nans_3d(np.asarray(motion.grip, dtype=np.float64))
        grip_x_win = grip_clean[start_tob:end_tob, 0]
        if grip_x_win.size > 0:
            grip_rev_idx = int(start_tob + np.nanargmin(grip_x_win))
            # If grip reversal is very close to speed minimum, use it or keep speed min
            if abs(grip_rev_idx - top_idx) <= 5:
                top_idx = grip_rev_idx

    # Guard: top_idx must be strictly before impact
    if top_idx >= impact_idx:
        top_idx = max(0, impact_idx - 1)
    return top_idx


def _detect_address_index(
    speed: np.ndarray,
    top_idx: int,
    impact_idx: int,
    dt: float,
    quiet_threshold_fraction: float,
    quiet_duration_s: float,
) -> int:
    """Detect address as the end of the last quiet run before the backswing peak."""
    peak_speed = float(speed[impact_idx])
    quiet_thresh = max(1e-3, quiet_threshold_fraction * peak_speed)
    quiet_samples = max(1, int(round(quiet_duration_s / dt)))

    # Find peak speed during backswing to avoid stopping at top-of-backswing reversal
    bs_search_end = max(2, int(0.85 * top_idx))
    bs_speed = speed[:bs_search_end]
    address_idx = 0
    if bs_speed.size > 0 and np.nanmax(bs_speed) > quiet_thresh:
        bs_peak_idx = int(np.nanargmax(bs_speed))
        run_len = 0
        for i in range(bs_peak_idx, -1, -1):
            if speed[i] < quiet_thresh:
                run_len += 1
                if run_len >= quiet_samples:
                    address_idx = i + run_len - 1
                    break
            else:
                run_len = 0

    if address_idx > top_idx:
        address_idx = top_idx
    return address_idx


def _detect_finish_index(
    speed: np.ndarray,
    impact_idx: int,
    n: int,
    finish_threshold_fraction: float,
) -> int:
    """Detect finish as the first low-speed frame after impact, or the last frame."""
    peak_speed = float(speed[impact_idx])
    finish_thresh = max(1e-3, finish_threshold_fraction * peak_speed)
    after_impact = np.where((np.arange(n) > impact_idx) & (speed < finish_thresh))[0]
    finish_idx = int(after_impact[0]) if after_impact.size > 0 else n - 1

    if finish_idx < impact_idx:
        finish_idx = impact_idx
    return finish_idx


def detect_events(
    motion: SwingMotion,
    *,
    quiet_threshold_fraction: float = 0.03,
    quiet_duration_s: float = 0.08,
    max_downswing_s: float = 0.50,
    finish_threshold_fraction: float = 0.05,
) -> SwingEvents:
    """Detect address, top of backswing, impact, and finish events from motion.

    Definitions:
    - Address: Start of motion after the last quiet resting period before backswing.
    - Top of backswing: Club-head direction reversal or minimum speed along path
      between address and impact.
    - Impact: Maximum club-head speed near the lowest club-head point (lowest Z).
    - Finish: First frame after impact where club-head speed drops below finish threshold,
      or end of the recorded sequence.

    Preconditions:
    - motion.t has length >= 4, is strictly monotonically increasing and all finite.
    - motion has at least one valid trajectory (club_head, grip, or marker).

    Postconditions:
    - 0 <= address_idx <= top_idx <= impact_idx <= finish_idx < len(motion.t)
    - address_time <= top_time <= impact_time <= finish_time

    Args:
        motion: SwingMotion instance in a right-handed frame with Z up.
        quiet_threshold_fraction: Fraction of peak speed defining quiet address.
        quiet_duration_s: Minimum duration in seconds of quiet period for address.
        max_downswing_s: Maximum anticipated downswing duration in seconds.
        finish_threshold_fraction: Fraction of peak speed defining finish.

    Returns:
        SwingEvents with frame indices and timestamps.
    """
    require(
        len(motion.t) >= 4, "motion.t must contain at least 4 frames", len(motion.t)
    )
    require(bool(np.all(np.isfinite(motion.t))), "motion.t must be finite")
    require(
        bool(np.all(np.diff(motion.t) > 0)),
        "motion.t must be strictly monotonically increasing",
    )

    t = np.asarray(motion.t, dtype=np.float64)
    n = len(t)
    dt = float(np.mean(np.diff(t)))

    pos_clean = _resolve_tracking_position(motion)
    speed = _compute_speed_profile(pos_clean, t)

    impact_idx = _detect_impact_index(pos_clean, speed, n)
    top_idx = _detect_top_of_backswing_index(
        motion, speed, impact_idx, dt, max_downswing_s
    )
    address_idx = _detect_address_index(
        speed, top_idx, impact_idx, dt, quiet_threshold_fraction, quiet_duration_s
    )
    finish_idx = _detect_finish_index(speed, impact_idx, n, finish_threshold_fraction)

    events = SwingEvents(
        address_idx=address_idx,
        address_time=float(t[address_idx]),
        top_idx=top_idx,
        top_time=float(t[top_idx]),
        impact_idx=impact_idx,
        impact_time=float(t[impact_idx]),
        finish_idx=finish_idx,
        finish_time=float(t[finish_idx]),
    )

    ensure(
        0
        <= events.address_idx
        <= events.top_idx
        <= events.impact_idx
        <= events.finish_idx
        < n,
        "Detected events must satisfy 0 <= address <= top <= impact <= finish < n",
    )
    return events
