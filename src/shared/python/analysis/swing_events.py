"""Canonical swing event detector from 1-D speed profile (Issue #11014).

Provides:
- SwingEventFrames: Immutable container for address, top, peak, and finish frames.
- peak_speed_index: Canonical helper locating the peak speed index.
- detect_swing_events: Frame-rate-aware event detector with DbC preconditions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.core.contracts import ensure, require


@dataclass(frozen=True)
class SwingEventFrames:
    """Frame indices of key golf swing events.

    Attributes:
        address: Start of swing motion (end of quiet address period).
        top: Top of backswing / transition (slowest frame before downswing).
        peak: Maximum speed (proxy for ball impact).
        finish: Completion of follow-through (speed returns below threshold).

    Postconditions:
        0 <= address <= top <= peak <= finish
    """

    address: int
    top: int
    peak: int
    finish: int

    def __post_init__(self) -> None:
        valid = 0 <= self.address <= self.top <= self.peak <= self.finish
        message = (
            "Frames must satisfy 0 <= address <= top <= peak <= finish; "
            f"got address={self.address}, top={self.top}, "
            f"peak={self.peak}, finish={self.finish}"
        )
        ensure(valid, message, (self.address, self.top, self.peak, self.finish))
        # Frame order is a data invariant, not a debug check: it holds even
        # when the contract level is OFF.
        if not valid:
            raise ValueError(message)


def peak_speed_index(speed: ArrayLike) -> int:
    """Return the frame index of maximum speed.

    Preconditions:
        - speed is 1-D, non-empty, and finite.
    """
    arr = np.asarray(speed, dtype=float)
    require(arr.ndim == 1, "speed must be a 1-D array", arr.ndim)
    require(arr.size > 0, "speed array must be non-empty", arr.size)
    require(bool(np.all(np.isfinite(arr))), "speed array must be finite")
    return int(np.argmax(arr))


def detect_swing_events(
    speed: ArrayLike,
    fps: float,
    *,
    quiet_fraction: float = 0.05,
    quiet_s: float = 0.15,
    max_downswing_s: float = 0.5,
    finish_fraction: float | None = None,
) -> SwingEventFrames:
    """Address, top, peak speed and finish from the speed profile.

    Peak speed is the global maximum. Address is the end of the last stretch
    of at least ``quiet_s`` seconds before the peak with speed under
    ``quiet_fraction`` of it. The top of the backswing is the slowest frame
    between the address and the peak, looking back at most
    ``max_downswing_s`` (a downswing is shorter than that, the backswing is
    not, so the search cannot land on the address itself). Finish is the
    first frame after the peak under the finish threshold (defaults to
    ``quiet_fraction``). Thresholds rather than local minima: at 100+ fps a
    real profile has a minimum every few frames. Every event is a frame
    index; nothing is interpolated.

    Preconditions:
        - fps > 0 and finite
        - speed is 1-D, finite, with at least 3 samples
        - 0 < quiet_fraction < 1, quiet_s > 0, max_downswing_s > 0
        - 0 < finish_fraction < 1 if finish_fraction is not None

    Postconditions:
        - 0 <= address <= top <= peak <= finish
    """
    require(
        isinstance(fps, (int, float, np.number))
        and fps > 0
        and math.isfinite(float(fps)),
        "fps must be positive and finite",
        fps,
    )
    require(
        0 < quiet_fraction < 1 and math.isfinite(quiet_fraction),
        "quiet_fraction must be in (0, 1)",
        quiet_fraction,
    )
    require(
        quiet_s > 0 and math.isfinite(quiet_s),
        "quiet_s must be positive and finite",
        quiet_s,
    )
    require(
        max_downswing_s > 0 and math.isfinite(max_downswing_s),
        "max_downswing_s must be positive and finite",
        max_downswing_s,
    )
    if finish_fraction is not None:
        require(
            0 < finish_fraction < 1 and math.isfinite(finish_fraction),
            "finish_fraction must be in (0, 1)",
            finish_fraction,
        )

    arr = np.asarray(speed, dtype=float)
    require(arr.ndim == 1, "speed must be a 1-D array", arr.ndim)
    require(arr.size >= 3, "speed must have at least 3 samples", arr.size)
    require(bool(np.all(np.isfinite(arr))), "speed must contain only finite numbers")

    peak = peak_speed_index(arr)
    quiet = quiet_fraction * arr[peak]
    window = max(int(round(quiet_s * fps)), 1)
    address = 0
    run = 0
    for t in range(peak - 1, -1, -1):
        run = run + 1 if arr[t] < quiet else 0
        if run >= window:
            address = t + window - 1
            break
    lookback = max(int(round(max_downswing_s * fps)), 2)
    start = max(address + 1, peak - lookback)
    top = start + int(np.argmin(arr[start:peak])) if start < peak else address
    eff_finish = finish_fraction if finish_fraction is not None else quiet_fraction
    finish_thresh = eff_finish * arr[peak]
    after = np.flatnonzero(arr[peak:] < finish_thresh)
    finish = int(peak + after[0]) if after.size else int(arr.size - 1)

    return SwingEventFrames(
        address=address,
        top=top,
        peak=peak,
        finish=finish,
    )
