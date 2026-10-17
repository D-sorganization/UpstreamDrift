"""Wall-clock playback timing for the results panel (#12948, epic #12878).

Qt-free. Play used to advance a fixed number of frames per timer tick, so a
slow render made the run play at a fraction of real time (about 0.14x).
:class:`PlaybackClock` maps *wall time* to a frame instead: the frame shown is
always the one the run reached after the elapsed real time (times the chosen
speed), and slow renders simply skip frames.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

__all__ = ["SPEEDS", "PlaybackClock"]

#: Playback speeds offered by the panel's speed selector.
SPEEDS: tuple[float, ...] = (0.25, 0.5, 1.0)


def _check_speed(speed: float) -> float:
    if not (math.isfinite(speed) and speed > 0):
        raise ValueError(f"speed must be a positive finite number, got {speed}")
    return float(speed)


class PlaybackClock:
    """Maps wall-clock seconds to a frame index of a run, looping at the end.

    Invariant: ``time_s`` is 1-D, finite, strictly increasing, with at least
    two samples; ``speed`` is positive and finite.
    """

    def __init__(self, time_s: NDArray[np.float64], speed: float = 1.0) -> None:
        times = np.asarray(time_s, dtype=float)
        if (
            times.ndim != 1
            or times.size < 2
            or not np.all(np.isfinite(times))
            or not np.all(np.diff(times) > 0)
        ):
            raise ValueError(
                "time axis must be 1-D, finite, strictly increasing, >= 2 samples"
            )
        self._time = times
        self._duration = float(times[-1] - times[0])
        self.speed = _check_speed(speed)
        self._wall0 = 0.0
        self._offset0 = 0.0

    def start(self, frame: int, now_s: float) -> None:
        """Begin (or rebase) playback from ``frame`` at wall time ``now_s``."""
        if not 0 <= frame < self._time.size:
            raise ValueError(
                f"frame {frame} is outside the run (0..{self._time.size - 1})"
            )
        self._wall0 = float(now_s)
        self._offset0 = float(self._time[frame] - self._time[0])

    def set_speed(self, speed: float, now_s: float) -> None:
        """Change speed without moving: the current frame stays current."""
        speed = _check_speed(speed)
        self._offset0 = self._offset_at(now_s)
        self._wall0 = float(now_s)
        self.speed = speed

    def _offset_at(self, now_s: float) -> float:
        raw = self._offset0 + (now_s - self._wall0) * self.speed
        return raw % self._duration

    def frame_at(self, now_s: float) -> int:
        """Index of the sample nearest the media time reached at ``now_s``."""
        target = self._time[0] + self._offset_at(now_s)
        upper = int(np.searchsorted(self._time, target))
        if upper == 0:
            return 0
        if upper >= self._time.size:
            return self._time.size - 1
        before, after = self._time[upper - 1], self._time[upper]
        return upper - 1 if target - before <= after - target else upper
