"""Impact-zone slow motion for the results playback (#15524, epic #15150).

Qt-free. :class:`ImpactZoneClock` is a :class:`PlaybackClock` whose media
time runs ``slow_factor`` times slower inside ``impact_s +/- half_width_s``,
so the strike (club, ball and the flight launch) can be watched while the
rest of the swing plays at the chosen speed. Frames still follow wall time,
so a slow render skips frames rather than stretching the zone.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from src.tools.model_explorer.playback_clock import PlaybackClock

__all__ = ["ImpactZoneClock"]


class ImpactZoneClock(PlaybackClock):
    """A clock that plays ``slow_factor`` times slower around the impact.

    Preconditions: ``impact_s`` is finite and inside the run, ``half_width_s``
    is positive and finite, ``0 < slow_factor <= 1`` (``1`` is plain
    playback). The zone is clamped to the run. Postcondition: one loop takes
    ``loop_wall_s`` of wall time at the current ``speed``.
    """

    def __init__(
        self,
        time_s: NDArray[np.float64],
        impact_s: float,
        half_width_s: float,
        slow_factor: float,
        speed: float = 1.0,
    ) -> None:
        super().__init__(time_s, speed)
        t0, t1 = float(self._time[0]), float(self._time[-1])
        if not (math.isfinite(impact_s) and t0 <= impact_s <= t1):
            raise ValueError(f"impact_s {impact_s} is outside the run ({t0}..{t1})")
        if not (math.isfinite(half_width_s) and half_width_s > 0):
            raise ValueError("half_width_s must be positive and finite")
        if not (math.isfinite(slow_factor) and 0 < slow_factor <= 1):
            raise ValueError("slow_factor must be in (0, 1]")
        self._slow = float(slow_factor)
        self._zone = (
            max(0.0, impact_s - half_width_s - t0),
            min(self._duration, impact_s + half_width_s - t0),
        )
        self._phase0 = 0.0  # loop-wall position at ``_wall0``

    @property
    def loop_wall_s(self) -> float:
        """Wall seconds one pass of the run takes at the current speed."""
        return self._media_to_wall(self._duration)

    def _media_to_wall(self, offset_s: float) -> float:
        lo, hi = self._zone
        lead = min(offset_s, lo)
        zone = min(max(offset_s - lo, 0.0), hi - lo)
        tail = max(offset_s - hi, 0.0)
        return (lead + zone / self._slow + tail) / self.speed

    def _wall_to_media(self, wall_s: float) -> float:
        lo, hi = self._zone
        media = wall_s * self.speed
        if media <= lo:
            return media
        media -= lo
        if media <= (hi - lo) / self._slow:
            return lo + media * self._slow
        return hi + media - (hi - lo) / self._slow

    def media_time_at(self, now_s: float) -> float:
        """The run time shown at wall time ``now_s`` (looping)."""
        return float(self._time[0]) + self._offset_at(now_s)

    def _offset_at(self, now_s: float) -> float:
        loop = self.loop_wall_s
        wall = (self._phase0 + (now_s - self._wall0)) % loop
        return min(self._wall_to_media(wall), self._duration)

    def start(self, frame: int, now_s: float) -> None:
        """Begin (or rebase) playback from ``frame`` at wall time ``now_s``."""
        super().start(frame, now_s)
        self._phase0 = self._media_to_wall(self._offset0)

    def set_speed(self, speed: float, now_s: float) -> None:
        """Change speed without moving: the current frame stays current."""
        offset = self._offset_at(now_s)
        super().set_speed(speed, now_s)
        self._phase0 = self._media_to_wall(offset)
