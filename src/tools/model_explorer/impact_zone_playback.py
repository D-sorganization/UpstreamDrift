"""Switchable impact-zone slow motion for the results playback (#15524).

Qt-free. :class:`ImpactZonePlayback` owns a plain :class:`PlaybackClock` and,
when the run has an impact window, an :class:`ImpactZoneClock`. It exposes the
clock interface the panel already uses (``start``, ``set_speed``, ``frame_at``,
``speed``) and adds ``set_enabled``: switching slow motion on or off mid-play
rebases the new clock on the frame currently shown, so the picture never
jumps and the chosen speed is kept.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.tools.model_explorer.impact_zone_clock import ImpactZoneClock
from src.tools.model_explorer.playback_clock import PlaybackClock

__all__ = ["DEFAULT_SLOW_FACTOR", "ImpactZonePlayback"]

#: Media-time rate inside the impact zone relative to the surrounding swing.
DEFAULT_SLOW_FACTOR = 0.1


class ImpactZonePlayback:
    """A playback clock whose impact-zone slow motion can be toggled live.

    Preconditions: ``time_s`` is a valid playback time axis; ``window`` is
    ``None`` (no qualified impact: slow motion unavailable) or
    ``(impact_s, half_width_s)`` accepted by :class:`ImpactZoneClock`;
    ``slow_factor`` is in ``(0, 1]``. Invalid input raises ``ValueError``
    at construction, even when slow motion starts disabled.
    Invariant: ``enabled`` implies ``available``. Postcondition of
    ``set_enabled``: ``frame_at(now_s)`` and ``speed`` are unchanged.
    """

    def __init__(
        self,
        time_s: NDArray[np.float64],
        window: tuple[float, float] | None,
        *,
        slow_factor: float = DEFAULT_SLOW_FACTOR,
        speed: float = 1.0,
    ) -> None:
        self._plain = PlaybackClock(time_s, speed)
        self._slow: ImpactZoneClock | None = None
        if window is not None:
            impact_s, half_width_s = window
            self._slow = ImpactZoneClock(
                time_s, impact_s, half_width_s, slow_factor, speed
            )
        self._active: PlaybackClock = self._plain
        times = np.asarray(time_s, dtype=float)
        self._duration = float(times[-1] - times[0])

    @property
    def available(self) -> bool:
        """Whether the run has an impact window to slow down."""
        return self._slow is not None

    @property
    def enabled(self) -> bool:
        """Whether slow motion is currently applied."""
        return self._active is not self._plain

    @property
    def speed(self) -> float:
        """Playback speed relative to real time."""
        return self._active.speed

    @property
    def loop_wall_s(self) -> float:
        """Wall seconds one pass of the run takes at the current speed."""
        if isinstance(self._active, ImpactZoneClock):
            return self._active.loop_wall_s
        return self._duration / self._active.speed

    def set_enabled(self, enabled: bool, now_s: float) -> None:
        """Switch slow motion on or off without moving the current frame."""
        if enabled == self.enabled:
            return
        if enabled and self._slow is None:
            raise ValueError("no impact window: slow motion is unavailable")
        target: PlaybackClock = self._slow if enabled and self._slow else self._plain
        frame = self._active.frame_at(now_s)
        target.speed = self._active.speed
        target.start(frame, now_s)
        self._active = target

    def start(self, frame: int, now_s: float) -> None:
        """Begin (or rebase) playback from ``frame`` at wall time ``now_s``."""
        self._active.start(frame, now_s)

    def set_speed(self, speed: float, now_s: float) -> None:
        """Change speed without moving: the current frame stays current."""
        self._active.set_speed(speed, now_s)

    def frame_at(self, now_s: float) -> int:
        """Index of the sample shown at wall time ``now_s``."""
        return self._active.frame_at(now_s)
