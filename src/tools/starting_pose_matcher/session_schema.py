"""Session-state schema helpers for the starting-pose matcher.

This module owns the ``playback`` block of the session JSON document
written by :mod:`src.tools.starting_pose_matcher.gui`.

The block was extended in issue #4482 (animated full-trajectory marker
preview) to record:

* ``current_frame`` — last viewed frame index
* ``speed``        — playback speed multiplier (0.1, 0.25, 0.5, 1, 2, 4)
* ``loop``         — whether playback wraps at the last frame
* ``trail_frames`` — number of trailing frames drawn behind moving markers

Older sessions (schema v3) that do not contain a ``playback`` block, or that
contain only a partial block, still load: missing keys fall back to the
defaults defined here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
from collections.abc import Mapping

# Allowed playback-speed multipliers exposed by the speed combo box.
ALLOWED_SPEEDS: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)

# Default trail length, in frames, used by the show-trail layer.
DEFAULT_TRAIL_FRAMES: int = 30


@dataclass
class PlaybackState:
    """Snapshot of the playback UI state.

    Attributes mirror the keys of the ``playback`` block in the session
    JSON document.
    """

    current_frame: int = 0
    speed: float = 1.0
    loop: bool = True
    trail_frames: int = DEFAULT_TRAIL_FRAMES

    def __post_init__(self) -> None:
        if self.current_frame < 0:
            raise ValueError(f"current_frame must be >= 0, got {self.current_frame}")
        if self.speed <= 0:
            raise ValueError(f"speed must be > 0, got {self.speed}")
        if self.trail_frames < 0:
            raise ValueError(f"trail_frames must be >= 0, got {self.trail_frames}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> PlaybackState:
        """Build a :class:`PlaybackState` from a possibly-partial mapping.

        Unknown keys are ignored. Missing keys fall back to the dataclass
        defaults so v3 sessions (which had no ``playback`` block, or only
        the legacy fields) load cleanly.
        """
        if not data:
            return cls()
        kwargs: dict[str, Any] = {}
        if "current_frame" in data:
            kwargs["current_frame"] = int(data["current_frame"])
        if "speed" in data:
            kwargs["speed"] = float(data["speed"])
        if "loop" in data:
            kwargs["loop"] = bool(data["loop"])
        if "trail_frames" in data:
            kwargs["trail_frames"] = int(data["trail_frames"])
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dict."""
        return asdict(self)

    def snap_speed_to_allowed(self) -> float:
        """Return the closest value in :data:`ALLOWED_SPEEDS` to ``self.speed``.

        Useful when an out-of-band speed value is read from disk and the UI
        needs to display the nearest combo-box choice.
        """
        return min(ALLOWED_SPEEDS, key=lambda s: abs(s - self.speed))


__all__ = [
    "ALLOWED_SPEEDS",
    "DEFAULT_TRAIL_FRAMES",
    "PlaybackState",
]
