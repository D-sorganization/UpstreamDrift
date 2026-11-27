"""Swing time of every frame of an exported clip (#15722, SRP-3).

The exporter writes ``<clip>.times.json`` beside each clip: the swing time
(s) each video frame shows. Clips re-sample the swing, hold the contact frame
and may hold the end for the ball flight, so a frame's time cannot be
recovered from the frame rate alone; freeze-frame sheets read this file.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path

__all__ = ["frame_at_time", "frame_times_path", "load_frame_times", "write_frame_times"]

_SUFFIX = ".times.json"


def frame_times_path(clip: Path) -> Path:
    """The frame-times sidecar of ``clip``."""
    clip = Path(clip)
    return clip.with_name(clip.stem + _SUFFIX)


def write_frame_times(paths: Mapping[str, Path], times_s: Sequence[float]) -> None:
    """Write ``times_s`` beside every clip of ``paths`` (one time per frame).

    Raises:
        ValueError: a time is not finite.
    """
    times = [float(t) for t in times_s]
    if not all(math.isfinite(t) for t in times):
        raise ValueError("frame times must be finite")
    doc = json.dumps({"schema_version": 1, "time_s": times})
    for clip in paths.values():
        frame_times_path(clip).write_text(doc, encoding="utf-8")


def load_frame_times(clip: Path) -> tuple[float, ...]:
    """The frame times of ``clip``; ``FileNotFoundError`` without a sidecar."""
    doc = json.loads(frame_times_path(clip).read_text(encoding="utf-8"))
    return tuple(float(t) for t in doc["time_s"])


def frame_at_time(times_s: Sequence[float], time_s: float) -> int:
    """The first frame whose time is nearest ``time_s``.

    Raises:
        ValueError: no frames, or ``time_s`` outside the clip's times.
    """
    if not times_s:
        raise ValueError("the clip has no frames")
    if not min(times_s) - 1e-9 <= time_s <= max(times_s) + 1e-9:
        raise ValueError(
            f"time {time_s} s lies outside the clip ({min(times_s)}..{max(times_s)} s)"
        )
    gaps = [abs(t - time_s) for t in times_s]
    return gaps.index(min(gaps))
