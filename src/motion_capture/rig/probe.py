"""Decode-probe a recording to learn what was actually written.

A recorder's exit code says the process ended cleanly; it says nothing about
how many frames landed on disk. On the real rig the first bring-up recordings
held 7.9–9.0 s of a requested 10 s because DirectShow devices take one to two
seconds to open and recorders were stopped one after another. The probe reads
the file back through the bundled ffmpeg and reports frames, duration, size
and nominal rate, so the session bundle can call a short recording
``degraded`` instead of ``supported``.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from src.shared.python.core.contracts import require

_FRAME = re.compile(r"frame=\s*(\d+)")
_TIME = re.compile(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)")
_STREAM = re.compile(r"Stream #0:0.*?(\d{2,5})x(\d{2,5}).*?(\d+(?:\.\d+)?) fps")


@dataclass(frozen=True)
class RecordingProbe:
    """What a decode pass found in one recording."""

    frames: int
    duration_s: float
    width: int | None
    height: int | None
    nominal_fps: float | None

    @property
    def achieved_fps(self) -> float | None:
        if self.duration_s <= 0:
            return None
        return self.frames / self.duration_s


Prober = Callable[[Path], RecordingProbe]


def parse_ffmpeg_decode_log(text: str) -> RecordingProbe:
    """Extract frames, decoded duration and stream geometry from ffmpeg stderr.

    ``frame=`` and ``time=`` appear on progress lines; the last occurrence wins.
    Geometry comes from the ``Stream #0:0`` line when present.
    """
    frames_found = _FRAME.findall(text)
    times = _TIME.findall(text)
    frames = int(frames_found[-1]) if frames_found else 0
    duration = 0.0
    if times:
        h, m, s = times[-1]
        duration = int(h) * 3600 + int(m) * 60 + float(s)
    width = height = None
    fps: float | None = None
    if stream := _STREAM.search(text):
        width, height, fps = (
            int(stream.group(1)),
            int(stream.group(2)),
            float(stream.group(3)),
        )
    return RecordingProbe(
        frames=frames, duration_s=duration, width=width, height=height, nominal_fps=fps
    )


def probe_recording(path: Path, *, timeout_s: float = 300.0) -> RecordingProbe:
    """Decode ``path`` to null output with the bundled ffmpeg and parse the log."""
    require(path.is_file(), "recording must exist", str(path))
    require(timeout_s > 0, "timeout_s must be positive", timeout_s)
    import imageio_ffmpeg

    result = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-i",
            str(path),
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    return parse_ffmpeg_decode_log(result.stderr + result.stdout)
