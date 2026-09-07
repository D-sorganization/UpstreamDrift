"""Compressed-stream recorders.

Decoding MJPEG to BGR during capture costs about two cores per 1920x1200 @ 60
stream; copying the compressed stream to disk costs almost nothing (measured:
60 fps, 11.2 MB/s per camera). A recorder therefore owns the camera for the
duration of a recording — Windows grants one process exclusive access — so a
session either *observes* through :mod:`.sources` or *records* through this
module, never both on the same camera at once.

Timing lesson from the rig: a DirectShow device takes one to two seconds to
open inside ffmpeg, and stopping recorders one after another lets the last
one run longest. :func:`record_all` therefore starts every recorder, waits a
warm-up before the duration clock starts, and stops all of them together —
signal first, then reap — so the views cover the same interval as closely as
the hardware allows. The bundle still probes what actually landed on disk.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from src.shared.python.core.contracts import StateError, require
from src.shared.python.core.process_safety import managed_popen
from src.shared.python.logging_pkg.logging_config import get_logger

from .plan import CaptureMode, RigPlan

logger = get_logger(__name__)

# KSCATEGORY_VIDEO_CAMERA — the DirectShow "alternative name" suffix.
_DSHOW_CATEGORY = "{65e8773d-8f56-11d0-a3b9-00a0c9223196}"
DEFAULT_WARMUP_S = 2.0


@dataclass(frozen=True)
class RecordingResult:
    """Outcome of one camera's recording."""

    identity: str
    path: Path
    returncode: int | None
    bytes_written: int

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.bytes_written > 0


class Recorder(Protocol):
    """Start / signal / stop lifecycle of one camera recording."""

    def start(
        self, identity: str, device_ref: str, mode: CaptureMode, path: Path
    ) -> None: ...

    def signal_stop(self) -> None:
        """Ask the recorder to finish; returns immediately."""
        ...

    def stop(self) -> RecordingResult:
        """Wait for the recorder to finish and report."""
        ...


def dshow_device_ref(camera_instance_id: str) -> str:
    """DirectShow device path for a PnP camera instance id.

    ``USB\\VID_32E4&PID_5234&MI_00\\6&FADBF3B&0&0000`` becomes
    ``@device_pnp_\\\\?\\usb#vid_32e4&pid_5234&mi_00#6&fadbf3b&0&0000#{...}\\global``
    — the inverse of :func:`.topology.parse_dshow_listing`. Addressing a camera
    this way is what keeps three identically named units unambiguous.
    """
    require(
        "\\" in camera_instance_id, "expected a PnP instance id", camera_instance_id
    )
    body = camera_instance_id.lower().replace("\\", "#")
    return f"@device_pnp_\\\\?\\{body}#{_DSHOW_CATEGORY}\\global"


def ffmpeg_stream_copy_args(
    ffmpeg_exe: str, device_ref: str, mode: CaptureMode, path: Path
) -> list[str]:
    """ffmpeg command that copies the camera's compressed stream to ``path``."""
    codec = "mjpeg" if mode.fourcc == "MJPG" else mode.fourcc.lower()
    return [
        ffmpeg_exe,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "dshow",
        "-vcodec",
        codec,
        "-video_size",
        f"{mode.width}x{mode.height}",
        "-framerate",
        str(mode.fps),
        "-i",
        f"video={device_ref}",
        "-c:v",
        "copy",
        "-y",
        str(path),
    ]


class NullRecorder:
    """Records nothing; useful for dry runs and tests."""

    def __init__(self) -> None:
        self._identity: str | None = None
        self._path: Path | None = None
        self.signalled = False

    def start(
        self, identity: str, device_ref: str, mode: CaptureMode, path: Path
    ) -> None:
        self._identity, self._path = identity, path

    def signal_stop(self) -> None:
        self.signalled = True

    def stop(self) -> RecordingResult:
        if self._identity is None or self._path is None:
            raise StateError("stop() before start()")
        result = RecordingResult(self._identity, self._path, 0, 0)
        self._identity = self._path = None
        return result


class FfmpegStreamCopyRecorder:
    """Stream-copies one DirectShow camera with the bundled imageio-ffmpeg binary."""

    def __init__(
        self, ffmpeg_exe: str | None = None, *, stop_timeout_s: float = 10.0
    ) -> None:
        require(stop_timeout_s > 0, "stop_timeout_s must be positive", stop_timeout_s)
        self._ffmpeg = ffmpeg_exe
        self._stop_timeout = stop_timeout_s
        self._stack: ExitStack | None = None
        self._proc: Any = None
        self._identity: str | None = None
        self._path: Path | None = None

    def _exe(self) -> str:
        if self._ffmpeg is None:
            import imageio_ffmpeg

            self._ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        return self._ffmpeg

    def start(
        self, identity: str, device_ref: str, mode: CaptureMode, path: Path
    ) -> None:
        import subprocess

        if self._stack is not None:
            raise StateError(f"recorder already running for {self._identity}")
        path.parent.mkdir(parents=True, exist_ok=True)
        args = ffmpeg_stream_copy_args(self._exe(), device_ref, mode, path)
        stack = ExitStack()
        self._proc = stack.enter_context(
            managed_popen(
                args,
                timeout=self._stop_timeout,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        )
        self._stack, self._identity, self._path = stack, identity, path
        logger.info("recording %s -> %s", identity, path)

    def signal_stop(self) -> None:
        """Send ffmpeg its graceful quit key; safe when already closed."""
        proc = self._proc
        if proc is None or proc.stdin is None:
            return
        try:
            proc.stdin.write("q\n")
            proc.stdin.flush()
        except (OSError, ValueError):
            logger.warning("ffmpeg stdin already closed for %s", self._identity)

    def stop(self) -> RecordingResult:
        if self._stack is None or self._identity is None or self._path is None:
            raise StateError("stop() before start()")
        self.signal_stop()
        self._stack.close()  # waits up to stop_timeout, then terminates
        size = self._path.stat().st_size if self._path.exists() else 0
        result = RecordingResult(
            self._identity, self._path, self._proc.returncode, size
        )
        self._stack = self._proc = self._identity = self._path = None
        return result


def record_all(
    plan: RigPlan,
    device_refs: Mapping[str, str],
    duration_s: float,
    out_dir: Path,
    recorder_factory: Callable[[], Recorder],
    *,
    warmup_s: float = DEFAULT_WARMUP_S,
    sleep: Callable[[float], None] = time.sleep,
) -> list[RecordingResult]:
    """Record every planned view together for ``duration_s`` seconds.

    Starts every recorder, waits ``warmup_s`` for the devices to open, runs the
    duration clock, then signals every recorder before reaping any — so all
    views stop within milliseconds of each other rather than in sequence.
    Precondition: ``device_refs`` maps every plan view to a DirectShow device ref.
    """
    require(duration_s > 0, "duration_s must be positive", duration_s)
    require(warmup_s >= 0, "warmup_s must be non-negative", warmup_s)
    missing = [c.view for c in plan.cameras if c.view not in device_refs]
    require(not missing, "device_refs missing views", missing)
    out_dir.mkdir(parents=True, exist_ok=True)
    active: list[Recorder] = []
    for binding in plan.cameras:
        rec = recorder_factory()
        rec.start(
            binding.identity,
            device_refs[binding.view],
            binding.mode,
            out_dir / f"{binding.view}_{binding.identity}.mkv",
        )
        active.append(rec)
    sleep(warmup_s + duration_s)
    for rec in active:
        rec.signal_stop()
    return [rec.stop() for rec in active]
