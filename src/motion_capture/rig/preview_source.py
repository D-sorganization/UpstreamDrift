"""Low-rate decoded preview of a camera through ffmpeg + DirectShow.

The recorder proves that ffmpeg addressing a camera by its DirectShow device
reference streams every rig camera at full rate; OpenCV's own device-index
backends do not (Media Foundation hangs on the third unit, DirectShow refuses
the full mode by index). So the live preview decodes through the same ffmpeg
path: one process per camera reading the compressed MJPEG at the capture
mode and writing downscaled raw BGR frames at a modest rate to a pipe.
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Sequence
from contextlib import ExitStack
from typing import Any

import numpy as np

from src.shared.python.core.contracts import StateError, require
from src.shared.python.core.process_safety import managed_popen

from .plan import CameraControls, CaptureMode
from .recorder import dshow_device_ref
from .sources import Frame

PREVIEW_WIDTH = 640
PREVIEW_FPS = 12
STOP_TIMEOUT_S = 3.0


def ffmpeg_preview_args(
    ffmpeg_exe: str,
    device_ref: str,
    mode: CaptureMode,
    *,
    width: int = PREVIEW_WIDTH,
    fps: int = PREVIEW_FPS,
) -> list[str]:
    """ffmpeg command: capture at ``mode``, emit ``width``-wide BGR frames at ``fps``.

    Precondition: positive width and fps. The output height keeps the aspect
    ratio (``-2`` lets ffmpeg round to an even number).
    """
    require(width > 0 and fps > 0, "positive preview width and fps", (width, fps))
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
        "-vf",
        f"fps={fps},scale={width}:-2",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "pipe:1",
    ]


def preview_frame_size(
    mode: CaptureMode, width: int = PREVIEW_WIDTH
) -> tuple[int, int]:
    """``(width, height)`` of the preview frames ffmpeg will emit for ``mode``."""
    height = int(round(mode.height * width / mode.width / 2) * 2)
    return width, max(height, 2)


class FfmpegPreviewSource:
    """A :class:`FrameSource` backed by an ffmpeg decode of one DirectShow camera."""

    def __init__(
        self,
        identity: str,
        camera_instance_id: str,
        *,
        ffmpeg_exe: str | None = None,
        width: int = PREVIEW_WIDTH,
        fps: int = PREVIEW_FPS,
    ) -> None:
        require(bool(identity), "identity must be non-empty")
        self._identity = identity
        self._device_ref = dshow_device_ref(camera_instance_id)
        self._ffmpeg = ffmpeg_exe
        self._width, self._fps = width, fps
        self._stack: ExitStack | None = None
        self._proc: Any = None
        self._frame_shape: tuple[int, int] = (0, 0)
        self._seq = 0

    @property
    def identity(self) -> str:
        return self._identity

    def _exe(self) -> str:
        if self._ffmpeg is None:
            import imageio_ffmpeg

            self._ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        return self._ffmpeg

    def open(
        self, mode: CaptureMode, controls: CameraControls | None = None
    ) -> CaptureMode:
        """Start ffmpeg; returns the preview mode (downscaled size, preview fps).

        UVC ``controls`` are not applied on the preview path (the recorder
        applies them at capture time). Raises ``StateError`` if ffmpeg exits
        before the first frame.
        """
        self.close()
        args = ffmpeg_preview_args(
            self._exe(), self._device_ref, mode, width=self._width, fps=self._fps
        )
        stack = ExitStack()
        self._proc = stack.enter_context(
            managed_popen(
                args,
                timeout=STOP_TIMEOUT_S,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        )
        self._stack = stack
        width, height = preview_frame_size(mode, self._width)
        self._frame_shape = (height, width)
        self._seq = 0
        first = self.read()
        if first is None:
            err = self._stderr_tail()
            self.close()
            raise StateError(
                f"ffmpeg preview of {self._identity} produced no frame: {err}"
            )
        return CaptureMode(width=width, height=height, fps=self._fps, fourcc="BGR3")

    def read(self) -> Frame | None:
        """The next decoded frame, or ``None`` when ffmpeg has stopped."""
        proc = self._proc
        if proc is None or proc.stdout is None:
            raise StateError("read() before open()")
        height, width = self._frame_shape
        need = height * width * 3
        buf = bytearray()
        while len(buf) < need:
            chunk = proc.stdout.read(need - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
        image = np.frombuffer(bytes(buf), dtype=np.uint8).reshape(height, width, 3)
        frame = Frame(image=image, seq=self._seq, t_ns=time.monotonic_ns())
        self._seq += 1
        return frame

    def _stderr_tail(self) -> str:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return ""
        try:
            _out, err = proc.communicate(timeout=STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return "ffmpeg still running"
        return (err or b"")[-400:].decode("utf-8", "replace").strip()

    def close(self) -> None:
        proc, stack = self._proc, self._stack
        self._proc, self._stack = None, None
        if proc is not None and proc.poll() is None:
            proc.terminate()  # a raw-video pipe has no graceful quit key
        if stack is not None:
            stack.close()


def ffmpeg_preview_sources(
    plan: Any, cams: Sequence[Any] | None = None, **kwargs: Any
) -> dict[str, FfmpegPreviewSource]:
    """A preview source per plan view, bound like the recorder binds cameras."""
    from .binding import locate_plan

    return {
        view: FfmpegPreviewSource(cam.identity, cam.camera, **kwargs)
        for view, cam in locate_plan(plan, cams).items()
    }
