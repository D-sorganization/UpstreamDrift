"""H.264-first MP4 writer shared by every app video export (#12947, #12878).

``cv2.VideoWriter`` with the ``mp4v`` fourcc produces MPEG-4 Part 2, which
browsers and web players cannot preview. :class:`Mp4Writer` is a drop-in
replacement that

1. prefers H.264 (``avc1``) when OpenCV has an encoder for it;
2. otherwise writes ``mp4v`` and, on :meth:`Mp4Writer.release`, transcodes the
   file to H.264 with ``ffmpeg`` (on ``PATH``, else the one bundled with
   ``imageio-ffmpeg``);
3. otherwise keeps ``mp4v`` and logs a warning.

Postcondition of ``release()``: the file at ``path`` is complete; ``codec``
names what it really contains (``"avc1"`` for H.264, ``"mp4v"``, or ``""`` if
no writer could be opened).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.logging_pkg.logging_config import get_logger

__all__ = ["H264", "MPEG4", "Mp4Writer", "find_ffmpeg"]

H264 = "avc1"
MPEG4 = "mp4v"
_TRANSCODE_TIMEOUT_S = 600
logger = get_logger(__name__)


def find_ffmpeg() -> str | None:
    """Return an ``ffmpeg`` executable (``PATH`` first, then ``imageio-ffmpeg``)."""
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path
    try:
        import imageio_ffmpeg

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except (ImportError, RuntimeError, OSError):
        return None


def _open_cv_writer(path: Path, codec: str, fps: float, size: tuple[int, int]) -> Any:
    """Open an OpenCV writer for ``codec``; the caller checks ``isOpened()``."""
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*codec)  # type: ignore[attr-defined]
    return cv2.VideoWriter(str(path), fourcc, float(fps), size)


def _transcode_to_h264(path: Path) -> bool:
    """Re-encode ``path`` to H.264 in place; ``False`` leaves it untouched."""
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        logger.warning(
            "OpenCV has no H.264 encoder and ffmpeg was not found: %s keeps the "
            "mp4v codec, which browsers cannot preview. Install ffmpeg or "
            "imageio-ffmpeg for H.264 output.",
            path,
        )
        return False
    temp = path.with_name(f"{path.stem}.h264tmp{path.suffix}")
    command = [
        ffmpeg, "-y", "-loglevel", "error", "-i", str(path),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(temp),
    ]  # fmt: skip
    try:
        subprocess.run(  # noqa: S603 - fixed argv, no shell
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=_TRANSCODE_TIMEOUT_S,
        )
        os.replace(temp, path)
    except (subprocess.SubprocessError, OSError) as exc:
        detail = getattr(exc, "stderr", None) or exc
        logger.warning(
            "ffmpeg H.264 transcode of %s failed (%s); keeping mp4v", path, detail
        )
        temp.unlink(missing_ok=True)
        return False
    return True


class Mp4Writer:
    """``cv2.VideoWriter``-compatible MP4 writer that ends up as H.264 if it can.

    Frames are BGR ``uint8`` arrays of ``size`` ``(width, height)``.
    """

    def __init__(self, path: str | Path, fps: float, size: Sequence[int]) -> None:
        if not fps > 0:
            raise ValueError(f"fps must be positive, got {fps}")
        if len(size) != 2 or min(size) <= 0:
            raise ValueError(f"size must be (width, height) > 0, got {tuple(size)}")
        self._path = Path(path)
        self._size = (int(size[0]), int(size[1]))
        self._fps = float(fps)
        self._writer: Any = None
        self.codec = ""
        for codec in (H264, MPEG4):
            candidate = _open_cv_writer(self._path, codec, self._fps, self._size)
            if candidate.isOpened():
                self._writer, self.codec = candidate, codec
                break
            candidate.release()

    def isOpened(self) -> bool:  # noqa: N802 - mirrors cv2.VideoWriter
        return self._writer is not None

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            raise RuntimeError("Mp4Writer is not open")
        self._writer.write(frame)

    def release(self) -> None:
        """Finish the file (transcoding to H.264 if needed); idempotent."""
        writer, self._writer = self._writer, None
        if writer is None:
            return
        writer.release()
        if self.codec == MPEG4 and _transcode_to_h264(self._path):
            self.codec = H264

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:  # noqa: BLE001 - never raise from a finalizer
            pass
