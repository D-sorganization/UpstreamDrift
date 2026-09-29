"""Frame sources: the seam between the rig and whatever produces images.

Every frame carries a host-monotonic timestamp and names its clock domain.
Arrival time is not exposure time (ADR-0041); the timing records that relate
the two belong to the session manifest, never to the frame.

Two implementations ship: :class:`SyntheticFrameSource` for deterministic,
hardware-free tests and :class:`OpenCvMsmfSource` for the ELP AR0234 units on
Windows. Both honour the same lifecycle — ``open`` negotiates a mode and proves
frames arrive, ``read`` never blocks forever, ``close`` is idempotent — because
a lost isochronous reservation is permanent on an OpenCV handle and the only
recovery is to recreate it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from src.shared.python.core.contracts import StateError, require
from src.shared.python.logging_pkg.logging_config import get_logger

from .plan import CameraControls, CaptureMode

logger = get_logger(__name__)

HOST_MONOTONIC = "host_monotonic_ns"


@dataclass(frozen=True)
class Frame:
    """One captured image with its arrival timestamp."""

    image: np.ndarray
    seq: int
    t_ns: int
    clock_domain: str = HOST_MONOTONIC

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.image.shape)


@runtime_checkable
class FrameSource(Protocol):
    """Lifecycle every frame producer implements."""

    @property
    def identity(self) -> str: ...

    def open(
        self, mode: CaptureMode, controls: CameraControls | None = None
    ) -> CaptureMode:
        """Negotiate ``mode``; return the effective mode. Raises on failure."""
        ...

    def read(self) -> Frame | None:
        """Return the next frame, or ``None`` when none arrived."""
        ...

    def close(self) -> None: ...


class SyntheticFrameSource:
    """Deterministic frames at a configured rate, with optional fault injection.

    ``fail_after`` makes every read after that many frames return ``None``
    (models a lost isochronous reservation); ``flash_at`` brightens exactly one
    frame (models the strobe used for time alignment).
    """

    def __init__(
        self,
        identity: str,
        *,
        fps: float | None = None,
        size: tuple[int, int] = (16, 12),
        fail_after: int | None = None,
        flash_at: int | None = None,
        realtime: bool = False,
    ) -> None:
        require(bool(identity), "identity must be non-empty")
        require(fps is None or fps > 0, "fps must be positive", fps)
        self._identity = identity
        self._fps_override = fps
        self._size = size
        self._fail_after = fail_after
        self._flash_at = flash_at
        self._realtime = realtime
        self._mode: CaptureMode | None = None
        self._seq = 0
        self._t0_ns: int | None = None
        self.controls_applied: dict[str, float] = {}

    @property
    def identity(self) -> str:
        return self._identity

    @property
    def effective_fps(self) -> float:
        mode = self._mode
        if mode is None:
            raise StateError("source not open")
        return self._fps_override or float(mode.fps)

    def open(
        self, mode: CaptureMode, controls: CameraControls | None = None
    ) -> CaptureMode:
        self._mode = mode
        self._seq = 0
        self._t0_ns = time.monotonic_ns()
        self.controls_applied = controls.as_overrides() if controls else {}
        return mode

    def read(self) -> Frame | None:
        if self._mode is None or self._t0_ns is None:
            raise StateError("read() before open()")
        if self._fail_after is not None and self._seq >= self._fail_after:
            return None
        period_ns = int(1e9 / self.effective_fps)
        if self._realtime:
            time.sleep(period_ns / 1e9)
        w, h = self._size
        level = 255 if self._seq == self._flash_at else 32
        image = np.full((h, w, 3), level, dtype=np.uint8)
        frame = Frame(
            image=image, seq=self._seq, t_ns=self._t0_ns + self._seq * period_ns
        )
        self._seq += 1
        return frame

    def close(self) -> None:
        self._mode = None


class OpenCvMsmfSource:
    """An ELP camera through OpenCV's Media Foundation backend (Windows).

    ``open`` requests the mode, applies control overrides, then requires
    ``warmup_frames`` real frames before declaring success: an ``isOpened()``
    handle whose reservation was refused reads ``False`` forever, and that
    must surface at open time rather than as a silent empty stream.
    """

    def __init__(self, identity: str, index: int, *, warmup_frames: int = 5) -> None:
        require(bool(identity), "identity must be non-empty")
        require(index >= 0, "index must be non-negative", index)
        require(warmup_frames >= 1, "warmup_frames must be >= 1", warmup_frames)
        self._identity = identity
        self._index = index
        self._warmup = warmup_frames
        self._cap: Any = None
        self._seq = 0

    @property
    def identity(self) -> str:
        return self._identity

    def open(
        self, mode: CaptureMode, controls: CameraControls | None = None
    ) -> CaptureMode:
        import cv2

        self.close()
        cap = cv2.VideoCapture(self._index, cv2.CAP_MSMF)
        if not cap.isOpened():
            raise StateError(
                f"camera {self._identity} (index {self._index}) would not open"
            )
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*mode.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, mode.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, mode.height)
        cap.set(cv2.CAP_PROP_FPS, mode.fps)
        self._apply_controls(cap, controls)
        if not any(cap.read()[0] for _ in range(self._warmup)):
            cap.release()
            raise StateError(
                f"camera {self._identity} opened but delivered no frames "
                "(isochronous reservation refused? one camera per USB 2.0 root port)"
            )
        self._cap = cap
        self._seq = 0
        return CaptureMode(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=int(round(cap.get(cv2.CAP_PROP_FPS))) or mode.fps,
            fourcc=mode.fourcc,
        )

    @staticmethod
    def _apply_controls(cap: Any, controls: CameraControls | None) -> None:
        if controls is None:
            return
        import cv2

        props = {
            "exposure": cv2.CAP_PROP_EXPOSURE,
            "gain": cv2.CAP_PROP_GAIN,
            "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
        }
        for name, value in controls.as_overrides().items():
            if not cap.set(props[name], value):
                logger.warning("camera control %s=%s not accepted", name, value)

    def read(self) -> Frame | None:
        if self._cap is None:
            raise StateError("read() before open()")
        ok, image = self._cap.read()
        if not ok or image is None:
            return None
        frame = Frame(image=image, seq=self._seq, t_ns=time.monotonic_ns())
        self._seq += 1
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
