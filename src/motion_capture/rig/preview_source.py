"""Low-rate decoded preview of a camera through ffmpeg + DirectShow.

The recorder proves that ffmpeg addressing a camera by its DirectShow device
reference streams every rig camera at full rate; OpenCV's own device-index
backends do not (Media Foundation hangs on the third unit, DirectShow refuses
the full mode by index). So the live preview decodes through the same ffmpeg
path: one process per camera reading the compressed MJPEG at the capture
mode and writing downscaled raw BGR frames at a modest rate to a pipe.

That decode now lives in the fleet's shared camera layer, Tools
``shared.python.camera`` (ported from this module and pinned there, #10204).
This module keeps only the rig-facing seam: :class:`SharedSourceAdapter`
presents a Tools :class:`~shared.python.sidekick.lab.mocap.acquisition.FrameSource`
through the rig's own :class:`~.sources.FrameSource` protocol, and the
plan-driven constructors bind it to cameras. The seam points one way — the
rig consumes the shared layer; nothing in Tools knows the rig exists.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from shared.python.camera import CaptureMode as SharedCaptureMode
from shared.python.camera import FfmpegDirectShowSource
from shared.python.contracts import StateError as SharedStateError
from shared.python.sidekick.lab.mocap.acquisition import FramePacket
from shared.python.sidekick.lab.mocap.acquisition import (
    FrameSource as SharedFrameSource,
)

from src.shared.python.core.contracts import StateError, require

from .plan import CameraControls, CaptureMode
from .recorder import require_pnp_instance_id
from .sources import Frame

PREVIEW_WIDTH = 640
PREVIEW_FPS = 12
_FOURCC_BY_PIXEL_FORMAT = {"bgr24": "BGR3"}

SharedSourceFactory = Callable[[SharedCaptureMode], SharedFrameSource]


def shared_capture_mode(mode: CaptureMode) -> SharedCaptureMode:
    """The rig's plan mode as the shared layer's mode (same four fields)."""
    return SharedCaptureMode(
        width=mode.width, height=mode.height, fps=mode.fps, fourcc=mode.fourcc
    )


def _frame_from_packet(packet: FramePacket) -> Frame:
    """A rig :class:`Frame` over a shared packet's pixels (no copy).

    The rig timestamps arrival on the host clock (ADR-0041), which the packet
    carries as ``host_monotonic_ns``; the packet's index-derived
    ``timestamp_ns`` is not a rig clock domain and is not used.
    """
    width, height = packet.resolution_px
    image = np.frombuffer(packet.image_bytes, dtype=np.uint8).reshape(height, width, 3)
    return Frame(image=image, seq=packet.sequence_number, t_ns=packet.host_monotonic_ns)


class SharedSourceAdapter:
    """The rig's ``open/read/close`` over a Tools ``initialize/start_capture/read_frame``.

    ``factory`` builds a fresh shared source for the mode ``open`` negotiates;
    the adapter owns it until ``close``. ``open`` proves the first frame (and
    consumes it) so a device that refuses fails at open time, ``read`` returns
    ``None`` once the stream ends — the rig contract is "nothing arrived",
    never an exception — and ``close`` is idempotent.
    """

    def __init__(self, identity: str, factory: SharedSourceFactory) -> None:
        require(bool(identity), "identity must be non-empty")
        self._identity = identity
        self._factory = factory
        self._source: SharedFrameSource | None = None

    @property
    def identity(self) -> str:
        return self._identity

    def open(
        self, mode: CaptureMode, controls: CameraControls | None = None
    ) -> CaptureMode:
        """Start the shared source; returns the mode it emits (size, rate, pixels).

        UVC ``controls`` are not applied on the preview path (the recorder
        applies them at capture time). Raises ``StateError`` if the source
        ends before its first frame, quoting what ffmpeg said.
        """
        self.close()
        source = self._factory(shared_capture_mode(mode))
        source.initialize()
        source.start_capture()
        self._source = source
        try:
            source.read_frame()
        except SharedStateError as exc:
            self.close()
            raise StateError(
                f"ffmpeg preview of {self._identity} produced no frame: {exc}"
            ) from exc
        caps = source.capabilities
        (width, height), fps = caps.resolutions_px[0], caps.frame_rates_hz[0]
        pixel_format = caps.pixel_formats[0]
        require(
            pixel_format in _FOURCC_BY_PIXEL_FORMAT, "a rig pixel format", pixel_format
        )
        return CaptureMode(
            width=width,
            height=height,
            fps=round(fps),
            fourcc=_FOURCC_BY_PIXEL_FORMAT[pixel_format],
        )

    def read(self) -> Frame | None:
        """The next decoded frame, or ``None`` when the stream has ended."""
        source = self._source
        if source is None:
            raise StateError("read() before open()")
        try:
            packet = source.read_frame()
        except SharedStateError:
            return None
        return _frame_from_packet(packet)

    def close(self) -> None:
        source, self._source = self._source, None
        if source is not None:
            source.close()


class FfmpegPreviewSource(SharedSourceAdapter):
    """A rig :class:`~.sources.FrameSource` over the shared ffmpeg DirectShow decode.

    Precondition: a non-empty identity, a PnP camera instance id and positive
    preview ``width``/``fps``. ``popen`` is forwarded to the shared source so
    the lifecycle is testable without a device.
    """

    def __init__(
        self,
        identity: str,
        camera_instance_id: str,
        *,
        ffmpeg_exe: str | None = None,
        width: int = PREVIEW_WIDTH,
        fps: int = PREVIEW_FPS,
        popen: Any = None,
    ) -> None:
        require(width > 0 and fps > 0, "positive preview width and fps", (width, fps))
        require_pnp_instance_id(camera_instance_id)
        self.camera_instance_id = camera_instance_id

        def build(mode: SharedCaptureMode) -> SharedFrameSource:
            return FfmpegDirectShowSource(
                camera_instance_id,
                mode,
                width=width,
                fps=fps,
                ffmpeg_exe=ffmpeg_exe,
                popen=popen,
            )

        super().__init__(identity, build)


def preview_sources_from_ids(
    plan: Any, camera_ids: Mapping[str, str], **kwargs: Any
) -> dict[str, FfmpegPreviewSource]:
    """Preview sources for already-bound cameras: no enumeration (seconds, not 30 s).

    Precondition: ``camera_ids`` names a DirectShow instance id for every
    plan view.
    """
    missing = [c.view for c in plan.cameras if c.view not in camera_ids]
    require(not missing, "camera_ids must cover every plan view", missing)
    return {
        c.view: FfmpegPreviewSource(c.identity, camera_ids[c.view], **kwargs)
        for c in plan.cameras
    }


def ffmpeg_preview_sources(
    plan: Any, cams: Sequence[Any] | None = None, **kwargs: Any
) -> dict[str, FfmpegPreviewSource]:
    """A preview source per plan view, bound like the recorder binds cameras."""
    from .binding import locate_plan

    return {
        view: FfmpegPreviewSource(cam.identity, cam.camera, **kwargs)
        for view, cam in locate_plan(plan, cams).items()
    }
