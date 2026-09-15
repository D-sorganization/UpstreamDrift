"""The preview path delegates to the shared camera layer (Tools) — #10204.

``shared.python.camera`` was ported from this rig, so the command the
preview launches must be the one the rig validated on the lab cameras. The
first test pins that command against the pre-port argument list; the rest
drive the adapter through an injected fake ffmpeg process, so the whole
lifecycle — open proves a frame, read frames the pipe, EOF is ``None``,
close is idempotent — runs on a machine with no camera.

Runs under ``run_checks.py``: the shared package is a
``sidekick.lab.mocap`` consumer, which the root test process deliberately
keeps unresolvable (UpstreamDrift's own Sidekick is cached first) while the
launcher bootstrap resolves it from the pinned tree.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from shared.python.camera import CaptureMode as SharedCaptureMode
from shared.python.camera import dshow_device_ref as shared_device_ref
from shared.python.camera import ffmpeg_raw_frame_args

from src.motion_capture.rig.plan import CameraBinding, CaptureMode, RigPlan
from src.motion_capture.rig.preview_source import (
    PREVIEW_FPS,
    PREVIEW_WIDTH,
    FfmpegPreviewSource,
    preview_sources_from_ids,
    shared_capture_mode,
)
from src.motion_capture.rig.recorder import dshow_device_ref
from src.motion_capture.rig.sources import HOST_MONOTONIC, Frame, FrameSource
from src.shared.python.core.contracts import StateError


INSTANCE = "USB\\VID_32E4&PID_5234&MI_00\\6&FADBF3B&0&0000"


class FakeProc:
    """Stands in for ffmpeg: serves raw BGR ``frames`` then EOF."""

    def __init__(self, frames: list[bytes], stderr: bytes = b"") -> None:
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO(b"".join(frames))
        self.stderr = io.BytesIO(stderr)
        self.terminated = False

    def poll(self) -> int | None:
        return 0 if self.terminated else None

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def kill(self) -> None:
        self.terminated = True


def _source(
    frames: list[bytes], *, stderr: bytes = b"", **kwargs: object
) -> tuple[FfmpegPreviewSource, list[list[str]], list[FakeProc]]:
    launched: list[list[str]] = []
    procs: list[FakeProc] = []

    def popen(args: list[str], **_: object) -> FakeProc:
        launched.append(list(args))
        procs.append(FakeProc(frames, stderr))
        return procs[-1]

    src = FfmpegPreviewSource(
        "cam-a", INSTANCE, ffmpeg_exe="ffmpeg", popen=popen, **kwargs
    )
    return src, launched, procs


def _legacy_preview_args(mode: CaptureMode) -> list[str]:
    """The command ``ffmpeg_preview_args`` built before the port (verbatim)."""
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "dshow",
        "-vcodec",
        "mjpeg",
        "-video_size",
        f"{mode.width}x{mode.height}",
        "-framerate",
        str(mode.fps),
        "-i",
        f"video={dshow_device_ref(INSTANCE)}",
        "-vf",
        f"fps={PREVIEW_FPS},scale={PREVIEW_WIDTH}:-2",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "pipe:1",
    ]


def test_preview_launches_the_shared_raw_frame_command() -> None:
    mode = CaptureMode(width=1920, height=1200, fps=60)
    frame = bytes(PREVIEW_WIDTH * 400 * 3)
    src, launched, _ = _source([frame])
    src.open(mode)
    src.close()
    assert launched == [
        ffmpeg_raw_frame_args(
            "ffmpeg",
            shared_device_ref(INSTANCE),
            shared_capture_mode(mode),
            width=PREVIEW_WIDTH,
            fps=PREVIEW_FPS,
        )
    ]
    # The only delta from the rig's own pre-port command is the real-time
    # buffer the shared builder always sets (the recorder's live path already
    # used it); every other token is byte-identical and in the same order.
    legacy = _legacy_preview_args(mode)
    assert launched[0] == legacy[:6] + ["-rtbufsize", "256M"] + legacy[6:]


def test_shared_capture_mode_carries_every_field() -> None:
    mode = CaptureMode(width=1280, height=720, fps=120, fourcc="yuy2")
    assert shared_capture_mode(mode) == SharedCaptureMode(1280, 720, 120, "YUY2")


def test_recorder_device_ref_is_the_shared_one_with_the_rig_precondition() -> None:
    assert dshow_device_ref(INSTANCE) == shared_device_ref(INSTANCE)
    assert dshow_device_ref(INSTANCE) == (
        "@device_pnp_\\\\?\\usb#vid_32e4&pid_5234&mi_00#6&fadbf3b&0&0000"
        "#{65e8773d-8f56-11d0-a3b9-00a0c9223196}\\global"
    )
    with pytest.raises(ValueError, match="PnP instance id"):
        dshow_device_ref("not-an-instance-id")


def test_adapter_lifecycle_through_the_rig_protocol() -> None:
    mode = CaptureMode(width=16, height=8, fps=60)
    frames = [bytes([i]) * (8 * 4 * 3) for i in range(3)]
    src, _, procs = _source(frames, width=8, fps=12)
    assert isinstance(src, FrameSource)
    assert src.identity == "cam-a" and src.camera_instance_id == INSTANCE
    with pytest.raises(StateError, match="before open"):
        src.read()

    effective = src.open(mode)
    assert effective == CaptureMode(width=8, height=4, fps=12, fourcc="BGR3")

    # open() proved the first frame and consumed it; reads continue from there.
    frame = src.read()
    assert isinstance(frame, Frame)
    assert frame.seq == 1 and frame.clock_domain == HOST_MONOTONIC
    assert frame.image.shape == (4, 8, 3) and frame.image.dtype == np.uint8
    assert int(frame.image[0, 0, 0]) == 1
    assert src.read() is not None
    assert src.read() is None  # EOF: the rig contract is None, never an exception
    assert src.read() is None

    src.close()
    assert procs[0].terminated
    src.close()  # idempotent
    with pytest.raises(StateError, match="before open"):
        src.read()


def test_open_reports_ffmpeg_stderr_when_no_frame_arrives() -> None:
    src, _, procs = _source([], stderr=b"[dshow] Could not find video device")
    with pytest.raises(StateError, match="cam-a.*Could not find video device"):
        src.open(CaptureMode(width=16, height=8, fps=60))
    assert procs[0].terminated
    with pytest.raises(StateError, match="before open"):
        src.read()


def test_reopen_replaces_the_process() -> None:
    frame = bytes(8 * 4 * 3)
    src, launched, procs = _source([frame, frame], width=8)
    src.open(CaptureMode(width=16, height=8, fps=60))
    src.open(CaptureMode(width=16, height=8, fps=30))
    assert len(launched) == 2 and procs[0].terminated and not procs[1].terminated
    src.close()


def test_preview_source_preconditions() -> None:
    with pytest.raises(ValueError, match="identity"):
        FfmpegPreviewSource("", INSTANCE, ffmpeg_exe="ffmpeg")
    with pytest.raises(ValueError, match="width and fps"):
        FfmpegPreviewSource("cam", INSTANCE, ffmpeg_exe="ffmpeg", width=0)
    with pytest.raises(ValueError, match="PnP instance id"):
        FfmpegPreviewSource("cam", "no-backslash", ffmpeg_exe="ffmpeg")


def test_preview_sources_from_ids_requires_every_view() -> None:
    plan = RigPlan(
        name="p",
        cameras=(
            CameraBinding(view="a", serial="1"),
            CameraBinding(view="b", serial="2"),
        ),
    )
    sources = preview_sources_from_ids(
        plan, {"a": INSTANCE, "b": INSTANCE}, ffmpeg_exe="ffmpeg"
    )
    assert set(sources) == {"a", "b"}
    assert all(isinstance(s, FfmpegPreviewSource) for s in sources.values())
    with pytest.raises(ValueError, match="every plan view"):
        preview_sources_from_ids(plan, {"a": INSTANCE}, ffmpeg_exe="ffmpeg")
