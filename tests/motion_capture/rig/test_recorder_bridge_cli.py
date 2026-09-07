"""Recorder argument building, the Tools schema probe, and the operator CLI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.motion_capture.rig import __main__ as cli
from src.motion_capture.rig.plan import CameraBinding, CaptureMode, RigPlan
from src.motion_capture.rig.recorder import (
    NullRecorder,
    RecordingResult,
    dshow_device_ref,
    ffmpeg_stream_copy_args,
    record_all,
)
from src.motion_capture.rig.tools_bridge import probe_tools_schema
from src.shared.python.core.contracts import StateError

pytestmark = pytest.mark.unit


def test_dshow_device_ref_is_the_inverse_of_listing_parse() -> None:
    ref = dshow_device_ref("USB\\VID_32E4&PID_5234&MI_00\\6&FADBF3B&0&0000")
    assert ref == (
        "@device_pnp_\\\\?\\usb#vid_32e4&pid_5234&mi_00#6&fadbf3b&0&0000"
        "#{65e8773d-8f56-11d0-a3b9-00a0c9223196}\\global"
    )
    with pytest.raises(Exception, match="PnP instance id"):
        dshow_device_ref("not-an-instance-id")


def test_ffmpeg_args_stream_copy_mjpeg(tmp_path: Path) -> None:
    args = ffmpeg_stream_copy_args(
        "ffmpeg", "video-ref", CaptureMode(), tmp_path / "a.mkv"
    )
    assert args[:6] == ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "dshow"]
    assert "-vcodec" in args and args[args.index("-vcodec") + 1] == "mjpeg"
    assert args[args.index("-video_size") + 1] == "1920x1200"
    assert args[args.index("-framerate") + 1] == "60"
    assert args[args.index("-i") + 1] == "video=video-ref"
    assert args[args.index("-c:v") + 1] == "copy"
    assert args[-1].endswith("a.mkv")


def test_null_recorder_lifecycle_and_record_all(tmp_path: Path) -> None:
    rec = NullRecorder()
    with pytest.raises(StateError):
        rec.stop()
    plan = RigPlan(
        name="p",
        cameras=(
            CameraBinding(view="a", serial="1"),
            CameraBinding(view="b", serial="2"),
        ),
    )
    with patch("src.motion_capture.rig.recorder.time.sleep") as sleep:
        results = record_all(
            plan, {"a": "ref-a", "b": "ref-b"}, 3.0, tmp_path / "rec", NullRecorder
        )
    sleep.assert_called_once_with(3.0)
    assert [r.identity for r in results] == ["1", "2"]
    assert all(isinstance(r, RecordingResult) and r.returncode == 0 for r in results)
    assert results[0].path.name == "a_1.mkv"
    with pytest.raises(Exception, match="missing views"):
        record_all(plan, {"a": "ref-a"}, 1.0, tmp_path, NullRecorder)


def test_tools_schema_probe_fails_closed_without_the_pinned_module() -> None:
    probe = probe_tools_schema("definitely.not.a.module")
    assert probe.status == "unavailable"
    assert probe.reason is not None and "not in the pinned vendor tree" in probe.reason
    assert probe.to_dict()["status"] == "unavailable"


def test_tools_schema_probe_reports_incompatible_partial_modules() -> None:
    # ``json`` is importable but has none of the mocap submodules.
    probe = probe_tools_schema("json")
    assert probe.status == "incompatible"
    assert probe.reason is not None and "lacks expected submodules" in probe.reason


def test_cli_capture_synthetic_writes_manifest_and_exits_zero(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    RigPlan(
        name="synthetic",
        cameras=(
            CameraBinding(view="a", serial="1"),
            CameraBinding(view="b", serial="2"),
        ),
    ).save(plan_path)
    out = tmp_path / "out"
    code = cli.main(
        [
            "capture",
            "--plan",
            str(plan_path),
            "--duration",
            "0.3",
            "--out",
            str(out),
            "--synthetic",
        ]
    )
    assert code == 0
    manifest = out / "session_manifest.json"
    assert manifest.is_file()
    text = manifest.read_text(encoding="utf-8")
    assert '"outcome": "supported"' in text
    assert '"tools_schema"' in text and '"status": "unavailable"' in text
