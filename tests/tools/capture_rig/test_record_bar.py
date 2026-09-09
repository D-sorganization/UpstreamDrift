"""Recording clock, transport controls, live snapshots and the recorder's tee."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
cv2 = pytest.importorskip("cv2")

import numpy as np
from PyQt6.QtWidgets import QApplication
from src.motion_capture.rig.plan import CaptureMode
from src.motion_capture.rig.recorder import (
    LiveOptions,
    NullRecorder,
    ffmpeg_stream_copy_args,
    record_all,
)
from src.tools.capture_rig import commands
from src.tools.capture_rig.commands import PlanSelection
from src.tools.capture_rig.preview import PreviewPanel
from src.tools.capture_rig.record_bar import (
    Phase,
    RecordBar,
    RecordingClock,
    clock_text,
)

pytestmark = [pytest.mark.unit, pytest.mark.ui]

_APP: QApplication | None = None


def _app() -> QApplication:
    global _APP
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    _APP = app  # type: ignore[assignment]
    return _APP  # type: ignore[return-value]


class FakeNow:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


# -- clock (no Qt) ----------------------------------------------------------------------
def test_clock_counts_down_then_records_with_bounded_readings() -> None:
    clock = RecordingClock()
    assert clock.phase is Phase.IDLE and clock.badge() == ""
    clock.arm(10.0, 3.0, now=0.0)
    assert clock.phase is Phase.COUNTDOWN and clock.badge() == "Starting in 3"
    clock.tick(2.2)
    assert not clock.countdown_done() and clock.badge() == "Starting in 1"
    clock.tick(3.0)
    assert clock.countdown_done() and clock.remaining_countdown == 0.0
    clock.recording_started(now=5.0)
    assert clock.phase is Phase.RECORDING and clock.elapsed == 0.0
    clock.tick(9.0)
    assert clock.elapsed == 4.0 and clock.remaining == 6.0
    assert clock.progress == pytest.approx(0.4)
    assert clock.badge() == "● REC 00:04 / 00:10"
    clock.tick(30.0)  # the recorder decides when it ends; readings stay bounded
    assert clock.remaining == 0.0 and clock.progress == 1.0
    clock.reset()
    assert clock.phase is Phase.IDLE and clock.badge() == ""
    with pytest.raises(Exception, match="positive"):
        clock.arm(0.0, 0.0, now=0.0)
    clock.arm(1.0, 0.0, now=0.0)
    with pytest.raises(Exception, match="backwards"):
        clock.tick(-1.0)
    with pytest.raises(Exception, match="idle"):
        clock.arm(1.0, 0.0, now=0.0)
    assert clock_text(0) == "00:00" and clock_text(125.9) == "02:05"


# -- bar --------------------------------------------------------------------------------
def test_record_bar_runs_the_countdown_then_asks_to_start_and_shows_rec() -> None:
    _app()
    now = FakeNow()
    bar = RecordBar(now=now)
    started, stopped, badges = [], [], []
    bar.start_requested.connect(lambda: started.append(True))
    bar.stop_requested.connect(lambda: stopped.append(True))
    bar.badge_changed.connect(badges.append)
    bar.preset_buttons[15.0].click()
    assert bar.duration_s() == 15.0
    bar.countdown_combo.setCurrentIndex(1)  # 3 s
    assert bar.indicator.text() == "ready"
    bar.record_button.click()
    assert bar.phase is Phase.COUNTDOWN and bar.record_button.text() == "Cancel"
    assert not bar.duration_spin.isEnabled() and badges[-1] == "Starting in 3"
    now.t += 1.0
    bar.tick()
    assert badges[-1] == "Starting in 2" and not started
    now.t += 2.0
    bar.tick()
    assert started == [True] and bar.phase is Phase.COUNTDOWN
    bar.recording_started()
    assert bar.phase is Phase.RECORDING and bar.record_button.text() == "■  Stop"
    now.t += 4.0
    bar.tick()
    assert "REC 00:04 / 00:15" in bar.indicator.text()
    assert badges[-1] == "● REC 00:04 / 00:15"
    assert bar.progress.value() == pytest.approx(4 / 15 * 1000, abs=1)
    bar.record_button.click()  # early stop is a request to the tile
    assert stopped == [True] and bar.phase is Phase.RECORDING
    bar.recording_finished()
    assert bar.phase is Phase.IDLE and bar.indicator.text() == "ready"
    assert badges[-1] == "" and bar.duration_spin.isEnabled()
    # Cancelling during the countdown never asks to start.
    bar.record_button.click()
    bar.record_button.click()
    assert bar.phase is Phase.IDLE and started == [True]


# -- recorder tee + stop file -----------------------------------------------------------
def test_ffmpeg_args_tee_a_live_preview_jpeg_next_to_the_stream_copy(
    tmp_path: Path,
) -> None:
    mode = CaptureMode(width=1920, height=1200, fps=60, fourcc="MJPG")
    plain = ffmpeg_stream_copy_args("ffmpeg", "ref", mode, tmp_path / "a.mkv")
    assert plain[-4:] == ["-c:v", "copy", "-y", str(tmp_path / "a.mkv")]
    live = ffmpeg_stream_copy_args(
        "ffmpeg", "ref", mode, tmp_path / "a.mkv", live_preview=tmp_path / "a.jpg"
    )
    joined = " ".join(live)
    head = joined[: joined.index(" -i ")]
    assert "-rtbufsize 256M" in head and "-lowres:v 2" in head  # input side
    tail = joined[joined.index(" -i ") :]
    assert "-map 0:v -c:v copy -y" in tail and str(tmp_path / "a.mkv") in tail
    assert "-vf fps=8,scale=480:-2" in tail and "-update 1" in tail
    assert tail.endswith(str(tmp_path / "a.jpg"))


def test_record_all_stops_early_when_the_stop_file_appears(tmp_path: Path) -> None:
    from src.motion_capture.rig.plan import RigPlan

    plan = RigPlan.model_validate(
        {
            "schema_version": "rig-plan/1.0.0",
            "name": "one",
            "cameras": [{"view": "a", "serial": "1"}],
        }
    )
    stop = tmp_path / ".stop"
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        if sum(slept) >= 1.0:
            stop.write_text("stop", encoding="utf-8")

    made: list[NullRecorder] = []

    def factory() -> NullRecorder:
        rec = NullRecorder()
        made.append(rec)
        return rec

    live = LiveOptions(live_preview_dir=tmp_path / ".live", stop_file=stop)
    results = record_all(
        plan,
        {"a": "ref"},
        30.0,
        tmp_path,
        factory,
        warmup_s=0.0,
        sleep=sleep,
        live=live,
    )
    assert len(results) == 1 and made[0].signalled
    assert sum(slept) < 5.0  # far short of the 30 s duration
    assert made[0].live_preview == tmp_path / ".live" / "a.jpg"
    assert not stop.exists()  # consumed so the next take is not stopped at once


def test_record_command_carries_live_preview_and_stop_file(tmp_path: Path) -> None:
    argv = commands.record_command(
        PlanSelection(plan=tmp_path / "p.json"),
        tmp_path / "take",
        duration_s=12,
        live_preview=tmp_path / "take" / ".live",
        stop_file=tmp_path / "take" / ".stop",
    )
    joined = " ".join(argv)
    assert "--live-preview" in joined and "--stop-file" in joined
    assert "--duration 12" in joined
    bound = commands.record_command(
        PlanSelection(plan=tmp_path / "p.json"),
        tmp_path / "take",
        cameras={"cam_a": "USB\\VID_1&PID_2\\6&ABC&0&0000"},
    )
    assert bound[-2:] == ["--camera", "cam_a=USB\\VID_1&PID_2\\6&ABC&0&0000"]


def test_cli_parses_camera_bindings_and_rejects_malformed_ones() -> None:
    from src.motion_capture.rig.__main__ import parse_camera_binding

    assert parse_camera_binding("cam_a=USB\\X&0") == ("cam_a", "USB\\X&0")
    with pytest.raises(SystemExit, match="VIEW=INSTANCE_ID"):
        parse_camera_binding("cam_a")


def test_preview_remembers_bound_camera_ids_for_the_recorder(tmp_path: Path) -> None:
    from src.motion_capture.rig.sources import SyntheticFrameSource

    class Real(SyntheticFrameSource):
        camera_instance_id = "USB\\VID&PID\\1&0&0000"

    plan = tmp_path / "plan.json"
    plan.write_text(
        '{"schema_version": "rig-plan/1.0.0", "name": "one", "cameras": ['
        '{"view": "cam_a", "serial": "1"}, {"view": "cam_b", "serial": "2"}]}',
        encoding="utf-8",
    )
    app = _app()
    panel = PreviewPanel(
        source_factory=lambda p: {
            "cam_a": Real("1"),
            "cam_b": SyntheticFrameSource("2"),  # no instance id
        }
    )
    assert panel.camera_ids() == {}
    panel.start(PlanSelection(plan=plan))
    for _ in range(50):
        app.processEvents()
        if panel.camera_ids():
            break
        __import__("time").sleep(0.02)
    panel.stop()
    assert panel.camera_ids() == {"cam_a": "USB\\VID&PID\\1&0&0000"}


# -- preview snapshots + badge ----------------------------------------------------------
def test_preview_shows_recorder_snapshots_and_stamps_the_badge(tmp_path: Path) -> None:
    app = _app()
    live = tmp_path / ".live"
    live.mkdir()
    frame = np.full((60, 96, 3), 90, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    (live / "cam_a.jpg").write_bytes(buf.tobytes())
    panel = PreviewPanel(source_factory=lambda plan: {})
    panel.set_badge("● REC 00:01 / 00:10")
    assert panel.badge == "● REC 00:01 / 00:10"
    panel.watch_snapshots(live, ("cam_a", "cam_b"))
    assert panel.views() == ("cam_a", "cam_b") and not panel.active
    for _ in range(30):
        app.processEvents()
        panel.poll_snapshots()
    assert panel.frames_seen("cam_a") >= 1 and panel.frames_seen("cam_b") == 0
    assert "recording" in panel.status.text()
    panel.stop_watching()
    assert "off" in panel.status.text()
    with pytest.raises(Exception, match="directory"):
        panel.watch_snapshots(tmp_path / "missing", ("cam_a",))
