"""A stalled camera must never hold the devices hostage.

A live preview reads its cameras through a blocking pipe. When ffmpeg stops
producing frames but stays alive (observed on the lab rig: three preview
processes alive at zero CPU for over an hour), the reader thread parks inside
that read and never looks at its stop flag again. The panel then cannot
release the cameras, so the *recorder* cannot open them and every take comes
back empty. These tests pin the two guarantees that prevent it: stopping
always closes the source, and a source that goes quiet is noticed and
released on its own.
"""

from __future__ import annotations

import os
import sys
import threading
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

from src.motion_capture.rig.plan import CaptureMode, RigPlan
from src.motion_capture.rig.sources import Frame
from src.tools.capture_rig.commands import PlanSelection
from src.tools.capture_rig.preview import PreviewPanel

pytestmark = [pytest.mark.unit, pytest.mark.ui]

_APP: QApplication | None = None


def _app() -> QApplication:
    global _APP
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    _APP = app  # type: ignore[assignment]
    return _APP  # type: ignore[return-value]


def _pump(seconds: float) -> None:
    app = _app()
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.005)


class StallingSource:
    """Delivers ``frames`` images, then blocks in ``read`` like a dead pipe.

    ``close`` releases the block, which is what terminating ffmpeg does to a
    reader parked on its stdout.
    """

    def __init__(self, identity: str, frames: int = 0) -> None:
        self._identity = identity
        self.frames = frames
        self.sent = 0
        self.closed = False
        self._released = threading.Event()

    @property
    def identity(self) -> str:
        return self._identity

    def open(self, mode: CaptureMode, controls: object = None) -> CaptureMode:
        return CaptureMode(width=32, height=24, fps=12, fourcc="BGR3")

    def read(self) -> Frame | None:
        if self.sent < self.frames:
            self.sent += 1
            image = np.full((24, 32, 3), self.sent % 256, dtype=np.uint8)
            return Frame(image=image, seq=self.sent, t_ns=time.monotonic_ns())
        self._released.wait(timeout=30)  # a stalled camera: no data, no EOF
        return None

    def close(self) -> None:
        self.closed = True
        self._released.set()


def _plan_file(tmp_path) -> object:
    path = tmp_path / "plan.json"
    path.write_text(
        '{"schema_version": "rig-plan/1.0.0", "name": "two", "cameras": ['
        '{"view": "cam_a", "serial": "1"}, {"view": "cam_b", "serial": "2"}]}',
        encoding="utf-8",
    )
    return path


def test_stop_closes_a_source_whose_reader_is_parked_in_a_blocking_read(
    tmp_path,
) -> None:
    """The camera is released even though the worker cannot see its stop flag."""
    _app()
    made: dict[str, StallingSource] = {}

    def factory(plan: RigPlan) -> dict[str, StallingSource]:
        made.update({c.view: StallingSource(c.identity) for c in plan.cameras})
        return made

    panel = PreviewPanel(source_factory=factory)
    panel.start(PlanSelection(plan=_plan_file(tmp_path)))
    _pump(1.2)  # the binder runs, then both workers park in read()
    assert made and all(not s.closed for s in made.values())

    started = time.monotonic()
    panel.stop()
    assert time.monotonic() - started < 10  # must not hang on the parked reads
    assert all(s.closed for s in made.values()), "a camera was left open"
    assert not panel.active
    assert "released" in panel.status.text()


class SlowToOpenSource(StallingSource):
    """Takes a while to hand over its first frame, like ffmpeg starting up."""

    def __init__(self, identity: str, open_delay_s: float) -> None:
        super().__init__(identity, frames=1000)
        self._open_delay_s = open_delay_s

    def open(self, mode: CaptureMode, controls: object = None) -> CaptureMode:
        time.sleep(self._open_delay_s)
        return super().open(mode, controls)


def test_a_camera_that_is_slow_to_open_is_not_called_stalled(tmp_path) -> None:
    """The watchdog must not fire before the first frame can possibly arrive.

    The stall clock starts when the worker is created, so a camera that takes
    seconds to open would look silent since the epoch and be killed on the
    first tick — which is exactly what stopped the live preview coming up.
    """
    _app()
    made: dict[str, SlowToOpenSource] = {}

    def factory(plan: RigPlan) -> dict[str, SlowToOpenSource]:
        made.update({c.view: SlowToOpenSource(c.identity, 1.5) for c in plan.cameras})
        return made

    panel = PreviewPanel(source_factory=factory)
    panel.set_stall_timeout(6.0)
    panel.start(PlanSelection(plan=_plan_file(tmp_path)))
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline and not all(
        panel.frames_seen(v) for v in panel.views()
    ):
        _pump(0.2)
    assert all(panel.frames_seen(v) for v in panel.views()), "killed before opening"
    assert panel.active and "stalled" not in panel.status.text().lower()
    panel.stop()
    assert all(s.closed for s in made.values())


def test_a_source_that_goes_quiet_is_reported_and_released(tmp_path) -> None:
    """A frozen view frees its camera instead of silently holding it."""
    _app()
    made: dict[str, StallingSource] = {}

    def factory(plan: RigPlan) -> dict[str, StallingSource]:
        made.update(
            {c.view: StallingSource(c.identity, frames=2) for c in plan.cameras}
        )
        return made

    panel = PreviewPanel(source_factory=factory)
    panel.set_stall_timeout(0.4)  # the rig's default is far longer
    panel.start(PlanSelection(plan=_plan_file(tmp_path)))
    _pump(1.2)
    assert panel.frames_seen("cam_a") > 0  # frames arrived, then stopped

    deadline = time.monotonic() + 8
    while time.monotonic() < deadline and not all(s.closed for s in made.values()):
        _pump(0.2)
    assert all(s.closed for s in made.values()), "a stalled camera was never released"
    assert "stalled" in panel.status.text().lower()
    assert not panel.active  # every view gone means the cameras are free
    panel.stop()
