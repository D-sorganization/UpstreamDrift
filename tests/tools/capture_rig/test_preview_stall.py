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
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

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


def _plan_file(tmp_path: Path) -> Path:
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


def _run_qt_lifetime_probe(
    script: str, plan_path: Path
) -> subprocess.CompletedProcess[str]:
    """Run a QThread lifetime probe in an isolated subprocess (qFatal aborts)."""
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    repo_root = Path(__file__).resolve().parents[3]
    return subprocess.run(  # noqa: S603 - controlled test python executable
        [sys.executable, "-c", textwrap.dedent(script), str(plan_path)],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def test_repeated_start_stop_cycles_do_not_destroy_running_qthread(
    tmp_path: Path,
) -> None:
    """Rapid start/stop cycles must not destroy BinderThread before OS unwind completes."""
    plan = _plan_file(tmp_path)
    assert isinstance(plan, Path)
    script = """
    import sys
    import time
    from pathlib import Path
    from PyQt6.QtWidgets import QApplication
    from src.motion_capture.rig.plan import CaptureMode
    from src.tools.capture_rig.commands import PlanSelection
    from src.tools.capture_rig.preview import PreviewPanel

    app = QApplication(sys.argv[:1])

    class QuickSource:
        identity = "1"
        def open(self, mode, controls=None):
            return CaptureMode(width=16, height=12, fps=10, fourcc="BGR3")
        def read(self):
            time.sleep(0.005)
            return None
        def close(self):
            pass

    panel = PreviewPanel(
        source_factory=lambda plan: {c.view: QuickSource() for c in plan.cameras}
    )
    plan_path = Path(sys.argv[1])
    for _ in range(250):
        panel.start(PlanSelection(plan=plan_path))
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not panel._workers:
            app.processEvents()
        assert panel._workers, "worker failed to bind during cycle"
        panel.stop()
    panel.close()
    """
    proc = _run_qt_lifetime_probe(script, plan)
    assert proc.returncode == 0, (
        f"subprocess exited {proc.returncode}:\nstdout={proc.stdout}\nstderr={proc.stderr}"
    )


def test_lingering_worker_past_join_timeout_is_retained_until_finished(
    tmp_path: Path,
) -> None:
    """Threads still active past WORKER_JOIN_MS must be retained until finished fires."""
    plan = _plan_file(tmp_path)
    assert isinstance(plan, Path)
    script = """
    import gc
    import sys
    import threading
    import time
    from pathlib import Path
    from PyQt6.QtWidgets import QApplication
    import src.tools.capture_rig.preview as preview_mod
    from src.motion_capture.rig.plan import CaptureMode
    from src.tools.capture_rig.commands import PlanSelection

    app = QApplication(sys.argv[:1])
    preview_mod.WORKER_JOIN_MS = 20

    entered = threading.Event()
    release = threading.Event()

    class UnresponsiveSource:
        identity = "1"
        def open(self, mode, controls=None):
            return CaptureMode(width=16, height=12, fps=10, fourcc="BGR3")
        def read(self):
            entered.set()
            release.wait(timeout=2.0)
            return None
        def close(self):
            pass  # intentionally ignores close until test releases it

    panel = preview_mod.PreviewPanel(
        source_factory=lambda plan: {
            c.view: UnresponsiveSource() for c in plan.cameras
        }
    )
    panel.start(PlanSelection(plan=Path(sys.argv[1])))

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not (entered.is_set() and panel._workers):
        app.processEvents()
        time.sleep(0.005)
    assert entered.is_set() and panel._workers
    lingering = list(panel._workers.values())

    try:
        panel.stop()
        assert all(w in preview_mod._LINGERING_THREADS for w in lingering), (
            "workers past join timeout were not retained"
        )
        gc.collect()
        app.processEvents()
        gc.collect()
    finally:
        release.set()

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and any(not w.isFinished() for w in lingering):
        app.processEvents()
        time.sleep(0.005)
    app.processEvents()
    assert all(w.isFinished() for w in lingering)
    assert not any(w in preview_mod._LINGERING_THREADS for w in lingering)
    panel.close()
    gc.collect()
    """
    proc = _run_qt_lifetime_probe(script, plan)
    assert proc.returncode == 0, (
        f"subprocess exited {proc.returncode}:\nstdout={proc.stdout}\nstderr={proc.stderr}"
    )
