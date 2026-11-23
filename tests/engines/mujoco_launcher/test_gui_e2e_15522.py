"""Native launcher E2E; Docker/download boundaries use local, real assets.

The child runs the shipped golfer through MuJoCo and emits real state packets.
This covers the Qt/process/config boundary, not Docker, X11 or GPU rendering.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = pytest.mark.unit


@pytest.fixture
def gui(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    try:
        from PyQt6.QtCore import Qt
        from PyQt6.QtTest import QTest
        from PyQt6.QtWidgets import QApplication, QPushButton
        import matplotlib  # noqa: F401
        import mujoco  # noqa: F401
    except (ImportError, OSError) as exc:
        pytest.skip(f"Optional Qt/matplotlib/MuJoCo unavailable: {exc}")
    from src.engines.physics_engines.mujoco.python import humanoid_launcher as mod

    app = QApplication.instance() or QApplication([])
    # The constructor's optional model-download probe must never use the network.
    model = Path(mod.__file__).resolve().parents[1] / "models/generated/golfer.xml"
    monkeypatch.setattr(
        mod,
        "StandardModelManager",
        lambda: SimpleNamespace(get_standard_humanoid_path=lambda: model),
    )
    window = mod.HumanoidLauncher(config_path=tmp_path / "config.json")
    window.show()
    QTest.qWait(30)
    try:
        yield SimpleNamespace(
            window=window,
            app=app,
            qt=Qt,
            test=QTest,
            model=model,
            tmp=tmp_path,
            button_type=QPushButton,
        )
    finally:
        worker = window.simulation_thread
        if worker is not None and worker.isRunning():
            worker.stop()
            assert worker.wait(5000), "Test child failed to stop"
        window.close()
        window.deleteLater()
        app.processEvents()


def _click(gui: SimpleNamespace, widget: Any) -> None:
    gui.test.mouseClick(widget, gui.qt.MouseButton.LeftButton)
    gui.app.processEvents()


def _wait(gui: SimpleNamespace, predicate: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 10
    while not predicate() and time.monotonic() < deadline:
        gui.test.qWait(20)
    assert predicate(), gui.window.txt_log.toPlainText()


def _local_child(
    gui: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, *, hold: bool
) -> None:
    script = gui.tmp / "local_simulation.py"
    script.write_text(
        "import json, sys, time, mujoco\n"
        "m = mujoco.MjModel.from_xml_path(sys.argv[1])\n"
        "d = mujoco.MjData(m)\n"
        "for _ in range(25): mujoco.mj_step(m, d)\n"
        "print('DATA_JSON:' + json.dumps({'time': d.time, "
        "'qpos': d.qpos.tolist(), 'qvel': d.qvel.tolist()}), flush=True)\n"
        "print('LOCAL_MODEL_READY', flush=True)\n"
        "if sys.argv[2] == 'hold': time.sleep(30)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        gui.window,
        "get_simulation_command",
        lambda: (
            [
                sys.executable,
                "-u",
                str(script),
                str(gui.model),
                "hold" if hold else "finish",
            ],
            {"MUJOCO_GL": "disable"},
        ),
    )


@pytest.mark.parametrize("mode", ["lqr", "poly"])
def test_native_controls_persist_selected_physics_configuration(
    gui: SimpleNamespace,
    mode: str,
) -> None:
    w = gui.window
    assert w.isVisible() and w.tabs.count() == 4
    w.combo_control.setFocus()
    gui.test.keyClick(w.combo_control, gui.qt.Key.Key_End)
    if mode == "lqr":
        gui.test.keyClick(w.combo_control, gui.qt.Key.Key_Up)
    assert w.combo_control.currentText() == mode
    assert w.btn_poly_generator.isEnabled() == (mode == "poly")
    assert w.btn_signal_toolkit.isEnabled() == (mode == "poly")
    _click(gui, w.chk_live)
    gui.test.mouseClick(
        w.tabs.tabBar(),
        gui.qt.MouseButton.LeftButton,
        pos=w.tabs.tabBar().tabRect(1).center(),
    )
    w.spin_height.setFocus()
    gui.test.keyClick(
        w.spin_height, gui.qt.Key.Key_A, gui.qt.KeyboardModifier.ControlModifier
    )
    gui.test.keyClicks(w.spin_height, "1.85")
    gui.test.keyClick(w.spin_height, gui.qt.Key.Key_Tab)
    save = next(
        b
        for b in w.findChildren(gui.button_type)
        if "Save Appearance Settings" in b.text()
    )
    _click(gui, save)
    saved = json.loads(w.config_path.read_text(encoding="utf-8"))
    assert saved["control_mode"] == mode
    assert saved["height_m"] == pytest.approx(1.85)
    assert saved["live_view"] == w.chk_live.isChecked()


def test_run_receives_real_mujoco_state_and_restores_controls(
    gui: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _local_child(gui, monkeypatch, hold=False)
    w = gui.window
    _click(gui, w.btn_run)
    worker = w.simulation_thread
    assert not w.btn_run.isEnabled() and w.btn_stop.isEnabled()
    _wait(gui, lambda: w.btn_run.isEnabled())
    assert worker.wait(1000)
    assert worker.process.returncode == 0
    assert w.recorder.data["times"] == pytest.approx([0.05])
    assert len(w.recorder.data["joint_positions"][0]) == 50
    assert not w.btn_stop.isEnabled()
    assert w.btn_data.isEnabled() and w.btn_video.isEnabled()


@pytest.mark.parametrize("action", ["stop", "close"])
def test_stop_or_close_releases_owned_process_and_thread(
    gui: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    action: str,
) -> None:
    _local_child(gui, monkeypatch, hold=True)
    w = gui.window
    _click(gui, w.btn_run)
    _wait(gui, lambda: "LOCAL_MODEL_READY" in w.txt_log.toPlainText())
    worker = w.simulation_thread
    assert worker.isRunning() and worker.process.poll() is None
    if action == "close":
        w.close()
    else:
        _click(gui, w.btn_stop)
    assert worker.wait(1500), "Closing launcher left its simulation running"
    assert worker.process.poll() is not None
    assert w.isVisible() == (action == "stop")
    if action == "stop":
        _wait(gui, lambda: w.btn_run.isEnabled())
        assert not w.btn_stop.isEnabled()
