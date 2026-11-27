"""Native offscreen QTest of Advanced Golf Analysis with real MuJoCo dynamics.

Only GPU rendering is replaced; model loading, workers, controls and recording
remain production code. This does not qualify rendered pixels or golf physics.
"""

from __future__ import annotations

import os
from time import monotonic

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytestmark = pytest.mark.unit


@pytest.fixture
def analysis_window(monkeypatch, tmp_path):
    try:
        from PyQt6 import QtCore, QtTest, QtWidgets
        import matplotlib  # noqa: F401
        import mujoco
    except (ImportError, OSError) as exc:
        pytest.skip(f"PyQt6/matplotlib/MuJoCo not loadable: {exc}")
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf import (
        sim_widget,
    )
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.gui.core import (
        AdvancedGolfAnalysisWindow,
    )

    class HeadlessRenderer:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

    monkeypatch.setattr(mujoco, "Renderer", HeadlessRenderer)
    monkeypatch.setattr(sim_widget.MuJoCoSimWidget, "_render_once", lambda self: None)
    monkeypatch.chdir(tmp_path)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = AdvancedGolfAnalysisWindow()
    window.resize(1280, 800)
    window.show()
    app.processEvents()
    yield window, QtCore.Qt, QtTest.QTest
    for child in window.findChildren(sim_widget.MuJoCoSimWidget):
        child.release_resources()
    window.grip_modelling_tab.close()
    window.status_timer.stop()
    window.timer.stop()
    window.close()
    window.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    app.processEvents()


def _wait(qtest, predicate, message: str) -> None:
    deadline = monotonic() + 8
    while not predicate() and monotonic() < deadline:
        qtest.qWait(10)
    assert predicate(), message


def _load_golf(window, qt, qtest):
    combo = window.physics_tab.model_combo
    _wait(qtest, combo.isEnabled, "initial model load did not finish")
    # Select the shipped upper-body golf model using native keyboard events.
    combo.setFocus()
    qtest.keyClick(combo, qt.Key.Key_Home)
    _wait(qtest, combo.isEnabled, "first model load did not finish")
    for _ in range(3):
        qtest.keyClick(combo, qt.Key.Key_Down)
        _wait(qtest, combo.isEnabled, "golf model load did not finish")
    assert (
        window.physics_tab.model_configs[combo.currentIndex()]["name"] == "upper_body"
    )
    assert window.sim_widget.model.nu > 0
    return window.sim_widget


def test_golf_launch_pause_step_reset_and_record(analysis_window):
    window, qt, qtest = analysis_window
    sim = _load_golf(window, qt, qtest)
    window.tab_widget.setCurrentWidget(window.controls_tab)
    controls = window.controls_tab
    qtest.mouseClick(controls.play_pause_btn, qt.MouseButton.LeftButton)
    assert not sim.running
    paused = sim.get_time()
    qtest.qWait(40)
    assert sim.get_time() == paused
    qtest.mouseClick(controls.record_btn, qt.MouseButton.LeftButton)
    qtest.mouseClick(controls.play_pause_btn, qt.MouseButton.LeftButton)
    _wait(qtest, lambda: sim.get_time() > paused, "MuJoCo did not advance")
    _wait(qtest, lambda: sim.recorder.get_num_frames() > 0, "no recording frames")
    qtest.mouseClick(controls.play_pause_btn, qt.MouseButton.LeftButton)
    qtest.mouseClick(controls.record_btn, qt.MouseButton.LeftButton)
    assert np.isfinite(sim.data.qpos).all()
    qtest.mouseClick(controls.reset_btn, qt.MouseButton.LeftButton)
    assert sim.get_time() == 0


def test_close_stops_golf_simulation_and_status_timers(analysis_window):
    window, qt, qtest = analysis_window
    sim = _load_golf(window, qt, qtest)
    _wait(qtest, lambda: sim.get_time() > 0, "MuJoCo did not start")
    window.close()
    assert not window.isVisible()
    assert not sim.running
    assert not sim.timer.isActive()
    assert not window.status_timer.isActive()
    hand = window.grip_modelling_tab.sim_widget
    assert not hand.timer.isActive()
    assert sim.renderer is None
    assert hand.renderer is None
    window.close()  # repeated shutdown must remain safe
    stopped = sim.get_time()
    qtest.qWait(50)
    assert sim.get_time() == stopped


def test_kinematic_pose_edit_reaches_real_model(analysis_window):
    window, qt, qtest = analysis_window
    sim = _load_golf(window, qt, qtest)
    mode = window.physics_tab.mode_combo
    mode.setFocus()
    qtest.keyClick(mode, qt.Key.Key_End)
    assert sim.operating_mode == "kinematic"
    window.tab_widget.setCurrentWidget(window.controls_tab)
    name, widgets = next(iter(window.controls_tab.joint_widgets.items()))
    spin = widgets["spin"]
    spin.setFocus()
    spin.selectAll()
    qtest.keyClicks(spin, "0.25")
    qtest.keyClick(spin, qt.Key.Key_Return)
    joint = sim.model.joint(name)
    assert sim.data.qpos[joint.qposadr[0]] == pytest.approx(0.25)
    qtest.qWait(40)
    assert sim.data.qpos[joint.qposadr[0]] == pytest.approx(0.25)
