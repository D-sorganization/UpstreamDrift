"""Native headless QTest flows through the real dashboard and MuJoCo runner.

The tiny gravity-driven hinge requires no renderer, model repository or network.
Only optional dependency imports skip; application failures must fail the test.
"""

from __future__ import annotations

import os
from time import monotonic

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytestmark = pytest.mark.unit


@pytest.fixture
def dashboard(tmp_path):
    try:
        from PyQt6 import QtCore, QtTest, QtWidgets
        import matplotlib  # noqa: F401
        import mujoco  # noqa: F401
    except (ImportError, OSError) as exc:
        pytest.skip(f"PyQt6/matplotlib/MuJoCo not loadable: {exc}")
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (
        MuJoCoPhysicsEngine,
    )
    from src.shared.python.dashboard.window import (
        ModelLoadStatus,
        UnifiedDashboardWindow,
    )

    model = tmp_path / "hinge.xml"
    model.write_text(
        '<mujoco><option timestep="0.002"/><worldbody><body pos="0 0 1">'
        '<joint name="hinge" type="hinge" axis="0 1 0"/>'
        '<geom type="capsule" fromto="0 0 0 0.3 0 -0.5" size="0.04"/>'
        "</body></worldbody></mujoco>",
        encoding="utf-8",
    )
    engine = MuJoCoPhysicsEngine()
    engine.load_from_path(str(model))
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = UnifiedDashboardWindow(
        engine, model_status=ModelLoadStatus("MuJoCo", "Synthetic Hinge")
    )
    window.resize(1280, 800)
    window.show()
    app.processEvents()
    assert window.isVisible()
    yield window, QtCore, QtTest.QTest
    window.runner.stop()
    window.close()
    window.deleteLater()
    app.processEvents()


def _wait(qtest, predicate, message: str) -> None:
    """Pump Qt events until a bounded, observable postcondition holds."""
    deadline = monotonic() + 3.0
    while not predicate() and monotonic() < deadline:
        qtest.qWait(10)
    assert predicate(), message


def _start(dashboard) -> None:
    window, core, qtest = dashboard
    qtest.mouseClick(window.control_panel.btn_start, core.Qt.MouseButton.LeftButton)
    _wait(qtest, lambda: window.recorder.current_idx >= 3, "No physics samples")


def test_launch_pause_resume_stop_and_plot(dashboard) -> None:
    window, core, qtest = dashboard
    assert window.identity_label.text() == "Engine: MuJoCo | Model: Synthetic Hinge"
    assert window.recorder.current_idx == 0
    _start(dashboard)
    pause = window.control_panel.btn_pause
    qtest.mouseClick(pause, core.Qt.MouseButton.LeftButton)
    _wait(qtest, lambda: window.runner.paused, "Pause was not applied")
    # A step already in flight may finish; then the paused model must stay still.
    qtest.qWait(80)
    paused_time = window.engine.get_time()
    paused_samples = window.recorder.current_idx
    qtest.qWait(80)
    assert window.engine.get_time() == paused_time
    assert window.recorder.current_idx == paused_samples
    qtest.mouseClick(pause, core.Qt.MouseButton.LeftButton)
    _wait(qtest, lambda: window.engine.get_time() > paused_time, "Did not resume")
    qtest.mouseClick(window.control_panel.btn_stop, core.Qt.MouseButton.LeftButton)
    assert not window.runner.isRunning()
    assert not window.recorder.is_recording
    times, positions = window.recorder.get_time_series("joint_positions")
    assert len(times) >= 3 and np.all(np.diff(times) > 0)
    assert positions.shape == (len(times), 1)
    assert np.isfinite(positions).all() and np.ptp(positions) > 0
    assert window.live_plot.ax.lines
    combo = window.live_plot.combo
    combo.setFocus()
    qtest.keyClick(combo, core.Qt.Key.Key_Down)
    assert window.live_plot.current_key == "joint_velocities"
    assert window.live_plot.ax.lines


def test_reset_while_paused_allows_fresh_run(dashboard) -> None:
    window, core, qtest = dashboard
    _start(dashboard)
    qtest.mouseClick(window.control_panel.btn_pause, core.Qt.MouseButton.LeftButton)
    assert window.runner.paused
    qtest.mouseClick(window.control_panel.btn_reset, core.Qt.MouseButton.LeftButton)
    assert not window.runner.isRunning()
    assert not window.runner.paused
    assert not window.control_panel.btn_pause.isChecked()
    assert window.engine.get_time() == 0
    assert window.recorder.current_idx == 0
    assert not window.live_plot.ax.lines
    _start(dashboard)
    assert window.engine.get_time() > 0


def test_close_stops_active_simulation(dashboard) -> None:
    window, _core, qtest = dashboard
    _start(dashboard)
    window.close()
    assert not window.isVisible()
    assert not window.runner.isRunning()
    assert not window.recorder.is_recording
    samples = window.recorder.current_idx
    qtest.qWait(60)
    assert window.recorder.current_idx == samples
