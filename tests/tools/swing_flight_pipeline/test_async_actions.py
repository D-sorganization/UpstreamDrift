"""#8880 proof-of-migration: Run Full Pipeline no longer blocks the GUI thread.

``swing_flight_pipeline`` was one of the ~23 tools still running its
simulation inline in a ``clicked`` handler. These tests assert the
migration's three deliverables: the pipeline run happens off the GUI
thread, a second click cannot queue a second run while one is in flight,
and the synchronous ``_run_pipeline`` core still behaves exactly as before
(it is what ``tests/tools/swing_flight_pipeline/test_gui.py`` drives).
"""

from __future__ import annotations

import os
import sys
import time
import types
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QCoreApplication, QThread  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.tools.swing_flight_pipeline.gui import SwingFlightWidget  # noqa: E402

_APP: QApplication | None = None


def _ensure_qapp() -> QApplication:
    global _APP
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    _APP = app
    return app


def _pump_until(predicate, timeout_s: float = 20.0) -> bool:  # noqa: ANN001
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        QCoreApplication.processEvents()
        if predicate():
            return True
        QThread.msleep(5)
    QCoreApplication.processEvents()
    return predicate()


@dataclass
class _MockImpactState:
    ball_velocity: np.ndarray
    ball_angular_velocity: np.ndarray


@dataclass
class _MockLaunch:
    velocity: float
    launch_angle: float
    spin_rate: float


@dataclass
class _MockTrajPoint:
    position: np.ndarray


@dataclass
class _MockResult:
    swing_state: Any
    impact_state: _MockImpactState
    launch_conditions: _MockLaunch
    carry_m: float
    max_height_m: float
    flight_time_s: float
    landing_angle_deg: float
    trajectory: list


class _MockSwingState:
    def __init__(self, **kwargs):  # noqa: ANN003
        for k, v in kwargs.items():
            setattr(self, k, v)


class _MockPipeline:
    thread: Any = None

    def run(self, swing):  # noqa: ANN001, ANN201
        _MockPipeline.thread = QThread.currentThread()
        return _MockResult(
            swing_state=swing,
            impact_state=_MockImpactState(
                ball_velocity=np.array([60.0, 0.0, 20.0]),
                ball_angular_velocity=np.array([0.0, 300.0, 0.0]),
            ),
            launch_conditions=_MockLaunch(
                velocity=63.0, launch_angle=12.5, spin_rate=300.0
            ),
            carry_m=220.0,
            max_height_m=28.0,
            flight_time_s=6.2,
            landing_angle_deg=42.0,
            trajectory=[
                _MockTrajPoint(position=np.array([0.0, 0.0, 0.0])),
                _MockTrajPoint(position=np.array([220.0, 0.0, 0.0])),
            ],
        )


def _install_mock_pipeline_module():  # noqa: ANN201
    mod = types.ModuleType("src.shared.python.physics.swing_ball_flight_pipeline")
    mod.SwingBallFlightPipeline = _MockPipeline
    mod.SwingState = _MockSwingState
    return patch.dict(
        sys.modules,
        {"src.shared.python.physics.swing_ball_flight_pipeline": mod},
    )


@pytest.fixture(scope="module", autouse=True)
def _qapp():  # noqa: ANN201
    return _ensure_qapp()


@pytest.fixture
def widget():  # noqa: ANN201
    w = SwingFlightWidget()
    yield w
    w.cleanup()
    w.deleteLater()


def test_pipeline_runs_off_the_gui_thread(widget: SwingFlightWidget) -> None:
    gui_thread = QThread.currentThread()
    with _install_mock_pipeline_module():
        widget._engine_combo.setCurrentText("manual")
        widget._run_pipeline_async()
        assert _pump_until(
            lambda: "Pipeline Complete" in widget._results_text.toPlainText()
        )
    assert _MockPipeline.thread is not None
    assert _MockPipeline.thread is not gui_thread, (
        "pipeline still ran on the GUI thread"
    )
    assert widget._result is not None
    assert widget._result.carry_m == pytest.approx(220.0)


def test_run_button_is_disabled_while_the_pipeline_runs(
    widget: SwingFlightWidget,
) -> None:
    release: list[bool] = []

    class _Blocking:
        def run(self, swing):  # noqa: ANN001, ANN201
            while not release:
                QThread.msleep(5)
            return _MockPipeline().run(swing)

    mod = types.ModuleType("src.shared.python.physics.swing_ball_flight_pipeline")
    mod.SwingBallFlightPipeline = _Blocking
    mod.SwingState = _MockSwingState

    with patch.dict(
        sys.modules, {"src.shared.python.physics.swing_ball_flight_pipeline": mod}
    ):
        widget._engine_combo.setCurrentText("manual")
        widget._run_btn.click()
        assert _pump_until(lambda: not widget._run_btn.isEnabled())
        release.append(True)
        assert _pump_until(lambda: widget._run_btn.isEnabled())


def test_synchronous_run_pipeline_still_works(widget: SwingFlightWidget) -> None:
    """The sync core (used directly by the existing GUI test suite) is unchanged."""
    with _install_mock_pipeline_module():
        widget._engine_combo.setCurrentText("manual")
        widget._run_pipeline()
    assert "Pipeline Complete" in widget._results_text.toPlainText()
    assert widget._result.carry_m == pytest.approx(220.0)
