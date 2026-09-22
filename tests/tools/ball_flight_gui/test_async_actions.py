"""#8880 proof-of-migration: Simulate Flight no longer blocks the GUI thread.

``ball_flight_gui`` was one of the ~23 tools still running its simulation
inline in a ``clicked`` handler. These tests assert the migration's three
deliverables: the integration runs off the GUI thread, a second click cannot
queue a second run while one is in flight, and the synchronous
``_run_simulation`` core still behaves exactly as before (it is what
``tests/tools/ball_flight_gui/test_ball_flight_gui.py`` drives).
"""

from __future__ import annotations

import os
import sys
import time
import types
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QCoreApplication, QThread  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.tools.ball_flight_gui.gui import BallFlightWidget  # noqa: E402

pytestmark = pytest.mark.unit

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


def _make_traj_point(x: float, y: float, z: float, t: float):  # noqa: ANN201
    from unittest.mock import MagicMock

    pt = MagicMock()
    pt.position = np.array([x, y, z])
    pt.time = t
    return pt


def _patch_physics(trajectory):  # noqa: ANN001
    """Inject fake physics modules that record which thread ran them."""
    sim_mod = types.ModuleType("src.shared.python.physics.ball_simulator")
    cond_mod = types.ModuleType("src.shared.python.physics.ball_launch_conditions")

    class _FakeSim:
        def __init__(self, *args, **kwargs):
            _FakeSim.last_env = kwargs.get("env")

        def simulate_trajectory(self, launch, max_time, dt):
            _FakeSim.last_call = (launch, max_time, dt)
            _FakeSim.thread = QThread.currentThread()
            return trajectory

    from src.shared.python.physics.ball_launch_conditions import (
        EnvironmentalConditions,
        LaunchConditions,
    )

    sim_mod.BallFlightSimulator = _FakeSim
    cond_mod.LaunchConditions = LaunchConditions
    cond_mod.EnvironmentalConditions = EnvironmentalConditions
    return (
        patch.dict(
            sys.modules,
            {
                "src.shared.python.physics.ball_simulator": sim_mod,
                "src.shared.python.physics.ball_launch_conditions": cond_mod,
            },
        ),
        _FakeSim,
    )


@pytest.fixture
def widget():  # noqa: ANN201
    _ensure_qapp()
    the_widget = BallFlightWidget()
    yield the_widget
    the_widget.cleanup()
    the_widget.deleteLater()


def test_simulate_flight_runs_off_the_gui_thread(widget: BallFlightWidget) -> None:
    gui_thread = QThread.currentThread()
    traj = [_make_traj_point(0.0, 0.0, 0.0, 0.0), _make_traj_point(50.0, 0.0, 5.0, 1.0)]
    ctx, fake_sim = _patch_physics(traj)
    with ctx:
        widget._run_simulation_async()
        assert _pump_until(
            lambda: "Ball Flight Results" in widget._results_text.toPlainText()
        )
    assert fake_sim.thread is not None
    assert fake_sim.thread is not gui_thread, (
        "the simulation still ran on the GUI thread"
    )


def test_run_button_is_disabled_while_the_simulation_runs(
    widget: BallFlightWidget,
) -> None:
    release: list[bool] = []
    traj = [_make_traj_point(0.0, 0.0, 0.0, 0.0)]

    sim_mod = types.ModuleType("src.shared.python.physics.ball_simulator")
    cond_mod = types.ModuleType("src.shared.python.physics.ball_launch_conditions")

    class _Blocking:
        def __init__(self, *args, **kwargs):
            pass

        def simulate_trajectory(self, launch, max_time, dt):
            while not release:
                QThread.msleep(5)
            return traj

    from src.shared.python.physics.ball_launch_conditions import (
        EnvironmentalConditions,
        LaunchConditions,
    )

    sim_mod.BallFlightSimulator = _Blocking
    cond_mod.LaunchConditions = LaunchConditions
    cond_mod.EnvironmentalConditions = EnvironmentalConditions

    with patch.dict(
        sys.modules,
        {
            "src.shared.python.physics.ball_simulator": sim_mod,
            "src.shared.python.physics.ball_launch_conditions": cond_mod,
        },
    ):
        widget._run_btn.click()
        assert _pump_until(lambda: not widget._run_btn.isEnabled())
        release.append(True)
        assert _pump_until(lambda: widget._run_btn.isEnabled())


def test_synchronous_run_simulation_still_works(widget: BallFlightWidget) -> None:
    """The sync core (used directly by the existing GUI test suite) is unchanged."""
    traj = [
        _make_traj_point(0.0, 0.0, 0.0, 0.0),
        _make_traj_point(150.0, 0.0, 20.0, 2.0),
    ]
    ctx, _fake_sim = _patch_physics(traj)
    with ctx:
        widget._run_simulation()
    assert "Ball Flight Results" in widget._results_text.toPlainText()
