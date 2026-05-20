"""Tests for the aerodynamic ball flight simulator GUI.

The scope `src/tools/ball_flight_gui/gui.py` is mostly PyQt6 widget
construction with one logic method (`_apply_preset`) and one orchestration
method (`_run_simulation`) that wires the UI to the shared physics engine.
We construct widgets under an offscreen QApplication and mock the physics
import so the suite runs fast and deterministic.
"""

from __future__ import annotations

import os
import sys
import types
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication, QMainWindow

from src.tools.ball_flight_gui.gui import (
    BallFlightWidget,
    BallFlightWindow,
    get_dockable_ui,
)

_APP: QApplication | None = None


def _ensure_qapp() -> QApplication:
    global _APP
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    _APP = app
    return app


@pytest.fixture()
def widget() -> BallFlightWidget:
    _ensure_qapp()
    return BallFlightWidget()


def _make_traj_point(x: float, y: float, z: float, t: float):
    pt = MagicMock()
    pt.position = np.array([x, y, z])
    pt.time = t
    return pt


def _patch_physics(trajectory):
    """Return a context manager that injects fake physics modules.

    The real physics modules are heavy and not strictly needed to exercise
    the GUI orchestration path. Each call returns ``trajectory`` from
    ``BallFlightSimulator().simulate_trajectory``.
    """
    sim_mod = types.ModuleType("src.shared.python.physics.ball_simulator")
    cond_mod = types.ModuleType("src.shared.python.physics.ball_launch_conditions")

    class _FakeSim:
        def simulate_trajectory(self, launch, max_time, dt):
            _FakeSim.last_call = (launch, max_time, dt)
            return trajectory

    class _FakeLaunchConditions:
        def __init__(self, velocity, launch_angle, spin_rate):
            self.velocity = velocity
            self.launch_angle = launch_angle
            self.spin_rate = spin_rate

    sim_mod.BallFlightSimulator = _FakeSim
    cond_mod.LaunchConditions = _FakeLaunchConditions
    return (
        patch.dict(
            sys.modules,
            {
                "src.shared.python.physics.ball_simulator": sim_mod,
                "src.shared.python.physics.ball_launch_conditions": cond_mod,
            },
        ),
        _FakeSim,
        _FakeLaunchConditions,
    )


# --- Construction --------------------------------------------------------


def test_widget_constructs_with_expected_defaults(widget: BallFlightWidget) -> None:
    assert widget._speed_spin.value() == pytest.approx(163.0)
    assert widget._angle_spin.value() == pytest.approx(11.0)
    assert widget._spin_spin.value() == pytest.approx(2500.0)
    assert widget._sidespin_spin.value() == pytest.approx(0.0)
    assert widget._wind_speed.value() == pytest.approx(0.0)
    assert widget._wind_dir.value() == pytest.approx(0.0)
    assert widget._altitude.value() == pytest.approx(0.0)
    assert widget._chk_dimples.isChecked() is True
    assert widget._chk_magnus.isChecked() is True
    assert widget._chk_seam.isChecked() is False


def test_widget_spinbox_ranges(widget: BallFlightWidget) -> None:
    # Speed range
    assert widget._speed_spin.minimum() == pytest.approx(50.0)
    assert widget._speed_spin.maximum() == pytest.approx(200.0)
    # Launch angle range
    assert widget._angle_spin.minimum() == pytest.approx(-5.0)
    assert widget._angle_spin.maximum() == pytest.approx(45.0)
    # Backspin range
    assert widget._spin_spin.minimum() == pytest.approx(0.0)
    assert widget._spin_spin.maximum() == pytest.approx(12000.0)
    # Sidespin range
    assert widget._sidespin_spin.minimum() == pytest.approx(-5000.0)
    assert widget._sidespin_spin.maximum() == pytest.approx(5000.0)
    # Wind / altitude
    assert widget._wind_speed.maximum() == pytest.approx(50.0)
    assert widget._wind_dir.maximum() == pytest.approx(360.0)
    assert widget._altitude.maximum() == pytest.approx(10000.0)


def test_widget_results_text_initial_message(widget: BallFlightWidget) -> None:
    text = widget._results_text.toPlainText()
    assert "Configure launch conditions" in text
    assert "Magnus" in text
    assert widget._results_text.isReadOnly()


def test_widget_run_button_exists(widget: BallFlightWidget) -> None:
    assert widget._run_btn.text() == "Simulate Flight"


# --- Presets -------------------------------------------------------------


@pytest.mark.parametrize(
    ("speed", "angle", "spin"),
    [
        (163.0, 11.0, 2500.0),  # Driver
        (118.0, 16.0, 7000.0),  # 7-Iron
        (94.0, 23.0, 9000.0),  # PW
    ],
)
def test_apply_preset_updates_spin_boxes(
    widget: BallFlightWidget, speed: float, angle: float, spin: float
) -> None:
    widget._apply_preset(speed, angle, spin)
    assert widget._speed_spin.value() == pytest.approx(speed)
    assert widget._angle_spin.value() == pytest.approx(angle)
    assert widget._spin_spin.value() == pytest.approx(spin)


def test_apply_preset_clamps_to_spinbox_range(widget: BallFlightWidget) -> None:
    # QDoubleSpinBox.setValue() clamps to the spin's min/max. Verify the
    # behaviour stays predictable when callers pass out-of-range values.
    widget._apply_preset(speed=9999.0, angle=999.0, spin=99999.0)
    assert widget._speed_spin.value() == widget._speed_spin.maximum()
    assert widget._angle_spin.value() == widget._angle_spin.maximum()
    assert widget._spin_spin.value() == widget._spin_spin.maximum()


# --- Simulation orchestration --------------------------------------------


def test_run_simulation_renders_results_for_valid_trajectory(
    widget: BallFlightWidget,
) -> None:
    traj = [
        _make_traj_point(0.0, 0.0, 0.0, 0.0),
        _make_traj_point(50.0, 0.0, 20.0, 1.0),
        _make_traj_point(150.0, 0.0, 30.0, 2.0),
        _make_traj_point(220.0, 5.0, 1.0, 4.5),
    ]
    ctx, fake_sim, fake_cond = _patch_physics(traj)
    with ctx:
        widget._speed_spin.setValue(150.0)
        widget._angle_spin.setValue(12.0)
        widget._spin_spin.setValue(3000.0)
        widget._run_simulation()

    text = widget._results_text.toPlainText()
    assert "Ball Flight Results" in text
    assert "Carry:" in text
    assert "Max Height:" in text
    # Carry = sqrt(220^2 + 5^2) ~= 220.06m
    assert "220" in text
    # Flight time taken from last point
    assert "4.50" in text
    # Points count
    assert "Points:      4" in text
    # Verify LaunchConditions was built with correct mph->m/s conversion
    launch = fake_sim.last_call[0]
    assert launch.velocity == pytest.approx(150.0 * 0.44704)
    assert launch.launch_angle == pytest.approx(12.0)
    expected_spin = 3000.0 / 60.0 * 2 * np.pi
    assert launch.spin_rate == pytest.approx(expected_spin)
    # max_time + dt forwarded
    assert fake_sim.last_call[1] == pytest.approx(10.0)
    assert fake_sim.last_call[2] == pytest.approx(0.01)


def test_run_simulation_empty_trajectory_path(widget: BallFlightWidget) -> None:
    ctx, _sim, _cond = _patch_physics([])
    with ctx:
        widget._run_simulation()
    assert widget._results_text.toPlainText() == "No trajectory generated."


def test_run_simulation_handles_import_error(widget: BallFlightWidget) -> None:
    # Force the import inside _run_simulation to fail by setting the
    # target modules to None in sys.modules — Python treats that as a
    # cached failed import.
    with patch.dict(
        sys.modules,
        {
            "src.shared.python.physics.ball_simulator": None,
            "src.shared.python.physics.ball_launch_conditions": None,
        },
    ):
        widget._run_simulation()
    assert "not available" in widget._results_text.toPlainText()


def test_run_simulation_handles_generic_exception(widget: BallFlightWidget) -> None:
    sim_mod = types.ModuleType("src.shared.python.physics.ball_simulator")
    cond_mod = types.ModuleType("src.shared.python.physics.ball_launch_conditions")

    class _Boom:
        def simulate_trajectory(self, *_a, **_kw):
            raise RuntimeError("boom")

    sim_mod.BallFlightSimulator = _Boom
    cond_mod.LaunchConditions = lambda **kw: MagicMock(**kw)
    with patch.dict(
        sys.modules,
        {
            "src.shared.python.physics.ball_simulator": sim_mod,
            "src.shared.python.physics.ball_launch_conditions": cond_mod,
        },
    ):
        widget._run_simulation()
    assert "Simulation error" in widget._results_text.toPlainText()
    assert "boom" in widget._results_text.toPlainText()


def test_run_simulation_replaces_existing_plot_item(widget: BallFlightWidget) -> None:
    if widget._gl_view is None:
        pytest.skip("pyqtgraph.opengl not available in this environment")
    traj = [_make_traj_point(0.0, 0.0, 0.0, 0.0), _make_traj_point(20.0, 0.0, 5.0, 1.0)]
    ctx, _sim, _cond = _patch_physics(traj)
    with ctx:
        widget._run_simulation()
        first_item = widget._plot_item
        assert first_item is not None
        widget._run_simulation()
        # A second run should replace, not stack, the trajectory plot
        assert widget._plot_item is not None
        assert widget._plot_item is not first_item


def test_run_simulation_skips_gl_when_view_absent(widget: BallFlightWidget) -> None:
    # Drop the GL view to take the no-3D branch.
    widget._gl_view = None
    traj = [_make_traj_point(0.0, 0.0, 0.0, 0.0), _make_traj_point(10.0, 0.0, 5.0, 0.5)]
    ctx, _sim, _cond = _patch_physics(traj)
    with ctx:
        widget._run_simulation()
    assert "Ball Flight Results" in widget._results_text.toPlainText()


# --- Cleanup / Window ----------------------------------------------------


def test_cleanup_is_safe(widget: BallFlightWidget) -> None:
    # Pure logging call — must not raise.
    widget.cleanup()


def test_ball_flight_window_construction_and_close() -> None:
    _ensure_qapp()
    win = BallFlightWindow()
    assert isinstance(win, QMainWindow)
    assert win.windowTitle() == "Aerodynamic Ball Flight Simulator"
    assert win.minimumWidth() >= 1100
    assert win.minimumHeight() >= 700
    assert isinstance(win.centralWidget(), BallFlightWidget)
    assert win.statusBar() is not None
    # closeEvent should call cleanup without raising
    win.close()


def test_get_dockable_ui_returns_window() -> None:
    _ensure_qapp()
    win = get_dockable_ui()
    assert isinstance(win, BallFlightWindow)
    win.close()
