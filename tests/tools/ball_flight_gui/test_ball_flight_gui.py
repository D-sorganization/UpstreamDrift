"""Tests for the aerodynamic ball flight simulator GUI.

The scope `src/tools/ball_flight_gui/gui.py` is mostly PyQt6 widget
construction with one logic method (`_apply_preset`) and one orchestration
method (`_run_simulation`) that wires the UI to the shared physics engine.
We construct widgets under an offscreen QApplication and mock the physics
import so the suite runs fast and deterministic.
"""

from __future__ import annotations

import os
import math
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

pytestmark = pytest.mark.unit

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
        def __init__(self, *args, **kwargs):
            # Production now passes env= (issue #8818); record it so tests
            # can assert the environment wiring.
            _FakeSim.last_env = kwargs.get("env")

        def simulate_trajectory(self, launch, max_time, dt):
            _FakeSim.last_call = (launch, max_time, dt)
            return trajectory

    # Use the REAL LaunchConditions and fake only the (heavy) solver.
    #
    # The previous hand-rolled `_FakeLaunchConditions` had no `from_user_units`
    # classmethod, which production has called since #8039 was filed. Every
    # simulation path therefore died in the widget's generic error handler and
    # all four tests below asserted against an exception message. A stand-in
    # that has to track a real constructor is exactly the thing that goes stale;
    # the real dataclass is cheap and cannot drift.
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
        LaunchConditions,
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
    # The decorative aero checkboxes were removed (#8818): the simulator has
    # no backing parameter for dimple/Magnus-toggle/seam effects.
    assert not hasattr(widget, "_chk_dimples")
    assert not hasattr(widget, "_chk_magnus")
    assert not hasattr(widget, "_chk_seam")


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
    # Verify LaunchConditions was built with correct unit conversions.
    # These now assert against the real dataclass contract documented on
    # LaunchConditions: velocity in m/s, launch_angle in RADIANS, spin_rate in
    # RPM (from_user_units passes RPM through unchanged).
    launch = fake_sim.last_call[0]
    assert launch.velocity == pytest.approx(150.0 * 0.44704)
    assert launch.launch_angle == pytest.approx(math.radians(12.0))
    assert launch.spin_rate == pytest.approx(3000.0)
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
    """The widget must surface an unexpected solver failure verbatim.

    This previously failed because a *different* exception (missing
    `from_user_units` on the launch-conditions stand-in) was raised before the
    solver was ever reached -- so the error-reporting path this test exists to
    cover had no coverage at all (#8039). The real LaunchConditions is used, so
    "boom" is now the only thing that can go wrong.
    """
    from src.shared.python.physics.ball_launch_conditions import (
        EnvironmentalConditions,
        LaunchConditions,
    )

    sim_mod = types.ModuleType("src.shared.python.physics.ball_simulator")
    cond_mod = types.ModuleType("src.shared.python.physics.ball_launch_conditions")

    class _Boom:
        def __init__(self, *args, **kwargs):
            pass

        def simulate_trajectory(self, *_a, **_kw):
            raise RuntimeError("boom")

    sim_mod.BallFlightSimulator = _Boom
    cond_mod.LaunchConditions = LaunchConditions
    cond_mod.EnvironmentalConditions = EnvironmentalConditions
    with patch.dict(
        sys.modules,
        {
            "src.shared.python.physics.ball_simulator": sim_mod,
            "src.shared.python.physics.ball_launch_conditions": cond_mod,
        },
    ):
        widget._run_simulation()
    text = widget._results_text.toPlainText()
    assert "Simulation error" in text
    assert "boom" in text
    assert "from_user_units" not in text, (
        "the launch-conditions stand-in broke before the solver was reached"
    )


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


# --- Environment / aero wiring (#8818) -----------------------------------


def test_build_wind_vector_convention_and_units():
    from src.tools.ball_flight_gui.gui import build_wind_vector

    tail = build_wind_vector(10.0, 0.0)
    np.testing.assert_allclose(tail, [10.0 * 0.44704, 0.0, 0.0], atol=1e-12)
    head = build_wind_vector(10.0, 180.0)
    np.testing.assert_allclose(head, [-10.0 * 0.44704, 0.0, 0.0], atol=1e-12)
    cross = build_wind_vector(10.0, 90.0)
    np.testing.assert_allclose(cross, [0.0, 10.0 * 0.44704, 0.0], atol=1e-12)


def test_build_wind_vector_rejects_negative_speed():
    from src.tools.ball_flight_gui.gui import build_wind_vector

    with pytest.raises(ValueError):
        build_wind_vector(-1.0, 0.0)


def test_combine_spins_pure_backspin_keeps_classic_axis():
    from src.tools.ball_flight_gui.gui import combine_spins

    rate, axis = combine_spins(2500.0, 0.0)
    assert rate == pytest.approx(2500.0)
    np.testing.assert_allclose(axis, [0.0, -1.0, 0.0])


def test_combine_spins_sidespin_tilts_axis():
    from src.tools.ball_flight_gui.gui import combine_spins

    rate, axis = combine_spins(2500.0, 1000.0)
    assert rate == pytest.approx(math.hypot(2500.0, 1000.0))
    assert np.linalg.norm(axis) == pytest.approx(1.0)
    # positive sidespin = rightward curve = spin about -z
    assert axis[2] < 0.0


def test_combine_spins_zero_spin_defaults():
    from src.tools.ball_flight_gui.gui import combine_spins

    rate, axis = combine_spins(0.0, 0.0)
    assert rate == 0.0
    np.testing.assert_allclose(axis, [0.0, -1.0, 0.0])


def test_wind_widgets_reach_simulator_environment(widget: BallFlightWidget) -> None:
    """Changing wind in the GUI must change what the simulator receives."""
    traj = [_make_traj_point(0.0, 0.0, 0.0, 0.0), _make_traj_point(10.0, 0.0, 5.0, 1.0)]
    ctx, fake_sim, _cond = _patch_physics(traj)
    with ctx:
        widget._wind_speed.setValue(0.0)
        widget._run_simulation()
        env_calm = fake_sim.last_env

        widget._wind_speed.setValue(20.0)
        widget._wind_dir.setValue(180.0)  # headwind
        widget._run_simulation()
        env_windy = fake_sim.last_env

    assert env_calm is not None and env_windy is not None
    np.testing.assert_allclose(env_calm.wind_velocity, [0.0, 0.0, 0.0])
    assert env_windy.wind_velocity[0] == pytest.approx(-20.0 * 0.44704)
    assert not np.allclose(env_calm.wind_velocity, env_windy.wind_velocity)


def test_altitude_widget_lowers_air_density(widget: BallFlightWidget) -> None:
    traj = [_make_traj_point(0.0, 0.0, 0.0, 0.0), _make_traj_point(10.0, 0.0, 5.0, 1.0)]
    ctx, fake_sim, _cond = _patch_physics(traj)
    with ctx:
        widget._altitude.setValue(0.0)
        widget._run_simulation()
        rho_sea = fake_sim.last_env.air_density

        widget._altitude.setValue(5000.0)  # feet
        widget._run_simulation()
        rho_high = fake_sim.last_env.air_density

    assert rho_high < rho_sea


def test_sidespin_widget_reaches_launch_conditions(widget: BallFlightWidget) -> None:
    traj = [_make_traj_point(0.0, 0.0, 0.0, 0.0), _make_traj_point(10.0, 0.0, 5.0, 1.0)]
    ctx, fake_sim, _cond = _patch_physics(traj)
    with ctx:
        widget._spin_spin.setValue(2500.0)
        widget._sidespin_spin.setValue(1000.0)
        widget._run_simulation()

    launch = fake_sim.last_call[0]
    assert launch.spin_rate == pytest.approx(math.hypot(2500.0, 1000.0))
    assert launch.spin_axis[2] < 0.0


def test_wind_changes_aerodynamic_forces() -> None:
    """Physics-level check: the environment wind alters the drag force.

    Uses the real simulator's Python force model (no Rust kernel needed),
    proving the GUI-built EnvironmentalConditions is not decorative.
    """
    from src.shared.python.physics.ball_launch_conditions import (
        EnvironmentalConditions,
        LaunchConditions,
    )
    from src.shared.python.physics.ball_simulator import BallFlightSimulator

    launch = LaunchConditions.from_user_units(
        velocity=70.0, launch_angle_deg=11.0, spin_rate_rpm=2500.0
    )
    vel = np.array([70.0, 0.0, 0.0])

    calm = BallFlightSimulator(env=EnvironmentalConditions())
    headwind = BallFlightSimulator(
        env=EnvironmentalConditions(wind_velocity=np.array([-10.0, 0.0, 0.0]))
    )
    drag_calm = calm._calculate_forces(vel, launch)["drag"]
    drag_head = headwind._calculate_forces(vel, launch)["drag"]
    assert np.linalg.norm(drag_head) > np.linalg.norm(drag_calm)


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
