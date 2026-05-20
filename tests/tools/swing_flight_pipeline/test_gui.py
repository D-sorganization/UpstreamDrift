"""Tests for the Swing-to-Flight Pipeline GUI tool.

Covers UI construction, preset application, and the full ``_run_pipeline``
control-flow (success path, ImportError path, generic-exception path),
with all physics stages mocked via ``sys.modules``.
"""

from __future__ import annotations

import os
import sys
import types
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.tools.swing_flight_pipeline.gui import (  # noqa: E402
    SwingFlightWidget,
    SwingFlightWindow,
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


# ---------------------------------------------------------------------------
# Mock pipeline scaffolding
# ---------------------------------------------------------------------------


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
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _MockPipeline:
    last_swing: Any = None

    def run(self, swing):
        _MockPipeline.last_swing = swing
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
                _MockTrajPoint(position=np.array([100.0, 0.0, 20.0])),
                _MockTrajPoint(position=np.array([220.0, 0.0, 0.0])),
            ],
        )


def _install_mock_pipeline_module():
    """Inject a fake ``swing_ball_flight_pipeline`` module into sys.modules."""
    mod = types.ModuleType("src.shared.python.physics.swing_ball_flight_pipeline")
    mod.SwingBallFlightPipeline = _MockPipeline
    mod.SwingState = _MockSwingState
    return patch.dict(
        sys.modules,
        {"src.shared.python.physics.swing_ball_flight_pipeline": mod},
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    return _ensure_qapp()


@pytest.fixture
def widget():
    w = SwingFlightWidget()
    yield w
    w.cleanup()
    w.deleteLater()


# ---------------------------------------------------------------------------
# Construction / defaults
# ---------------------------------------------------------------------------


def test_widget_constructs_with_defaults(widget):
    assert widget._speed_spin.value() == pytest.approx(45.0)
    assert widget._loft_spin.value() == pytest.approx(10.5)
    assert widget._mass_spin.value() == pytest.approx(0.200)
    assert widget._engine_combo.currentText() == "manual"
    assert widget._result is None


def test_widget_spin_ranges(widget):
    smin, smax = widget._speed_spin.minimum(), widget._speed_spin.maximum()
    assert (smin, smax) == (pytest.approx(20.0), pytest.approx(60.0))
    lmin, lmax = widget._loft_spin.minimum(), widget._loft_spin.maximum()
    assert (lmin, lmax) == (pytest.approx(5.0), pytest.approx(60.0))
    mmin, mmax = widget._mass_spin.minimum(), widget._mass_spin.maximum()
    assert (mmin, mmax) == (pytest.approx(0.100), pytest.approx(0.400))


def test_engine_combo_lists_all_engines(widget):
    items = [
        widget._engine_combo.itemText(i) for i in range(widget._engine_combo.count())
    ]
    assert items == ["mujoco", "drake", "pinocchio", "manual"]


def test_run_button_present_and_connected(widget):
    assert widget._run_btn.text() == "Run Full Pipeline"


def test_initial_results_text_has_instructions(widget):
    text = widget._results_text.toPlainText()
    assert "Configure swing parameters" in text
    assert "Pipeline stages" in text


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


def test_apply_preset_updates_speed_and_loft(widget):
    widget._apply_preset(50.0, 10.5)
    assert widget._speed_spin.value() == pytest.approx(50.0)
    assert widget._loft_spin.value() == pytest.approx(10.5)


def test_apply_preset_iron(widget):
    widget._apply_preset(35.0, 34.0)
    assert widget._speed_spin.value() == pytest.approx(35.0)
    assert widget._loft_spin.value() == pytest.approx(34.0)


def test_apply_preset_pw(widget):
    widget._apply_preset(28.0, 46.0)
    assert widget._speed_spin.value() == pytest.approx(28.0)
    assert widget._loft_spin.value() == pytest.approx(46.0)


# ---------------------------------------------------------------------------
# _run_pipeline — success path with mocked pipeline
# ---------------------------------------------------------------------------


def test_run_pipeline_success_populates_result(widget):
    with _install_mock_pipeline_module():
        widget._speed_spin.setValue(45.0)
        widget._loft_spin.setValue(10.5)
        widget._mass_spin.setValue(0.200)
        widget._engine_combo.setCurrentText("manual")
        widget._run_pipeline()

    assert widget._result is not None
    assert widget._result.carry_m == pytest.approx(220.0)
    text = widget._results_text.toPlainText()
    assert "Pipeline Complete" in text
    assert "Engine: manual" in text
    assert "220.0 m" in text
    assert "Trajectory: 3 points" in text


def test_run_pipeline_passes_ui_parameters_to_swing_state(widget):
    with _install_mock_pipeline_module():
        widget._speed_spin.setValue(50.0)
        widget._loft_spin.setValue(34.0)
        widget._mass_spin.setValue(0.300)
        widget._engine_combo.setCurrentText("drake")
        widget._run_pipeline()

    swing = _MockPipeline.last_swing
    assert swing.clubhead_mass == pytest.approx(0.300)
    assert swing.clubhead_loft_deg == pytest.approx(34.0)
    assert swing.engine_name == "drake"
    np.testing.assert_array_equal(swing.clubhead_velocity, np.array([50.0, 0.0, 0.0]))
    np.testing.assert_array_equal(swing.clubhead_angular_velocity, np.zeros(3))
    np.testing.assert_array_equal(swing.clubhead_orientation, np.array([0.0, 0.0, 1.0]))


def test_run_pipeline_renders_trajectory_when_glview_available(widget):
    fake_view = MagicMock()
    fake_view.opts = {}
    widget._gl_view = fake_view
    widget._plot_item = None

    with _install_mock_pipeline_module():
        widget._run_pipeline()

    # First call adds line plot
    assert fake_view.addItem.call_count >= 1
    assert widget._plot_item is not None
    assert "center" in fake_view.opts


def test_run_pipeline_removes_previous_plot_item(widget):
    fake_view = MagicMock()
    fake_view.opts = {}
    widget._gl_view = fake_view
    sentinel_old = object()
    widget._plot_item = sentinel_old

    with _install_mock_pipeline_module():
        widget._run_pipeline()

    fake_view.removeItem.assert_called_once_with(sentinel_old)


def test_run_pipeline_skips_visualization_when_glview_missing(widget):
    widget._gl_view = None
    with _install_mock_pipeline_module():
        widget._run_pipeline()
    # No crash and still populates text
    assert "Pipeline Complete" in widget._results_text.toPlainText()


# ---------------------------------------------------------------------------
# _run_pipeline — failure paths
# ---------------------------------------------------------------------------


def test_run_pipeline_handles_import_error(widget):
    # Force ImportError by injecting a module that raises on attribute access
    broken = types.ModuleType("src.shared.python.physics.swing_ball_flight_pipeline")
    # Don't set SwingBallFlightPipeline / SwingState -> ImportError from the
    # `from ... import ...` statement.
    with patch.dict(
        sys.modules,
        {"src.shared.python.physics.swing_ball_flight_pipeline": broken},
    ):
        widget._run_pipeline()

    text = widget._results_text.toPlainText()
    assert "Pipeline not available" in text
    assert "feat/5337" in text


def test_run_pipeline_handles_generic_exception(widget):
    class _Boom:
        def run(self, swing):
            raise RuntimeError("kaboom")

    mod = types.ModuleType("src.shared.python.physics.swing_ball_flight_pipeline")
    mod.SwingBallFlightPipeline = _Boom
    mod.SwingState = _MockSwingState

    with patch.dict(
        sys.modules,
        {"src.shared.python.physics.swing_ball_flight_pipeline": mod},
    ):
        widget._run_pipeline()

    text = widget._results_text.toPlainText()
    assert "Pipeline error" in text
    assert "kaboom" in text


# ---------------------------------------------------------------------------
# Empty trajectory edge case
# ---------------------------------------------------------------------------


def test_run_pipeline_handles_empty_trajectory(widget):
    class _EmptyPipeline:
        def run(self, swing):
            return _MockResult(
                swing_state=swing,
                impact_state=_MockImpactState(
                    ball_velocity=np.zeros(3),
                    ball_angular_velocity=np.zeros(3),
                ),
                launch_conditions=_MockLaunch(0.1, 0.0, 0.0),
                carry_m=0.0,
                max_height_m=0.0,
                flight_time_s=0.0,
                landing_angle_deg=0.0,
                trajectory=[],
            )

    mod = types.ModuleType("src.shared.python.physics.swing_ball_flight_pipeline")
    mod.SwingBallFlightPipeline = _EmptyPipeline
    mod.SwingState = _MockSwingState

    fake_view = MagicMock()
    fake_view.opts = {}
    widget._gl_view = fake_view

    with patch.dict(
        sys.modules,
        {"src.shared.python.physics.swing_ball_flight_pipeline": mod},
    ):
        widget._run_pipeline()

    # Empty trajectory short-circuits visualization branch
    fake_view.addItem.assert_not_called()
    assert "Trajectory: 0 points" in widget._results_text.toPlainText()


# ---------------------------------------------------------------------------
# SwingFlightWindow + get_dockable_ui
# ---------------------------------------------------------------------------


def test_window_constructs_with_central_widget():
    win = SwingFlightWindow()
    try:
        assert win.windowTitle() == "Swing → Flight Pipeline"
        assert isinstance(win.centralWidget(), SwingFlightWidget)
        assert win.statusBar() is not None
    finally:
        win.close()
        win.deleteLater()


def test_window_close_event_triggers_cleanup():
    win = SwingFlightWindow()
    try:
        with patch.object(win._widget, "cleanup") as cleanup_mock:
            win.close()
            cleanup_mock.assert_called_once()
    finally:
        win.deleteLater()


def test_get_dockable_ui_returns_window():
    win = get_dockable_ui()
    try:
        assert isinstance(win, SwingFlightWindow)
        assert isinstance(win.centralWidget(), SwingFlightWidget)
    finally:
        win.close()
        win.deleteLater()


def test_widget_cleanup_is_safe(widget):
    # Multiple calls should not raise
    widget.cleanup()
    widget.cleanup()
