"""Tests for the Bunker Shot 3D Simulator GUI.

Most of the module is Qt-driven UI, but the ``_run_simulation`` method
contains pure-logic math (kinetic energy estimate, particle-count
clamping, spray pattern) worth covering. We exercise the widget through
its public surface using a headless QApplication.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QMainWindow  # noqa: E402

from src.tools.bunker_shot_gui import gui as gui_mod  # noqa: E402
from src.tools.bunker_shot_gui.gui import (  # noqa: E402
    BunkerShotWidget,
    BunkerShotWindow,
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


@pytest.fixture
def widget() -> BunkerShotWidget:
    _ensure_qapp()
    return BunkerShotWidget()


# ---------------------------------------------------------------------------
# Module-level / import surface
# ---------------------------------------------------------------------------


def test_pyqtgraph_available_flag_is_bool() -> None:
    assert isinstance(gui_mod.PYQTGRAPH_AVAILABLE, bool)


def test_logger_named_after_module() -> None:
    assert gui_mod.logger.name == "src.tools.bunker_shot_gui.gui"


# ---------------------------------------------------------------------------
# BunkerShotWidget construction
# ---------------------------------------------------------------------------


def test_widget_initial_spinbox_defaults(widget: BunkerShotWidget) -> None:
    assert widget._velocity_spin.value() == pytest.approx(35.0)
    assert widget._angle_spin.value() == pytest.approx(45.0)
    assert widget._depth_spin.value() == pytest.approx(3.0)


def test_widget_spinbox_ranges(widget: BunkerShotWidget) -> None:
    assert widget._velocity_spin.minimum() == pytest.approx(10.0)
    assert widget._velocity_spin.maximum() == pytest.approx(60.0)
    assert widget._angle_spin.minimum() == pytest.approx(10.0)
    assert widget._angle_spin.maximum() == pytest.approx(60.0)
    assert widget._depth_spin.minimum() == pytest.approx(1.0)
    assert widget._depth_spin.maximum() == pytest.approx(10.0)


def test_widget_spinbox_suffixes(widget: BunkerShotWidget) -> None:
    assert widget._velocity_spin.suffix() == " m/s"
    assert "°" in widget._angle_spin.suffix()
    assert widget._depth_spin.suffix() == " cm"


def test_widget_has_run_button_and_results_text(widget: BunkerShotWidget) -> None:
    assert widget._run_btn.text() == "Simulate Impact"
    assert widget._results_text.isReadOnly()
    assert "Configure impact parameters" in widget._results_text.toPlainText()


def test_widget_has_gl_view_when_pyqtgraph_available(
    widget: BunkerShotWidget,
) -> None:
    if gui_mod.PYQTGRAPH_AVAILABLE:
        assert widget._gl_view is not None
        assert widget._particles_item is None
        assert widget._vector_item is None
    else:
        assert widget._gl_view is None


# ---------------------------------------------------------------------------
# _run_simulation — pure-logic outputs reflected in results text
# ---------------------------------------------------------------------------


def test_run_simulation_updates_results_text_with_inputs(
    widget: BunkerShotWidget,
) -> None:
    widget._velocity_spin.setValue(40.0)
    widget._angle_spin.setValue(50.0)
    widget._depth_spin.setValue(5.0)
    widget._run_simulation()

    text = widget._results_text.toPlainText()
    assert "Club Velocity: 40.0 m/s" in text
    assert "Attack Angle:  50.0" in text
    assert "Entry Depth:   5.0 cm" in text
    # Kinetic energy: 0.5 * 0.3 * 40^2 = 240.0
    assert "Est. Force:    240.0 N" in text
    assert "Chrono DEM simulation mock completed." in text


def test_run_simulation_kinetic_energy_formula(widget: BunkerShotWidget) -> None:
    """Est. Force = 0.5 * 0.3 * v**2."""
    widget._velocity_spin.setValue(20.0)
    widget._angle_spin.setValue(30.0)
    widget._depth_spin.setValue(2.0)
    widget._run_simulation()
    # 0.5 * 0.3 * 400 = 60.0
    assert "Est. Force:    60.0 N" in widget._results_text.toPlainText()


def test_run_simulation_particle_count_lower_clamp(
    widget: BunkerShotWidget,
) -> None:
    """v*depth*50000 with smallest inputs would fall below 500 floor."""
    # 10 m/s * 0.01 m * 50000 = 5000 -> displayed as ~50000
    widget._velocity_spin.setValue(10.0)
    widget._angle_spin.setValue(10.0)
    widget._depth_spin.setValue(1.0)
    widget._run_simulation()
    text = widget._results_text.toPlainText()
    # raw 5000 stays within [500, 10000]; displayed is *10
    assert "Displaced sand particles: ~50000" in text


def test_run_simulation_particle_count_upper_clamp(
    widget: BunkerShotWidget,
) -> None:
    """High inputs are clamped at 10000 internally -> ~100000 displayed."""
    widget._velocity_spin.setValue(60.0)
    widget._angle_spin.setValue(45.0)
    widget._depth_spin.setValue(10.0)
    widget._run_simulation()
    text = widget._results_text.toPlainText()
    assert "Displaced sand particles: ~100000" in text


def test_run_simulation_particle_count_floor_when_tiny_product(
    widget: BunkerShotWidget,
) -> None:
    """Force the unclamped product below 500 by mocking int()."""
    # Patch int() to make num_particles tiny so the max(..., 500) branch fires.
    import src.tools.bunker_shot_gui.gui as g

    real_int = int

    def fake_int(x: object) -> int:  # type: ignore[override]
        return 10 if isinstance(x, float) else real_int(x)

    with patch.object(g, "int", fake_int, create=True):
        widget._run_simulation()
    # clamped to 500 -> displayed *10
    assert "Displaced sand particles: ~5000" in widget._results_text.toPlainText()


def test_run_simulation_creates_particle_and_vector_items_when_gl_available(
    widget: BunkerShotWidget,
) -> None:
    if not gui_mod.PYQTGRAPH_AVAILABLE:
        pytest.skip("pyqtgraph not installed")
    widget._run_simulation()
    assert widget._particles_item is not None
    assert widget._vector_item is not None


def test_run_simulation_replaces_prior_items_on_rerun(
    widget: BunkerShotWidget,
) -> None:
    if not gui_mod.PYQTGRAPH_AVAILABLE:
        pytest.skip("pyqtgraph not installed")
    widget._run_simulation()
    first_particles = widget._particles_item
    first_vector = widget._vector_item
    widget._run_simulation()
    assert widget._particles_item is not first_particles
    assert widget._vector_item is not first_vector


def test_run_simulation_without_gl_view_skips_3d_path(
    widget: BunkerShotWidget,
) -> None:
    """When _gl_view is None, simulation still updates the text panel."""
    widget._gl_view = None
    widget._run_simulation()
    assert "Bunker Shot Impact" in widget._results_text.toPlainText()


def test_run_simulation_handles_none_prior_items(
    widget: BunkerShotWidget,
) -> None:
    """First call must tolerate _particles_item / _vector_item being None."""
    if not gui_mod.PYQTGRAPH_AVAILABLE:
        pytest.skip("pyqtgraph not installed")
    assert widget._particles_item is None
    assert widget._vector_item is None
    widget._run_simulation()  # should not raise
    assert widget._particles_item is not None


def test_run_button_click_triggers_simulation(widget: BunkerShotWidget) -> None:
    widget._run_btn.click()
    assert "Bunker Shot Impact" in widget._results_text.toPlainText()


# ---------------------------------------------------------------------------
# cleanup() and BunkerShotWindow
# ---------------------------------------------------------------------------


def test_cleanup_is_noop_safe(widget: BunkerShotWidget) -> None:
    # Two calls in a row must not raise.
    widget.cleanup()
    widget.cleanup()


def test_window_basic_properties() -> None:
    _ensure_qapp()
    window = BunkerShotWindow()
    assert isinstance(window, QMainWindow)
    assert window.windowTitle() == "Bunker Shot 3D Simulator"
    assert window.minimumWidth() == 1000
    assert window.minimumHeight() == 700
    assert isinstance(window.centralWidget(), BunkerShotWidget)
    assert window.statusBar() is not None
    assert "Ready" in window.statusBar().currentMessage()


def test_window_close_event_invokes_widget_cleanup() -> None:
    _ensure_qapp()
    window = BunkerShotWindow()
    window._widget.cleanup = MagicMock()  # type: ignore[method-assign]
    window.close()
    window._widget.cleanup.assert_called_once()


def test_get_dockable_ui_returns_window() -> None:
    _ensure_qapp()
    ui = get_dockable_ui()
    assert isinstance(ui, BunkerShotWindow)
    assert isinstance(ui.centralWidget(), BunkerShotWidget)


# ---------------------------------------------------------------------------
# PYQTGRAPH_AVAILABLE=False branch in _build_ui
# ---------------------------------------------------------------------------


def test_widget_build_without_pyqtgraph_sets_gl_view_none() -> None:
    _ensure_qapp()
    with patch.object(gui_mod, "PYQTGRAPH_AVAILABLE", False):
        w = BunkerShotWidget()
    assert w._gl_view is None
    # Simulation should still work (text-only path)
    w._run_simulation()
    assert "Bunker Shot Impact" in w._results_text.toPlainText()
