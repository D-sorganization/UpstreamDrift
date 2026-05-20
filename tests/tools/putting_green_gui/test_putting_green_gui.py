"""Tests for the putting green simulator GUI (non-GUI logic + smoke).

Qt runs in ``offscreen`` mode so these tests stay headless-safe and fast.
"""

from __future__ import annotations

import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
from PyQt6.QtWidgets import QApplication, QMainWindow  # noqa: E402

from src.tools.putting_green_gui import gui as gui_mod  # noqa: E402
from src.tools.putting_green_gui.gui import (  # noqa: E402
    PuttingGreenWidget,
    PuttingGreenWindow,
    get_dockable_ui,
)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    return app


@pytest.fixture
def widget(qapp: QApplication) -> PuttingGreenWidget:
    return PuttingGreenWidget()


# ---------------------------------------------------------------------------
# Construction / defaults
# ---------------------------------------------------------------------------


def test_widget_constructs_with_default_values(widget: PuttingGreenWidget) -> None:
    assert widget._speed_spin.value() == pytest.approx(2.5)
    assert widget._aim_spin.value() == pytest.approx(0.0)
    assert widget._distance_spin.value() == pytest.approx(10.0)
    assert widget._stimp_spin.value() == pytest.approx(10.0)
    assert widget._slope_spin.value() == pytest.approx(1.0)


def test_spin_ranges_match_documented_bounds(widget: PuttingGreenWidget) -> None:
    assert widget._speed_spin.minimum() == pytest.approx(0.5)
    assert widget._speed_spin.maximum() == pytest.approx(8.0)
    assert widget._aim_spin.minimum() == pytest.approx(-45.0)
    assert widget._aim_spin.maximum() == pytest.approx(45.0)
    assert widget._distance_spin.minimum() == pytest.approx(1.0)
    assert widget._distance_spin.maximum() == pytest.approx(30.0)
    assert widget._stimp_spin.minimum() == pytest.approx(6.0)
    assert widget._stimp_spin.maximum() == pytest.approx(14.0)
    assert widget._slope_spin.minimum() == pytest.approx(0.0)
    assert widget._slope_spin.maximum() == pytest.approx(5.0)


def test_default_results_text_shows_help(widget: PuttingGreenWidget) -> None:
    text = widget._results_text.toPlainText()
    assert "Configure putt parameters" in text
    assert "Physics model" in text
    assert widget._results_text.isReadOnly()


def test_3d_view_initialized_when_pyqtgraph_present(
    widget: PuttingGreenWidget,
) -> None:
    # pyqtgraph.opengl is installed in CI; if so the view + items exist.
    assert widget._gl_view is not None
    assert widget._terrain_item is not None
    assert widget._cup_item is not None
    assert widget._path_item is not None


# ---------------------------------------------------------------------------
# Preset behavior
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "speed, dist",
    [(1.5, 5.0), (2.5, 15.0), (4.0, 30.0), (0.5, 1.0), (8.0, 30.0)],
)
def test_apply_preset_updates_speed_and_distance(
    widget: PuttingGreenWidget, speed: float, dist: float
) -> None:
    widget._apply_preset(speed, dist)
    assert widget._speed_spin.value() == pytest.approx(speed)
    assert widget._distance_spin.value() == pytest.approx(dist)


def test_apply_preset_does_not_touch_aim_or_green(
    widget: PuttingGreenWidget,
) -> None:
    widget._aim_spin.setValue(5.0)
    widget._stimp_spin.setValue(11.0)
    widget._slope_spin.setValue(2.5)

    widget._apply_preset(2.0, 7.0)

    assert widget._aim_spin.value() == pytest.approx(5.0)
    assert widget._stimp_spin.value() == pytest.approx(11.0)
    assert widget._slope_spin.value() == pytest.approx(2.5)


def test_apply_preset_clamped_to_spin_bounds(widget: PuttingGreenWidget) -> None:
    # QDoubleSpinBox clamps out-of-range values.
    widget._apply_preset(99.0, 999.0)
    assert widget._speed_spin.value() == pytest.approx(widget._speed_spin.maximum())
    assert widget._distance_spin.value() == pytest.approx(
        widget._distance_spin.maximum()
    )


# ---------------------------------------------------------------------------
# Simulation behavior
# ---------------------------------------------------------------------------


def test_run_simulation_updates_results_text(widget: PuttingGreenWidget) -> None:
    widget._speed_spin.setValue(3.0)
    widget._aim_spin.setValue(2.0)
    widget._stimp_spin.setValue(11.0)
    widget._slope_spin.setValue(1.5)
    widget._distance_spin.setValue(12.0)

    widget._run_simulation()
    text = widget._results_text.toPlainText()

    assert "Putting Simulation" in text
    assert "3.0 m/s" in text
    assert "2.0" in text
    assert "11.0" in text
    assert "1.5" in text


def test_run_simulation_does_not_error_for_each_preset(
    widget: PuttingGreenWidget,
) -> None:
    for s, d in [(1.5, 5.0), (2.5, 15.0), (4.0, 30.0)]:
        widget._apply_preset(s, d)
        widget._run_simulation()
        assert "error" not in widget._results_text.toPlainText().lower()


def test_run_simulation_updates_3d_items(widget: PuttingGreenWidget) -> None:
    widget._distance_spin.setValue(8.0)
    widget._run_simulation()
    # Cup item should be moved to a non-default position.
    pos = widget._cup_item.pos
    assert pos.shape == (1, 3)
    # 8 ft -> 2.4384 m
    assert pos[0, 0] == pytest.approx(8.0 * 0.3048, rel=1e-3)


def test_run_simulation_negative_aim(widget: PuttingGreenWidget) -> None:
    widget._aim_spin.setValue(-30.0)
    widget._run_simulation()
    assert "-30.0" in widget._results_text.toPlainText()


def test_run_simulation_handles_simulator_import_failure(
    widget: PuttingGreenWidget, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force the dynamic import to fail.
    monkeypatch.setitem(sys.modules, "src.engines.physics_engines", None)
    widget._run_simulation()
    text = widget._results_text.toPlainText()
    assert "Simulator not available" in text or "Putting Simulation" in text


def test_run_simulation_handles_runtime_exception(
    widget: PuttingGreenWidget, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_a, **_kw):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(np, "linspace", boom)
    widget._run_simulation()
    assert "Simulation error" in widget._results_text.toPlainText()


# ---------------------------------------------------------------------------
# Window / dockable interface
# ---------------------------------------------------------------------------


def test_get_dockable_ui_returns_window(qapp: QApplication) -> None:
    win = get_dockable_ui()
    assert isinstance(win, PuttingGreenWindow)
    assert isinstance(win, QMainWindow)
    assert win.windowTitle() == "Putting Green Simulator"
    assert win.minimumSize().width() == 1000
    assert win.minimumSize().height() == 700
    assert isinstance(win.centralWidget(), PuttingGreenWidget)
    assert win.statusBar() is not None


def test_window_close_event_calls_cleanup(qapp: QApplication) -> None:
    win = PuttingGreenWindow()
    called = []
    win._widget.cleanup = lambda: called.append(True)  # type: ignore[assignment]
    win.close()
    assert called == [True]


def test_widget_cleanup_is_safe(widget: PuttingGreenWidget) -> None:
    # cleanup should be idempotent and not raise.
    widget.cleanup()
    widget.cleanup()


# ---------------------------------------------------------------------------
# Module-level
# ---------------------------------------------------------------------------


def test_module_exposes_public_classes() -> None:
    assert hasattr(gui_mod, "PuttingGreenWidget")
    assert hasattr(gui_mod, "PuttingGreenWindow")
    assert hasattr(gui_mod, "get_dockable_ui")
    assert callable(gui_mod.get_dockable_ui)
