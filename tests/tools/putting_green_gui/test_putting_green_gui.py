"""Tests for the putting green simulator GUI.

The GUI is a thin renderer over the real :class:`PuttingGreenSimulator`
(via :mod:`src.tools.putting_green_gui._scene_builder`). Qt runs in
``offscreen`` mode so these stay headless-safe; OpenGL-specific assertions
are skipped when ``pyqtgraph`` is unavailable.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QMainWindow  # noqa: E402

from src.tools.putting_green_gui import gui as gui_mod  # noqa: E402
from src.tools.putting_green_gui.gui import (  # noqa: E402
    PuttingGreenWidget,
    PuttingGreenWindow,
    get_dockable_ui,
)

pytestmark = pytest.mark.unit

_HAS_PYQTGRAPH = True
try:  # pragma: no cover - environment dependent
    import pyqtgraph.opengl  # noqa: F401
except ImportError:  # pragma: no cover - environment dependent
    _HAS_PYQTGRAPH = False

_needs_gl = pytest.mark.skipif(
    not _HAS_PYQTGRAPH, reason="pyqtgraph.opengl not installed"
)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    return app


@pytest.fixture
def widget(qapp: QApplication) -> PuttingGreenWidget:
    w = PuttingGreenWidget()
    yield w
    w.cleanup()


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


@_needs_gl
def test_3d_view_initialized_when_pyqtgraph_present(
    widget: PuttingGreenWidget,
) -> None:
    assert widget._gl_view is not None
    for item in (
        widget._terrain_item,
        widget._cup_item,
        widget._path_item,
        widget._ball_item,
        widget._flag_item,
        widget._aim_item,
        widget._start_item,
    ):
        assert item is not None


# ---------------------------------------------------------------------------
# Preset behaviour
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
    widget._apply_preset(99.0, 999.0)
    assert widget._speed_spin.value() == pytest.approx(widget._speed_spin.maximum())
    assert widget._distance_spin.value() == pytest.approx(
        widget._distance_spin.maximum()
    )


# ---------------------------------------------------------------------------
# Simulation behaviour (real physics)
# ---------------------------------------------------------------------------


def test_run_simulation_reports_real_metrics(widget: PuttingGreenWidget) -> None:
    widget._speed_spin.setValue(2.5)
    widget._distance_spin.setValue(12.0)
    widget._run_simulation()
    text = widget._results_text.toPlainText()
    assert "Putting Simulation" in text
    assert "Total roll" in text
    assert "Peak break" in text
    assert ("HOLED" in text) or ("Missed" in text)


def test_run_simulation_stores_scene_with_trajectory(
    widget: PuttingGreenWidget,
) -> None:
    widget._run_simulation()
    assert widget._scene is not None
    assert widget._scene.trajectory_xyz.shape[0] >= 1
    assert len(widget._scene.roll_modes) == widget._scene.trajectory_xyz.shape[0]


def test_flat_straight_putt_reports_holed(widget: PuttingGreenWidget) -> None:
    widget._speed_spin.setValue(2.2)
    widget._aim_spin.setValue(0.0)
    widget._distance_spin.setValue(12.0)
    widget._slope_spin.setValue(0.0)
    widget._run_simulation()
    assert widget._scene.holed is True
    assert "HOLED" in widget._results_text.toPlainText()


def test_cross_slope_straight_putt_breaks_offline(widget: PuttingGreenWidget) -> None:
    widget._speed_spin.setValue(2.6)
    widget._aim_spin.setValue(0.0)
    widget._distance_spin.setValue(15.0)
    widget._slope_spin.setValue(3.0)
    widget._run_simulation()
    assert widget._scene.holed is False
    assert widget._scene.peak_break_m > 0.05
    assert "Missed" in widget._results_text.toPlainText()


def test_run_simulation_does_not_error_for_each_preset(
    widget: PuttingGreenWidget,
) -> None:
    for s, d in [(1.5, 5.0), (2.5, 15.0), (4.0, 30.0)]:
        widget._apply_preset(s, d)
        widget._run_simulation()
        assert "error" not in widget._results_text.toPlainText().lower()


def test_run_simulation_negative_aim(widget: PuttingGreenWidget) -> None:
    widget._aim_spin.setValue(-30.0)
    widget._run_simulation()
    assert widget._scene is not None
    assert "error" not in widget._results_text.toPlainText().lower()


def test_run_simulation_handles_builder_failure(
    widget: PuttingGreenWidget, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(_config: object) -> None:
        raise ValueError("kaboom")

    monkeypatch.setattr(gui_mod, "build_putt_scene", boom)
    widget._run_simulation()
    assert "Simulation error" in widget._results_text.toPlainText()


def test_run_simulation_without_gl_still_updates_metrics(
    widget: PuttingGreenWidget,
) -> None:
    widget._gl_view = None  # simulate the headless / no-pyqtgraph fallback
    widget._run_simulation()
    assert "Putting Simulation" in widget._results_text.toPlainText()
    assert widget._scene is not None


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------


def test_animation_advances_index(widget: PuttingGreenWidget) -> None:
    """`_advance_animation` must step the index and push the ball position.

    This was red because `_advance_animation` returns early when `_ball_item`
    is None, and `_ball_item` is only created on the pyqtgraph.opengl path --
    an OPTIONAL extra (`body-part-viz-gl`) that is absent from the default
    install. The stepping logic itself was never broken (#8039). Injecting a
    stub ball item tests that logic in every environment instead of silently
    depending on a GPU stack.
    """
    widget._run_simulation()
    ball_item = MagicMock()
    widget._ball_item = ball_item
    widget._anim_index = 0

    widget._advance_animation()
    first_index = widget._anim_index
    assert first_index > 0, "index must advance on the first step"

    widget._advance_animation()
    assert widget._anim_index > first_index, "index must keep advancing"

    # The ball must actually be repositioned, not just counted.
    assert ball_item.setData.call_count == 2


def test_animation_stops_at_end(widget: PuttingGreenWidget) -> None:
    widget._run_simulation()
    widget._ball_item = MagicMock()
    widget._anim_index = widget._scene.trajectory_xyz.shape[0] + 5
    widget._advance_animation()
    assert not widget._anim_timer.isActive()


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


def test_widget_cleanup_is_idempotent(widget: PuttingGreenWidget) -> None:
    widget.cleanup()
    widget.cleanup()
    assert not widget._anim_timer.isActive()


# ---------------------------------------------------------------------------
# Module-level
# ---------------------------------------------------------------------------


def test_module_exposes_public_classes() -> None:
    assert hasattr(gui_mod, "PuttingGreenWidget")
    assert hasattr(gui_mod, "PuttingGreenWindow")
    assert hasattr(gui_mod, "get_dockable_ui")
    assert callable(gui_mod.get_dockable_ui)
