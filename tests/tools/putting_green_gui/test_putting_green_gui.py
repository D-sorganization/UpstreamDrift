"""Tests for the putting green simulator GUI.

The GUI is a thin renderer over the shared putting physics core
(via :mod:`src.tools.putting_green_gui._scene_builder`). Qt runs in
``offscreen`` mode so these stay headless-safe; OpenGL-specific assertions
are skipped when ``pyqtgraph`` is unavailable.
"""

from __future__ import annotations

import os
import sys
import time
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QMainWindow  # noqa: E402

from src.shared.python.ui.units import UnitSystem
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

# The cup-distance control is SI (metres); these mirror the historical
# 1-30 ft documented range exactly (issue #8886).
_FT_TO_M = 0.3048


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
    # The distance spinbox displays 2 decimal places (metres), so compare
    # with a tolerance that absorbs its own rounding.
    assert widget._distance_spin.value() == pytest.approx(10.0 * _FT_TO_M, abs=0.005)
    assert widget._stimp_spin.value() == pytest.approx(10.0)
    assert widget._slope_spin.value() == pytest.approx(1.0)


def test_spin_ranges_match_documented_bounds(widget: PuttingGreenWidget) -> None:
    assert widget._speed_spin.minimum() == pytest.approx(0.5)
    assert widget._speed_spin.maximum() == pytest.approx(8.0)
    assert widget._aim_spin.minimum() == pytest.approx(-45.0)
    assert widget._aim_spin.maximum() == pytest.approx(45.0)
    assert widget._distance_spin.minimum() == pytest.approx(1.0 * _FT_TO_M, abs=0.005)
    assert widget._distance_spin.maximum() == pytest.approx(30.0 * _FT_TO_M, abs=0.005)
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
    [
        (1.5, 5.0 * _FT_TO_M),
        (2.5, 15.0 * _FT_TO_M),
        (4.0, 30.0 * _FT_TO_M),
        (0.5, 1.0 * _FT_TO_M),
        (8.0, 30.0 * _FT_TO_M),
    ],
)
def test_apply_preset_updates_speed_and_distance(
    widget: PuttingGreenWidget, speed: float, dist: float
) -> None:
    widget._apply_preset(speed, dist)
    assert widget._speed_spin.value() == pytest.approx(speed)
    assert widget._distance_spin.value() == pytest.approx(dist, abs=0.005)


def test_apply_preset_does_not_touch_aim_or_green(
    widget: PuttingGreenWidget,
) -> None:
    widget._aim_spin.setValue(5.0)
    widget._stimp_spin.setValue(11.0)
    widget._slope_spin.setValue(2.5)

    widget._apply_preset(2.0, 7.0 * _FT_TO_M)

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
    widget._distance_spin.setValue(12.0 * _FT_TO_M)
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


def _first_holing_speed_on_the_spin_grid(distance_m: float) -> float:
    """Slowest 0.1 m/s speed the shared builder holes (no tuned constant)."""
    from src.tools.putting_green_gui._scene_builder import PuttConfig, build_putt_scene

    for tenths in range(5, 81):
        config = PuttConfig(
            putter_speed_ms=tenths / 10, cup_distance_m=distance_m, slope_deg=0.0
        )
        if build_putt_scene(config).holed:
            return tenths / 10
    raise AssertionError("no holing speed on the spin grid")


def test_flat_straight_putt_reports_holed(widget: PuttingGreenWidget) -> None:
    speed = _first_holing_speed_on_the_spin_grid(12.0 * _FT_TO_M)
    widget._speed_spin.setValue(speed)
    widget._aim_spin.setValue(0.0)
    widget._distance_spin.setValue(12.0 * _FT_TO_M)
    widget._slope_spin.setValue(0.0)
    widget._run_simulation()
    assert widget._scene.holed is True
    assert "HOLED" in widget._results_text.toPlainText()


def test_cross_slope_straight_putt_breaks_offline(widget: PuttingGreenWidget) -> None:
    widget._speed_spin.setValue(2.6)
    widget._aim_spin.setValue(0.0)
    widget._distance_spin.setValue(15.0 * _FT_TO_M)
    widget._slope_spin.setValue(3.0)
    widget._run_simulation()
    assert widget._scene.holed is False
    assert widget._scene.peak_break_m > 0.05
    assert "Missed" in widget._results_text.toPlainText()


def test_run_simulation_does_not_error_for_each_preset(
    widget: PuttingGreenWidget,
) -> None:
    for s, d in [
        (1.5, 5.0 * _FT_TO_M),
        (2.5, 15.0 * _FT_TO_M),
        (4.0, 30.0 * _FT_TO_M),
    ]:
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


# ---------------------------------------------------------------------------
# Unit System Tests (Issue #8886)
# ---------------------------------------------------------------------------


def test_putting_widget_imperial_mode(qapp: QApplication) -> None:
    w = PuttingGreenWidget(unit_system=UnitSystem.IMPERIAL)
    try:
        assert w.unit_system == UnitSystem.IMPERIAL
        assert w._speed_spin.suffix() == " mph"
        assert w._distance_spin.suffix() == " ft"
        assert w._speed_spin.minimum() == pytest.approx(1.0)
        assert w._speed_spin.maximum() == pytest.approx(18.0)
        assert w._distance_spin.minimum() == pytest.approx(1.0)
        assert w._distance_spin.maximum() == pytest.approx(30.0)
    finally:
        w.cleanup()


def test_putting_widget_set_unit_system_toggle(widget: PuttingGreenWidget) -> None:
    """Switching unit system dynamically updates spinbox ranges, suffixes, and values."""
    assert widget.unit_system == UnitSystem.METRIC
    assert widget._speed_spin.suffix() == " m/s"
    assert widget._distance_spin.suffix() == " m"

    widget._speed_spin.setValue(2.0)  # 2.0 m/s
    widget._distance_spin.setValue(3.0)  # 3.0 m

    widget.set_unit_system(UnitSystem.IMPERIAL)
    assert widget.unit_system == UnitSystem.IMPERIAL
    assert widget._speed_spin.suffix() == " mph"
    assert widget._distance_spin.suffix() == " ft"
    assert widget._speed_spin.value() == pytest.approx(2.0 * 2.23694, rel=1e-2)
    assert widget._distance_spin.value() == pytest.approx(3.0 * 3.28084, rel=1e-2)

    # Switch back to metric
    widget.set_unit_system(UnitSystem.METRIC)
    assert widget.unit_system == UnitSystem.METRIC
    assert widget._speed_spin.suffix() == " m/s"
    assert widget._distance_spin.suffix() == " m"
    assert widget._speed_spin.value() == pytest.approx(2.0, rel=1e-2)
    assert widget._distance_spin.value() == pytest.approx(3.0, rel=1e-2)


def test_putting_widget_imperial_metrics_formatting(widget: PuttingGreenWidget) -> None:
    widget.set_unit_system(UnitSystem.IMPERIAL)
    widget._run_simulation()
    text = widget._results_text.toPlainText()
    assert "ft (" in text
    assert "mph (" in text


# ---------------------------------------------------------------------------
# Strike location, putter balance and impact readout (#14461)
# ---------------------------------------------------------------------------


def test_strike_and_balance_controls_reach_the_config(
    widget: PuttingGreenWidget,
) -> None:
    assert widget._strike_spin.value() == pytest.approx(0.0)
    assert widget._strike_spin.minimum() == pytest.approx(-40.0)
    assert widget._strike_spin.maximum() == pytest.approx(40.0)
    assert widget._balance_combo.currentData() is None
    assert widget._balance_combo.count() == 5
    widget._strike_spin.setValue(12.0)
    widget._balance_combo.setCurrentIndex(widget._balance_combo.findData("toe_hang"))
    config = widget._current_config()
    assert config.impact_toe_m == pytest.approx(0.012)
    assert config.balance_type == "toe_hang"


def test_strike_offset_converts_with_the_unit_system(
    widget: PuttingGreenWidget,
) -> None:
    widget._strike_spin.setValue(-12.7)
    widget.set_unit_system(UnitSystem.IMPERIAL)
    assert widget._strike_spin.suffix() == " in"
    assert widget._strike_spin.value() == pytest.approx(-0.5)
    assert widget._current_config().impact_toe_m == pytest.approx(-0.0127)
    widget.set_unit_system(UnitSystem.METRIC)
    assert widget._strike_spin.value() == pytest.approx(-12.7)


@pytest.mark.parametrize(
    ("system", "skid_unit"), [(UnitSystem.METRIC, " cm"), (UnitSystem.IMPERIAL, " in")]
)
def test_metrics_show_the_impact_in_display_units(
    widget: PuttingGreenWidget, system: UnitSystem, skid_unit: str
) -> None:
    widget.set_unit_system(system)
    widget._strike_spin.setValue(widget._strike_spin.maximum() / 2)
    widget._run_simulation()
    text = widget._results_text.toPlainText()
    impact = widget._scene.impact
    for label in ("Strike:", "Launch angle:", "Spin:", "Face twist:", "Energy loss:"):
        assert label in text
    assert f"{impact.launch_angle_deg:.1f}°" in text
    assert "toe" in text and skid_unit in text and "rpm backspin" in text


# Green types (#14463)
# ---------------------------------------------------------------------------


def test_green_selector_offers_the_cross_slope_and_every_preset(
    widget: PuttingGreenWidget,
) -> None:
    from src.shared.python.putting_dynamics import GREEN_PRESETS

    combo = widget._green_combo
    assert [combo.itemData(i) for i in range(combo.count())] == [
        None,
        *GREEN_PRESETS,
    ]
    assert widget._current_config().green_type is None
    combo.setCurrentIndex(combo.findData("tiered"))
    assert widget._current_config().green_type == "tiered"
    # A preset carries its own shape, so the cross-slope controls rest.
    assert not widget._slope_spin.isEnabled()
    assert not widget._undulation_spin.isEnabled()
    widget._run_simulation()
    assert "Green:         Tiered" in widget._results_text.toPlainText()


def test_undulation_is_unit_selectable_and_seeded(widget: PuttingGreenWidget) -> None:
    assert widget._undulation_spin.suffix() == " mm"
    widget._undulation_spin.setValue(10.0)
    widget._seed_spin.setValue(7)
    config = widget._current_config()
    assert config.undulation_m == pytest.approx(0.010)
    assert config.green_seed == 7
    widget.set_unit_system(UnitSystem.IMPERIAL)
    assert widget._undulation_spin.suffix() == " in"
    assert widget._undulation_spin.value() == pytest.approx(10.0 / 25.4, abs=0.005)
    assert widget._current_config().undulation_m == pytest.approx(0.010, abs=1e-4)


def test_every_input_follows_the_dark_panel_theme(widget: PuttingGreenWidget) -> None:
    # The integer Green Seed box rendered white on the dark panel (#14463).
    from PyQt6.QtWidgets import QWidget

    style = widget.findChild(QWidget, "puttControls").styleSheet()
    for spin in (widget._seed_spin, widget._undulation_spin, widget._green_combo):
        assert type(spin).__name__ in style
    assert ":disabled" in style, "a preset greys out the shape inputs visibly"
    # The disabled look comes from the shared theme tokens, not new hex literals.
    from src.shared.python.theme.tool_stylesheet import get_tool_colors

    disabled = style[style.index(":disabled") :]
    for token in ("surface_secondary", "text_secondary", "border_default"):
        assert get_tool_colors()[token] in disabled


# ---------------------------------------------------------------------------
# Auto-Solve (#14465)
# ---------------------------------------------------------------------------


def _wait_for_solve(qapp: QApplication, widget: PuttingGreenWidget) -> None:
    deadline = time.monotonic() + 60.0
    while widget.is_solving() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.02)
    assert not widget.is_solving()


def test_auto_solve_writes_the_core_read_into_the_controls(
    qapp: QApplication, widget: PuttingGreenWidget
) -> None:
    from src.tools.putting_green_gui._scene_builder import solve_putt

    widget.set_unit_system(UnitSystem.IMPERIAL)
    widget._slope_spin.setValue(2.0)
    widget._aim_spin.setValue(-20.0)
    want = solve_putt(widget._current_config())
    widget._solve_btn.click()
    assert widget.is_solving()  # returned at once; the worker is solving
    assert not widget._solve_btn.isEnabled()
    _wait_for_solve(qapp, widget)
    assert widget._solve_btn.isEnabled()
    assert widget._speed_spin.value() == pytest.approx(
        want.putter_speed_ms * 2.236936, abs=0.06
    )
    assert widget._aim_spin.value() == pytest.approx(want.aim_deg, abs=0.06)
    assert widget._scene is not None and widget._scene.holed
    text = widget._results_text.toPlainText()
    assert "Optimum read:" in text and "mph" in text and "HOLED" in text


def test_auto_solve_rolls_the_read_on_the_green_it_solved(
    qapp: QApplication, widget: PuttingGreenWidget
) -> None:
    # A control changed mid-solve must not get the old read rolled on it,
    # and the controls rest while the worker solves (#14465, Lane 3).
    import dataclasses

    from src.tools.putting_green_gui._scene_builder import (
        build_putt_scene,
        solve_putt,
    )

    widget._slope_spin.setValue(2.0)
    solved = widget._current_config()
    want = solve_putt(solved)
    widget._solve_btn.click()
    assert not widget._slope_spin.isEnabled(), "controls rest during the solve"
    widget._slope_spin.setValue(0.0)  # a change that slipped in mid-solve
    _wait_for_solve(qapp, widget)
    assert widget._slope_spin.isEnabled()
    rolled = build_putt_scene(
        dataclasses.replace(
            solved, putter_speed_ms=want.putter_speed_ms, aim_deg=want.aim_deg
        )
    )
    assert widget._scene is not None and widget._scene.holed
    assert widget._scene.trajectory_xyz.tolist() == rolled.trajectory_xyz.tolist()


def test_auto_solve_reports_a_rejected_putt(
    qapp: QApplication, widget: PuttingGreenWidget, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(_config: object) -> None:
        raise ValueError("no read")

    monkeypatch.setattr(gui_mod, "solve_putt", refuse)
    widget._solve_btn.click()
    _wait_for_solve(qapp, widget)
    assert "Auto-Solve rejected: no read" in widget._results_text.toPlainText()
