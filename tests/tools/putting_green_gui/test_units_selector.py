"""Units selector in the Putting Green Simulator (#14457, epic #14452).

The window exposes a Metric/Imperial combo that drives the widget's
existing ``set_unit_system`` and persists the choice through the shared
``set_unit_preference`` (``src/shared/python/ui/units.py``), like the other
tools. The putting core stays SI; only the display changes.
"""

from __future__ import annotations

import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.shared.python.ui.units import UnitSystem  # noqa: E402
from src.tools.putting_green_gui import gui as gui_mod  # noqa: E402
from src.tools.putting_green_gui.gui import PuttingGreenWidget  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication(sys.argv[:1])


@pytest.fixture
def saved(monkeypatch: pytest.MonkeyPatch) -> list[UnitSystem]:
    """Capture persisted preferences instead of writing the user's config."""
    calls: list[UnitSystem] = []
    monkeypatch.setattr(gui_mod, "set_unit_preference", calls.append)
    return calls


@pytest.fixture
def widget(qapp: QApplication, saved: list[UnitSystem]) -> PuttingGreenWidget:
    w = PuttingGreenWidget(unit_system=UnitSystem.METRIC)
    yield w
    w.cleanup()


def test_selector_offers_both_unit_systems(widget: PuttingGreenWidget) -> None:
    combo = widget.units_combo
    values = [combo.itemData(i) for i in range(combo.count())]
    assert values == [UnitSystem.METRIC.value, UnitSystem.IMPERIAL.value]
    assert combo.currentData() == UnitSystem.METRIC.value


def test_selector_starts_on_the_widget_unit_system(
    qapp: QApplication, saved: list[UnitSystem]
) -> None:
    w = PuttingGreenWidget(unit_system=UnitSystem.IMPERIAL)
    try:
        assert w.units_combo.currentData() == UnitSystem.IMPERIAL.value
        assert saved == []  # building the UI never persists anything
    finally:
        w.cleanup()


def test_choosing_imperial_converts_controls_and_persists(
    widget: PuttingGreenWidget, saved: list[UnitSystem]
) -> None:
    widget.units_combo.setCurrentIndex(1)
    assert widget.unit_system == UnitSystem.IMPERIAL
    assert widget._speed_spin.suffix() == " mph"
    assert widget._distance_spin.suffix() == " ft"
    assert saved == [UnitSystem.IMPERIAL]


def test_unit_switch_keeps_the_si_putt_unchanged(
    widget: PuttingGreenWidget,
) -> None:
    before = widget._current_config()
    widget.units_combo.setCurrentIndex(1)
    after = widget._current_config()
    assert after.putter_speed_ms == pytest.approx(before.putter_speed_ms, abs=0.03)
    assert after.cup_distance_m == pytest.approx(before.cup_distance_m, abs=0.02)


def _roll_line(widget: PuttingGreenWidget) -> str:
    text = widget._results_text.toPlainText()
    return next(line for line in text.splitlines() if line.startswith("Total roll"))


def test_metrics_rerender_in_the_chosen_units(widget: PuttingGreenWidget) -> None:
    widget._run_simulation()
    assert _roll_line(widget).endswith("ft)")  # metric first: "x m (y ft)"
    widget.units_combo.setCurrentIndex(1)
    assert _roll_line(widget).endswith("m)")  # imperial first: "x ft (y m)"


def test_programmatic_set_unit_system_syncs_selector(
    widget: PuttingGreenWidget, saved: list[UnitSystem]
) -> None:
    widget.set_unit_system(UnitSystem.IMPERIAL)
    assert widget.units_combo.currentData() == UnitSystem.IMPERIAL.value
    assert saved == []  # only a user choice persists the preference
