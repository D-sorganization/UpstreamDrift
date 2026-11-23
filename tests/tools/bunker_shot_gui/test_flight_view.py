"""The ball-flight tab: units switch and refusal banner (issue #15497)."""

from __future__ import annotations

import math
import os
import sys

import pytest

pytest.importorskip("PyQt6", reason="the workbench shell needs a Qt binding")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.tools.bunker_shot_gui.flight import BunkerFlight  # noqa: E402
from src.tools.bunker_shot_gui.render_flight import FlightUnits  # noqa: E402
from src.tools.bunker_shot_gui.viewport_widgets import BallFlightWidget  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.fixture(scope="session", autouse=True)
def qapp() -> QApplication:
    """One offscreen QApplication for the module."""
    return QApplication.instance() or QApplication(sys.argv[:1])


def _flight() -> BunkerFlight:
    arc = tuple((float(x), 0.0, 3.0 * math.sin(math.pi * x / 9)) for x in range(10))
    return BunkerFlight(arc, ((9.0, 0.0, 0.0), (11.0, 0.5, 0.0)), carry_m=9.0)


def test_starts_empty_with_a_banner_and_no_lines() -> None:
    widget = BallFlightWidget("flight")
    assert not widget.has_flight
    assert all(len(ax.lines) == 0 for ax in widget._canvas.fig.axes)


def test_flight_then_refusal_leaves_nothing_stale() -> None:
    widget = BallFlightWidget("flight")
    widget.set_flight(_flight(), reason="")
    assert widget.has_flight
    assert any(len(ax.lines) for ax in widget._canvas.fig.axes)
    widget.set_flight(None, reason="refused: out of envelope")
    assert not widget.has_flight
    assert all(len(ax.lines) == 0 for ax in widget._canvas.fig.axes)
    assert "refused: out of envelope" in widget._readout.text()


def test_units_switch_redraws_in_yards() -> None:
    widget = BallFlightWidget("flight")
    widget.set_flight(_flight(), reason="")
    widget.set_units(FlightUnits.YARDS)
    assert "yd" in widget._canvas.fig.axes[0].get_xlabel()
    assert "yd" in widget._readout.text()


def test_workbench_shell_has_a_flight_tab() -> None:
    from src.tools.bunker_shot_gui.gui import BunkerShotWidget

    shell = BunkerShotWidget()
    tabs = [shell._views.tabText(i) for i in range(shell._views.count())]
    assert "Ball flight and green" in tabs
    assert not shell._flight_a.has_flight


def _seeded_shell():  # noqa: ANN202
    from src.tools.bunker_shot_gui.gui import BunkerShotWidget

    shell = BunkerShotWidget()
    shell._flight_a.set_flight(_flight(), reason="")
    assert shell._flight_a.has_flight
    assert any(len(ax.lines) for ax in shell._flight_a._canvas.fig.axes)
    return shell


def test_loading_a_saved_record_clears_a_stale_flight(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A loaded (refused) record carries no flight, so none may stay drawn."""
    from src.tools.bunker_shot_gui import gui

    record = {
        "inputs": {},
        "verdict": {"status": "beyond_validation", "headline": "refused"},
        "digest": "d",
        "stamp": "s",
        "schema_version": "bunker-workbench-evaluation/1",
    }
    monkeypatch.setattr(gui, "load_workbench_evaluation", lambda _path: record)
    shell = _seeded_shell()
    shell.load_evaluation(tmp_path / "refused.json")
    assert not shell._flight_a.has_flight
    assert all(len(ax.lines) == 0 for ax in shell._flight_a._canvas.fig.axes)
    assert "loaded" in shell._flight_a._readout.text()


def test_input_error_clears_a_stale_flight() -> None:
    """A rejected input must not sit beside another shot's trajectory."""
    from src.tools.bunker_shot_gui.gui import WorkbenchInputError

    shell = _seeded_shell()
    shell._show_input_error(WorkbenchInputError("loft out of range"))
    assert not shell._flight_a.has_flight
    assert all(len(ax.lines) == 0 for ax in shell._flight_a._canvas.fig.axes)
    assert "input" in shell._flight_a._readout.text()
