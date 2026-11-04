"""Saved scenarios and JSON setups in the Putting Green Simulator (#14471, epic #14452).

The desktop simulator saves the dialled green and stroke as named scenarios,
compares them, and writes or reads the shared setup document. Golden parity:
the scenarios, comparison rows and documents are exactly what
:mod:`service_setup` returns, so the desktop, web and studio files agree.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.shared.python.putting_analytics import service_setup  # noqa: E402
from src.shared.python.ui.units import UnitSystem  # noqa: E402
from src.tools.putting_green_gui._scenario_panel import ScenarioPanel  # noqa: E402

pytestmark = pytest.mark.unit

_FT_M = 0.3048
_GREEN = {
    "distance_ft": 8.0,
    "grade_pct": 1.5,
    "aspect_deg": -90.0,
    "stimp": 10.0,
    "face_sd_deg": 1.2,
    "speed_cv": 0.07,
    "n": 20,
    "seed": 1,
}


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication(sys.argv[:1])


class _Green:
    """The simulator's green: the panel reads it and applies imported setups."""

    def __init__(self) -> None:
        self.args: dict[str, Any] = dict(_GREEN)
        self.applied: list[dict[str, float]] = []

    def apply(self, setup: dict[str, float]) -> None:
        self.applied.append(setup)


def _panel(green: _Green, units: list[UnitSystem]) -> ScenarioPanel:
    return ScenarioPanel(lambda: green.args, green.apply, lambda: units[0])


def test_saving_records_the_service_scenario(qapp: QApplication) -> None:
    green = _Green()
    panel = _panel(green, [UnitSystem.IMPERIAL])
    scenario = panel.run_save("Base")
    expected = service_setup.scenario_study(
        {"name": "Base", "setup": _GREEN, "n": 20, "seed": 1}
    )
    assert scenario == expected
    assert panel.scenarios() == [expected]


def test_table_shows_the_comparison_rows(qapp: QApplication) -> None:
    green = _Green()
    panel = _panel(green, [UnitSystem.IMPERIAL])
    panel.run_save("Base")
    green.args = {**_GREEN, "stimp": 12.0}
    panel.run_save("Fast")
    rows = service_setup.compare_scenarios({"scenarios": panel.scenarios()})["rows"]
    table = panel.table_text()
    assert [r[0] for r in table] == ["Base", "Fast"]
    assert table[1][1] == f"{100 * rows[1]['make_rate']:.0f}%"
    assert table[1][2] == f"{100 * rows[1]['make_rate_delta']:+.0f}"


def test_leave_columns_follow_the_units(qapp: QApplication) -> None:
    green = _Green()
    units = [UnitSystem.IMPERIAL]
    panel = _panel(green, units)
    scenario = panel.run_save("Base")
    leave = scenario["result"]["mean_leave_ft"]
    if leave is None:
        pytest.skip("every putt holed; no leave to show")
    assert panel.table_text()[0][3] == f"{leave:.2f} ft"
    units[0] = UnitSystem.METRIC
    panel.redraw()
    assert panel.table_text()[0][3] == f"{leave * _FT_M:.2f} m"


def test_export_writes_the_service_document(qapp: QApplication, tmp_path: Path) -> None:
    green = _Green()
    panel = _panel(green, [UnitSystem.IMPERIAL])
    panel.run_save("Base")
    path = tmp_path / "setup.json"
    panel.export_to(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc == service_setup.export_setup(
        {"setup": _GREEN, "scenarios": panel.scenarios()}
    )


def test_import_applies_the_setup_and_loads_the_scenarios(
    qapp: QApplication, tmp_path: Path
) -> None:
    source = _panel(_Green(), [UnitSystem.IMPERIAL])
    source.run_save("Base")
    path = tmp_path / "setup.json"
    source.export_to(path)
    green = _Green()
    panel = _panel(green, [UnitSystem.IMPERIAL])
    panel.import_from(path)
    assert green.applied == [service_setup.setup_from_args(_GREEN)]
    assert panel.scenarios() == source.scenarios()


def test_a_bad_file_is_reported_not_raised(qapp: QApplication, tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"schema_version": "9.0.0"}', encoding="utf-8")
    green = _Green()
    panel = _panel(green, [UnitSystem.IMPERIAL])
    panel.import_from(path)
    assert "schema_version" in panel.status_text()
    assert green.applied == []


def test_save_button_runs_off_the_gui_thread(qapp: QApplication) -> None:
    panel = _panel(_Green(), [UnitSystem.IMPERIAL])
    panel.start_save("Worker")
    assert panel.is_busy()
    deadline = time.monotonic() + 60.0
    while panel.is_busy() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.02)
    assert [s["name"] for s in panel.scenarios()] == ["Worker"]
