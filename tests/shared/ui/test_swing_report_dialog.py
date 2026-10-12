"""Swing report dialog (#12329): headless, view-model driven."""

from __future__ import annotations

import ast
import dataclasses
import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.shared.python.reporting.swing_report import (  # noqa: E402
    select_plots,
    validate_swing_report,
)
from src.shared.python.reporting.swing_report.exporter import (  # noqa: E402
    load_package,
)
from src.shared.python.reporting.swing_report.view_model import (  # noqa: E402
    ReportViewModel,
)
from src.shared.python.ui.swing_report_dialog import SwingReportDialog  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "tests" / "fixtures" / "swing_report" / "valid_full.json"
MODULES = (
    ROOT / "src/shared/python/ui/swing_report_dialog.py",
    ROOT / "src/shared/python/reporting/swing_report/view_model.py",
)


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture
def report():
    return validate_swing_report(json.loads(FIXTURE.read_text(encoding="utf-8")))


@pytest.fixture
def dialog(app, report, tmp_path):
    return SwingReportDialog(report, choose_directory=lambda _p: str(tmp_path / "out"))


def test_withheld_checkbox_disabled_with_visible_reason(dialog):
    box = dialog.plot_checkbox("face_angle_ts")
    assert not box.isEnabled()
    assert "Not Validated" in box.text()
    label = dialog.reason_label("face_angle_ts")
    assert label is not None and not label.isHidden()
    assert "Not validated" in label.text()


def test_advertised_plot_shows_uncertainty(dialog):
    label = dialog.reason_label("clubhead_speed_ts")
    assert label is not None
    assert "provisional_cross_engine_spread" in label.text()


def test_view_model_exposes_no_numeric_value_for_withheld(report):
    row = {r.plot_id: r for r in ReportViewModel(report).rows}["face_angle_ts"]
    assert not row.enabled
    for field in dataclasses.fields(row):
        value = getattr(row, field.name)
        assert not isinstance(value, (int, float)) or isinstance(value, bool)


def test_withheld_plot_cannot_be_selected(report):
    with pytest.raises(ValueError):
        ReportViewModel(report).set_custom_selection(["face_angle_ts"])


def test_full_selects_all_enabled(dialog):
    assert dialog.selected_plot_ids() == ("clubhead_speed_ts", "clubhead_path")
    assert dialog.plot_checkbox("clubhead_path").isChecked()
    assert not dialog.plot_checkbox("face_angle_ts").isChecked()


def test_custom_round_trips_selection(dialog):
    dialog.set_custom_selection(["clubhead_path"])
    assert dialog.selected_plot_ids() == ("clubhead_path",)
    assert dialog.plot_checkbox("clubhead_path").isChecked()
    assert not dialog.plot_checkbox("clubhead_speed_ts").isChecked()
    dialog.plot_checkbox("clubhead_speed_ts").setChecked(True)
    assert dialog.selected_plot_ids() == ("clubhead_speed_ts", "clubhead_path")
    dialog.full_radio.setChecked(True)
    dialog.custom_radio.setChecked(True)
    assert dialog.selected_plot_ids() == ("clubhead_speed_ts", "clubhead_path")


def test_empty_custom_selection_disables_export(dialog):
    dialog.set_custom_selection([])
    assert not dialog.export_button.isEnabled()


def test_export_calls_select_plots_and_round_trips(dialog, report, tmp_path):
    dialog.set_custom_selection(["clubhead_path"])
    assert dialog.export_selected()
    loaded = load_package(tmp_path / "out")
    assert loaded == select_plots(report, ["clubhead_path"]).report
    assert loaded.scope == "custom"


def test_cancelled_directory_exports_nothing(app, report):
    dlg = SwingReportDialog(report, choose_directory=lambda _p: "")
    assert not dlg.export_selected()


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_no_engine_imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    forbidden = [n for n in names if "engine" in n.lower() or "physics" in n.lower()]
    assert forbidden == []
