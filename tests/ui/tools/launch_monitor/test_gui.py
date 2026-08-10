from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PyQt6")

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

FIXTURES = Path(__file__).parents[3] / "fixtures" / "launch_monitor"


@pytest.fixture
def widget(qapp):  # noqa: ANN001, ANN201
    from src.tools.launch_monitor_analytics.gui import MainWidget

    result = MainWidget()
    yield result
    result.deleteLater()


def test_workbench_exposes_all_analysis_workspaces(widget) -> None:  # noqa: ANN001
    labels = [widget.tabs.tabText(index) for index in range(widget.tabs.count())]
    assert labels == [
        "Sessions",
        "Data Treatment",
        "Relationships",
        "Flexible Analysis",
        "Models",
        "Monitor Comparison",
        "Dispersion",
        "Trends",
        "Reports",
    ]
    assert widget.windowTitle() == "Launch Monitor Analytics"


def test_flexible_analysis_uses_arbitrary_numeric_source_fields(widget) -> None:  # noqa: ANN001
    count = 30
    speed = np.linspace(40.0, 50.0, count)
    frame = pd.DataFrame(
        {
            "shot_id": [f"shot-{index}" for index in range(count)],
            "session_id": "session-a",
            "monitor_vendor": "TrackMan",
            "ball_speed": speed * 1.48,
            "source::temperature": np.linspace(15.0, 25.0, count),
        }
    )
    widget.analysis_frame = frame
    widget._refresh_all()
    panel = widget.flexible_analysis
    panel.outcome_combo.setCurrentText("ball_speed")
    for index in range(panel.predictor_list.count()):
        item = panel.predictor_list.item(index)
        item.setSelected(item.text() == "source::temperature")
    panel.min_samples_spin.setValue(10)

    result = panel.run_analysis()

    assert result.request.predictors == ("source::temperature",)
    assert result.dataset.monitor_vendors == ("TrackMan",)
    assert panel.summary_table.rowCount() >= 1
    assert result.dataset.fingerprint_sha256 in panel.details.toPlainText()


def test_flexible_analysis_controls_are_accessibly_named(widget) -> None:  # noqa: ANN001
    panel = widget.flexible_analysis
    controls = (
        panel.outcome_combo,
        panel.predictor_list,
        panel.mode_combo,
        panel.method_combo,
        panel.missing_combo,
        panel.group_combo,
        panel.min_samples_spin,
        panel.confidence_spin,
        panel.run_button,
    )
    assert all(control.accessibleName() for control in controls)
    assert all(control.toolTip() for control in controls)


def test_flexible_analysis_button_reports_invalid_selection_inline(
    widget, qapp
) -> None:  # noqa: ANN001
    frame = pd.DataFrame(
        {
            "shot_id": [f"shot-{index}" for index in range(12)],
            "ball_speed": np.linspace(50.0, 60.0, 12),
            "club_speed": np.linspace(35.0, 45.0, 12),
        }
    )
    panel = widget.flexible_analysis
    panel.set_frame(frame)
    panel.outcome_combo.setCurrentText("ball_speed")
    for index in range(panel.predictor_list.count()):
        item = panel.predictor_list.item(index)
        item.setSelected(item.text() == "ball_speed")

    panel.run_button.click()
    qapp.processEvents()

    assert panel.last_result is None
    assert "outcome cannot also be a predictor" in panel.status_label.text()
    assert panel.status_label.accessibleName() == "Flexible Analysis Status"


def test_import_refreshes_sessions_data_and_metric_controls(widget) -> None:  # noqa: ANN001
    session = widget.import_file(FIXTURES / "trackman.csv")
    assert session.manifest.profile_id == "trackman"
    assert widget.session_tree.topLevelItemCount() == 1
    assert widget.data_table.rowCount() == 2
    assert widget.relationship_metrics.count() >= 8
    assert "2 shots" in widget.status_label.text()


def test_relationship_analysis_populates_matrix_and_scientific_warning(widget) -> None:  # noqa: ANN001
    widget.import_file(FIXTURES / "trackman.csv")
    widget.import_file(FIXTURES / "garmin.csv")
    for index in range(widget.relationship_metrics.count()):
        item = widget.relationship_metrics.item(index)
        if item.text() in {"club_speed", "ball_speed", "carry_distance"}:
            item.setSelected(True)
    result = widget.run_relationship_analysis()
    assert result.coefficients.shape == (3, 3)
    assert widget.relationship_table.rowCount() == 3
    assert "caus" in widget.scientific_boundary.text().lower()


def test_project_save_and_load_round_trip(widget, tmp_path) -> None:  # noqa: ANN001
    widget.import_file(FIXTURES / "uneekor.csv")
    destination = tmp_path / "player.lmproject"
    widget.save_project(destination)
    widget.clear_project()
    assert widget.data_table.rowCount() == 0
    widget.load_project(destination)
    assert widget.session_tree.topLevelItemCount() == 1
    assert widget.data_table.rowCount() == 2


def test_import_mapping_dialog_captures_direction_and_measurement_status(
    qapp,
) -> None:  # noqa: ANN001
    from PyQt6 import QtWidgets

    from src.tools.launch_monitor_analytics.widgets import ImportMappingDialog

    dialog = ImportMappingDialog(FIXTURES / "trackman.csv")
    try:
        row = dialog.headers.index("Club Speed (mph)")
        direction = dialog.mapping_table.cellWidget(row, 3)
        status = dialog.mapping_table.cellWidget(row, 4)
        assert isinstance(direction, QtWidgets.QComboBox)
        assert isinstance(status, QtWidgets.QComboBox)
        direction.setCurrentIndex(1)
        status.setCurrentText("measured")
        mapping = next(
            item
            for item in dialog.import_options().mappings
            if item.source_column == "Club Speed (mph)"
        )
        assert mapping.multiplier == -1.0
        assert mapping.measurement_status == "measured"
    finally:
        dialog.deleteLater()
