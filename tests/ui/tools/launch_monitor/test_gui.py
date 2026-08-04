from __future__ import annotations

from pathlib import Path

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
        "Models",
        "Monitor Comparison",
        "Dispersion",
        "Trends",
        "Reports",
    ]
    assert widget.windowTitle() == "Launch Monitor Analytics"


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
