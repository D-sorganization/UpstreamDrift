from __future__ import annotations

import hashlib
import json
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


def test_relationship_analysis_title_identifies_the_project(widget) -> None:  # noqa: ANN001
    widget.import_file(FIXTURES / "trackman.csv")
    widget.import_file(FIXTURES / "garmin.csv")
    for index in range(widget.relationship_metrics.count()):
        item = widget.relationship_metrics.item(index)
        if item.text() in {"club_speed", "ball_speed", "carry_distance"}:
            item.setSelected(True)
    widget.run_relationship_analysis()

    assert widget.project.name in widget.relationship_plot.axes.get_title()


def test_data_change_clears_every_stale_analysis_canvas(widget) -> None:  # noqa: ANN001
    """Regression test for #8825.

    ``_refresh_all`` used to rebuild trees/tables/combos but never touch
    the five matplotlib analysis canvases, so a chart built against a
    previous project/session set (including extra colorbar axes and
    plotted artists) stayed visible after ``clear_project``,
    ``import_file``, ``load_project``, ``_remove_selected_sessions``, or
    ``_run_treatment_ui`` changed the underlying data.
    """
    widget.import_file(FIXTURES / "trackman.csv")
    widget.import_file(FIXTURES / "garmin.csv")
    for index in range(widget.relationship_metrics.count()):
        item = widget.relationship_metrics.item(index)
        if item.text() in {"club_speed", "ball_speed", "carry_distance"}:
            item.setSelected(True)
    widget.run_relationship_analysis()

    canvases = (
        widget.relationship_plot,
        widget.model_plot,
        widget.comparison_plot,
        widget.dispersion_plot,
        widget.trend_plot,
    )
    # Sanity check the fixture actually populated the canvas: a colorbar
    # adds a second axes to the figure, and the heatmap is an image.
    assert len(widget.relationship_plot.figure.axes) > 1
    assert widget.relationship_plot.axes.images

    widget.clear_project()

    for canvas in canvases:
        assert len(canvas.figure.axes) == 1, (
            "stale colorbar axes survived a project switch"
        )
        assert not canvas.axes.images
        assert not canvas.axes.collections
        assert not canvas.axes.lines
        assert canvas.axes.texts, "canvas should show a placeholder after data changes"
        assert widget.project.name in canvas.axes.texts[0].get_text()


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


def _write_synthetic_corpus(root: Path) -> Path:
    """Write a two-source synthetic Parquet corpus under a checkout root."""
    pytest.importorskip("pyarrow")
    dataset = root / "data" / "authority" / "database" / "shot_corpus_parquet"
    rows = pd.DataFrame(
        {
            "monitor": ["TrackMan", "FlightScope Mevo+"],
            "file": ["a.csv", "b.csv"],
            "row_index": [0, 0],
            "club": ["Driver", "7 Iron"],
            "club_speed_mph": [100.0, 80.0],
            "ball_speed_mph": [150.0, 110.0],
            "smash_factor": [1.5, 1.375],
            "launch_angle_deg": [12.0, 18.0],
            "launch_direction_deg": [1.0, -0.5],
            "spin_rate_rpm": [2700.0, 6500.0],
            "back_spin_rpm": [2600.0, 6400.0],
            "side_spin_rpm": [300.0, -200.0],
            "spin_axis_deg": [4.0, -2.0],
            "attack_angle_deg": [-1.2, -4.0],
            "club_path_deg": [0.5, 1.5],
            "face_angle_deg": [0.2, 0.8],
            "carry_yd": [250.0, 165.0],
            "total_yd": [270.0, 172.0],
            "apex_native": [95.0, 28.0],
            "descent_angle_deg": [38.0, 45.0],
            "native_json": ["{}", "{}"],
        }
    )
    for source_id, group in (
        ("synthetic_trackman", rows.iloc[:1]),
        ("synthetic_mevo", rows.iloc[1:]),
    ):
        partition = dataset / f"source_id={source_id}"
        partition.mkdir(parents=True)
        group.to_parquet(partition / "part-0.parquet", index=False)
    (root / "data" / "authority" / "AUTHORITY_MANIFEST.json").write_text(
        "{}", encoding="utf-8"
    )
    return root


def test_load_private_corpus_adds_one_session_per_source(
    widget,  # noqa: ANN001
    tmp_path: Path,
) -> None:
    root = _write_synthetic_corpus(tmp_path / "checkout")

    added = widget.load_private_corpus_sessions(root)

    assert added == 2
    assert {session.session_id for session in widget.project.sessions} == {
        "synthetic_trackman",
        "synthetic_mevo",
    }
    assert len(widget.analysis_frame) == 2
    assert widget.is_dirty()
    manifests = {
        session.session_id: session.manifest for session in widget.project.sessions
    }
    assert manifests["synthetic_trackman"].profile_id == "private_corpus"
    assert manifests["synthetic_trackman"].vendor == "TrackMan"
    assert manifests["synthetic_trackman"].source_path == "corpus://synthetic_trackman"


def test_load_private_corpus_is_repeatable_without_duplicates(
    widget,  # noqa: ANN001
    tmp_path: Path,
) -> None:
    root = _write_synthetic_corpus(tmp_path / "checkout")

    assert widget.load_private_corpus_sessions(root) == 2
    assert widget.load_private_corpus_sessions(root) == 0
    assert len(widget.project.sessions) == 2


def test_load_private_corpus_fails_closed_without_authority(
    widget,  # noqa: ANN001
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        widget.load_private_corpus_sessions(tmp_path / "absent")
    assert widget.project.sessions == []


def test_sessions_tab_exposes_a_corpus_load_action(widget) -> None:  # noqa: ANN001
    assert widget.load_corpus_button.text() == "Load Private Corpus"
    assert "LAUNCH_MONITOR_DATA_ROOT" in widget.load_corpus_button.toolTip()


def test_window_file_menu_offers_the_corpus_load_action(qapp) -> None:  # noqa: ANN001
    from src.tools.launch_monitor_analytics.gui import LaunchMonitorAnalyticsWindow

    window = LaunchMonitorAnalyticsWindow()
    try:
        menu_bar = window.menuBar()
        file_menu = menu_bar.actions()[0].menu()
        labels = [action.text() for action in file_menu.actions()]
        assert "Load &Private Corpus" in labels
    finally:
        window.deleteLater()


def test_exported_data_and_manifest_share_a_verifiable_export_id(
    widget, tmp_path
) -> None:  # noqa: ANN001
    """The data export and the reproducibility manifest must stay linkable.

    Regression test for #8826: `_on_export_data` used to write a bare CSV
    with no header, and `_on_export_manifest` wrote a wholly separate JSON
    with no shared identifier, so the pair was unattributable once split
    on disk. `export_data` now stamps an export ID into the CSV itself and
    records the file's own SHA-256; `export_manifest` embeds both into a
    `data_export` block.
    """
    widget.import_file(FIXTURES / "trackman.csv")

    data_path = tmp_path / "launch_monitor_data.csv"
    manifest_path = tmp_path / "launch_monitor_manifest.json"
    widget.export_data(data_path)
    widget.export_manifest(manifest_path)

    data_bytes = data_path.read_bytes()
    first_line = data_bytes.split(b"\n", 1)[0].decode("utf-8")
    assert first_line.startswith("# export_id=")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_export = manifest["data_export"]
    assert data_export is not None
    assert data_export["export_id"] in first_line

    # The manifest's recorded hash must match the exported file's actual
    # on-disk content -- the verifiable link the issue asked for.
    assert data_export["data_sha256"] == hashlib.sha256(data_bytes).hexdigest()
    assert data_export["data_file"] == data_path.name


def test_export_data_parquet_embeds_export_id_in_schema_metadata(
    widget, tmp_path
) -> None:  # noqa: ANN001
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    widget.import_file(FIXTURES / "trackman.csv")
    data_path = tmp_path / "launch_monitor_data.parquet"
    manifest_path = tmp_path / "launch_monitor_manifest.json"

    widget.export_data(data_path, as_parquet=True)
    widget.export_manifest(manifest_path)

    schema_metadata = pq.read_schema(data_path).metadata
    embedded_export_id = schema_metadata[b"export_id"].decode("utf-8")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["data_export"]["export_id"] == embedded_export_id
    assert (
        manifest["data_export"]["data_sha256"]
        == hashlib.sha256(data_path.read_bytes()).hexdigest()
    )


def test_export_manifest_without_a_prior_data_export_is_still_valid(
    widget, tmp_path
) -> None:  # noqa: ANN001
    manifest_path = tmp_path / "launch_monitor_manifest.json"
    widget.export_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["data_export"] is None
