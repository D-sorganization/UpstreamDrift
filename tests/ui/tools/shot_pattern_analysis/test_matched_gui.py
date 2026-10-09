"""Offscreen controls and lifecycle contracts for the matched study mode."""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.fixture
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        from PyQt6.QtWidgets import QApplication
    except (ImportError, OSError) as exc:
        pytest.skip(f"PyQt6 not loadable: {exc}")
    return QApplication.instance() or QApplication([])


def _study(qapp):  # noqa: ANN001
    from src.tools.shot_pattern_analysis.matched_gui import MatchedStudyWidget

    return MatchedStudyWidget()


def test_legacy_widget_exposes_a_separate_matched_delivery_mode(qapp) -> None:  # noqa: ANN001
    from src.tools.shot_pattern_analysis.gui import MainWidget

    widget = MainWidget()
    assert widget.mode_tabs.count() == 2
    assert widget.mode_tabs.tabText(0) == "Independent Pattern Comparison"
    assert widget.mode_tabs.tabText(1) == "Matched Delivery Study"
    assert widget.matched_study is widget.mode_tabs.widget(1)
    assert widget.cleanup()


def test_matched_controls_cover_seven_sd_inputs_units_and_nominal_loft(qapp) -> None:  # noqa: ANN001
    widget = _study(qapp)

    assert len(widget.noise_inputs) == 7
    assert widget.club_speed_mph.decimals() == 1
    assert widget.noise_inputs["speed_mps"].decimals() == 2
    assert widget.noise_inputs["contact_horizontal_m"].value() == pytest.approx(0.25)
    assert widget.noise_inputs["contact_vertical_m"].value() == pytest.approx(0.25)
    assert widget.noise_inputs["path_deg"].value() == pytest.approx(0.5)
    assert widget.noise_inputs["shaft_rotation_deg"].value() == pytest.approx(1.2)
    assert "Static Loft" in widget.static_loft_label.text()
    assert "illustrative" in widget.assumption_label.text().lower()
    assert widget.cleanup()


def test_modified_preset_keeps_club_scoring_context_and_custom_is_explicit(
    qapp,
) -> None:  # noqa: ANN001
    widget = _study(qapp)
    widget.club_choice.setCurrentIndex(widget.club_choice.findData("seven_iron"))
    widget.shaft_lean.setValue(5.0)
    modified = widget.build_experiment()
    assert modified.club.club_id == "seven_iron"
    assert "modified" in widget.assumption_label.text().lower()

    widget.club_choice.setCurrentIndex(widget.club_choice.findData("custom"))
    custom = widget.build_experiment()
    assert custom.club.club_id == "custom"
    assert custom.club.shaft_lean_deg == pytest.approx(5.0)
    assert widget.cleanup()


def test_contact_law_selector_records_the_selected_model_and_limitations(qapp) -> None:  # noqa: ANN001
    widget = _study(qapp)
    widget.contact_law.setCurrentIndex(
        widget.contact_law.findData("maximum_dissipation_newton")
    )

    request = widget.build_experiment()

    assert request.club.contact_law == "maximum_dissipation_newton"
    assert "maximum-dissipation" in widget.contact_law.currentText().lower()
    assert "neither has been validated" in widget.contact_law_help.text().lower()
    assert widget.cleanup()


def test_inputs_convert_units_and_approximate_face_path_rho_sign(qapp) -> None:  # noqa: ANN001
    widget = _study(qapp)
    widget.noise_inputs["shaft_rotation_deg"].setValue(1.7)
    widget.noise_inputs["path_deg"].setValue(0.6)
    widget.noise_inputs["speed_mps"].setValue(0.43)
    widget.noise_inputs["contact_horizontal_m"].setValue(0.3)
    widget.noise_inputs["contact_vertical_m"].setValue(0.2)
    widget.noise_inputs["shaft_lean_deg"].setValue(0.4)
    widget.noise_inputs["attack_angle_deg"].setValue(0.25)
    widget.set_correlation("shaft_rotation_deg", "path_deg", 0.35)
    request = widget.build_experiment()

    assert request.noise.standard_deviations[2] == pytest.approx(0.43 * 0.44704)
    assert request.noise.standard_deviations[3] == pytest.approx(0.3 * 0.0254)
    assert request.noise.standard_deviations[4] == pytest.approx(0.2 * 0.0254)
    assert request.noise.correlation_matrix()[0, 1] == pytest.approx(0.35)
    assert request.noise.correlation_matrix()[1, 0] == pytest.approx(0.35)

    widget.face_path_rho.setValue(0.4)
    widget.apply_approximate_face_path_correlation()
    assert widget.correlation_value("shaft_rotation_deg", "path_deg") == pytest.approx(
        -0.4
    )
    assert "approximate" in widget.correlation_help.text().lower()
    assert widget.cleanup()


def test_invalid_non_psd_correlation_is_rejected_without_repair(qapp) -> None:  # noqa: ANN001
    widget = _study(qapp)
    widget.set_correlation("shaft_rotation_deg", "path_deg", 0.9)
    widget.set_correlation("shaft_rotation_deg", "speed_mps", 0.9)
    widget.set_correlation("path_deg", "speed_mps", -0.9)
    before = widget.correlation_matrix().copy()

    with pytest.raises(ValueError, match="positive semidefinite"):
        widget.build_experiment()
    assert (widget.correlation_matrix() == before).all()
    assert widget.cleanup()


def test_complete_config_json_round_trip_preserves_advanced_club_inputs(
    qapp, tmp_path: Path
) -> None:  # noqa: ANN001
    widget = _study(qapp)
    request_type = type(widget.build_experiment())
    request = request_type(
        club=replace(
            widget.club_config,
            club_id="seven_iron",
            club_speed_mps=34.876,
            loft_deg=33.7,
            cg_depth_m=0.0123,
            inertia_diagonal_kg_m2=(0.00013, 0.00021, 0.00027),
        ),
        noise=widget.build_experiment().noise,
        n_shots=128,
        seed=90210,
        curve_scale=1.2,
        target_x_m=180.0,
        match_tolerance_m=0.03,
        target_radius_m=12.0,
    )
    path = tmp_path / "study.json"
    widget.export_config(path, request)
    widget.load_config(path)
    round_trip = widget.build_experiment()

    assert round_trip == request
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["schema"] == "shot-pattern-matched-config/1"
    assert saved["club"]["inertia_diagonal_kg_m2"] == [0.00013, 0.00021, 0.00027]
    assert widget.cleanup()


def test_invalid_correlation_prevents_worker_start(qapp, tmp_path, monkeypatch) -> None:  # noqa: ANN001
    widget = _study(qapp)
    widget.output_edit.setText(str(tmp_path))
    widget.set_correlation("shaft_rotation_deg", "path_deg", 0.9)
    widget.set_correlation("shaft_rotation_deg", "speed_mps", 0.9)
    widget.set_correlation("path_deg", "speed_mps", -0.9)
    started = []
    monkeypatch.setattr(
        widget.action_bar, "start", lambda *args, **kwargs: started.append(args)
    )

    widget.run_analysis()

    assert not started
    assert "positive semidefinite" in widget.status_label.text().lower()
    assert widget.cleanup()


def test_unavailable_nominal_report_does_not_show_density_or_fake_shots(
    qapp, tmp_path: Path
) -> None:  # noqa: ANN001
    widget = _study(qapp)
    (tmp_path / "summary.json").write_text(
        json.dumps({"status": "unavailable", "reason": "nominal_not_supported"}),
        encoding="utf-8",
    )
    widget._completed({"output_dir": tmp_path})

    assert "unavailable" in widget.status_label.text().lower()
    assert "nominal_not_supported" in widget.result_label.text()
    assert widget.preview.property("sourcePixmap") is None
    assert widget.cleanup()


def test_completed_report_shows_solved_nominal_delivery_and_target_protocol(
    qapp, tmp_path: Path
) -> None:  # noqa: ANN001
    from dataclasses import asdict

    from src.tools.shot_pattern_analysis.matched_config import matched_preset

    widget = _study(qapp)
    nominal = {
        "face_deg": 1.25,
        "path_deg": -1.75,
        "speed_mps": 32.0,
        "residual_m": 0.012,
    }
    payload = {
        "status": "completed",
        "config": {"club": asdict(matched_preset("seven_iron")), "target_x_m": None},
        "target_x_m": 180.0,
        "nominals": dict.fromkeys(("Straight", "Draw", "Fade"), nominal),
        "summary": {
            name: {
                "total_n": 8,
                "supported_n": 8,
                "unsupported_reasons": {},
                "realized_face_path_correlation": -0.42,
            }
            for name in ("Straight", "Draw", "Fade")
        },
    }
    (tmp_path / "summary.json").write_text(json.dumps(payload), encoding="utf-8")

    widget._completed({"output_dir": tmp_path})

    rendered = widget.result_label.text()
    assert "180.0 m (matched within this lean setting)" in rendered
    assert "+1.25°/-1.75°" in rendered
    assert "71.6 mph" in rendered
    assert "° loft; 0.012 m residual; 8/8 supported" in rendered
    assert "empirical face/path ρ=-0.420" in rendered
    assert widget.cleanup()


def test_completed_report_handles_zero_spin_with_undefined_tilt(qapp, tmp_path) -> None:  # noqa: ANN001
    from dataclasses import asdict

    from src.tools.shot_pattern_analysis.matched_config import matched_preset

    widget = _study(qapp)
    nominal = {
        "face_deg": 0.0,
        "path_deg": 0.0,
        "speed_mps": 32.0,
        "residual_m": 0.0,
        "outcome": {"spin_rpm": 0.0, "spin_axis_tilt_deg": None},
    }
    payload = {
        "status": "completed",
        "config": {"club": asdict(matched_preset("seven_iron")), "target_x_m": None},
        "target_x_m": 180.0,
        "nominals": dict.fromkeys(("Straight", "Draw", "Fade"), nominal),
        "summary": {
            name: {
                "total_n": 1,
                "supported_n": 1,
                "unsupported_reasons": {},
                "realized_face_path_correlation": None,
            }
            for name in ("Straight", "Draw", "Fade")
        },
    }
    (tmp_path / "summary.json").write_text(json.dumps(payload), encoding="utf-8")

    widget._completed({"output_dir": tmp_path})

    assert "spin_axis_tilt_deg" not in widget.result_label.text()
    assert "empirical face/path ρ=unavailable" in widget.result_label.text()
    assert widget.cleanup()


def test_cancellation_during_runner_never_starts_export(
    qapp, tmp_path, monkeypatch
) -> None:  # noqa: ANN001
    from src.tools.shot_pattern_analysis.core import AnalysisCancelled
    import src.tools.shot_pattern_analysis.matched_gui as matched_gui

    widget = _study(qapp)
    widget.output_edit.setText(str(tmp_path))

    def fake_execute(request, output_dir, *, progress, cancelled, **kwargs):  # noqa: ANN001
        while True:
            if cancelled():
                raise AnalysisCancelled("matched shot analysis cancelled")
            progress(0, 3 * request.n_shots)
            time.sleep(0.01)

    monkeypatch.setattr(matched_gui, "_execute_and_export", fake_execute)
    widget.run_analysis()
    deadline = time.monotonic() + 3.0
    while not widget.action_bar.is_running and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert widget.action_bar.is_running
    widget.action_bar.cancel()
    deadline = time.monotonic() + 5.0
    while widget.action_bar.is_running and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()

    assert not widget.action_bar.is_running
    assert not list(tmp_path.iterdir())
    assert "cancelled" in widget.status_label.text().lower()
    assert widget.cleanup()


def test_cleanup_joins_an_active_matched_worker(qapp, tmp_path, monkeypatch) -> None:  # noqa: ANN001
    from src.tools.shot_pattern_analysis.core import AnalysisCancelled
    import src.tools.shot_pattern_analysis.matched_gui as matched_gui

    widget = _study(qapp)
    widget.output_edit.setText(str(tmp_path))

    def cancellable_execute(request, output_dir, *, progress, cancelled, **kwargs):  # noqa: ANN001
        while not cancelled():
            progress(0, 3 * request.n_shots)
            time.sleep(0.01)
        raise AnalysisCancelled("matched shot analysis cancelled")

    monkeypatch.setattr(matched_gui, "_execute_and_export", cancellable_execute)
    widget.run_analysis()
    deadline = time.monotonic() + 3.0
    while not widget.action_bar.is_running and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)

    assert widget.action_bar.is_running
    assert widget.cleanup()
    assert not widget.action_bar.is_running
    assert not list(tmp_path.iterdir())


def test_matched_delivery_controls_fit_launcher_height_with_scroll(qapp) -> None:  # noqa: ANN001
    from PyQt6 import QtCore, QtWidgets

    widget = _study(qapp)
    try:
        widget.resize(1000, 760)
        widget.show()
        qapp.processEvents()
        assert widget.height() <= 760
        assert widget.rect().contains(
            widget.run_button.mapTo(widget, QtCore.QPoint(0, 0))
        )
        scroll = widget.findChild(QtWidgets.QScrollArea, "matchedControlsScroll")
        assert scroll is not None
        assert scroll.widgetResizable()
        assert scroll.verticalScrollBar().maximum() > 0
        scroll.ensureWidgetVisible(widget.correlation_table)
        qapp.processEvents()
        origin = widget.correlation_table.mapTo(scroll.viewport(), QtCore.QPoint(0, 0))
        assert (
            scroll.viewport()
            .rect()
            .intersects(QtCore.QRect(origin, widget.correlation_table.size()))
        )
    finally:
        widget.cleanup()
        widget.close()


def test_launcher_matched_tab_respects_embedded_minimum_height(qapp) -> None:  # noqa: ANN001
    from src.tools.shot_pattern_analysis.gui import MainWidget

    widget = MainWidget()
    try:
        widget.mode_tabs.setCurrentIndex(1)
        widget.resize(1000, 760)
        widget.show()
        qapp.processEvents()
        assert widget.height() <= 760
    finally:
        widget.cleanup()
        widget.close()


def test_correlation_headers_are_readable_and_identify_export_fields(qapp) -> None:  # noqa: ANN001
    from src.tools.shot_pattern_analysis.noise import NOISE_FIELDS

    widget = _study(qapp)
    try:
        widget.resize(1000, 760)
        widget.show()
        qapp.processEvents()
        table = widget.correlation_table
        metrics = table.horizontalHeader().fontMetrics()
        for column, field in enumerate(NOISE_FIELDS):
            header = table.horizontalHeaderItem(column)
            assert field in header.toolTip()
            assert (
                table.columnWidth(column)
                >= metrics.horizontalAdvance(header.text()) + 8
            )
    finally:
        widget.cleanup()
        widget.close()
