"""Offscreen controls and lifecycle contracts for the matched study mode."""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

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


def test_nominal_bounds_are_visible_editable_and_preserved_across_config_json(
    qapp, tmp_path: Path
) -> None:  # noqa: ANN001
    widget = _study(qapp)
    request = replace(
        widget.build_experiment(),
        match_face_bounds_deg=(-6.0, 6.0),
        match_speed_factors=(0.75, 1.30),
    )
    path = tmp_path / "supplement.json"
    widget.export_config(path, request)
    widget.load_config(path)

    assert widget.use_absolute_face_bounds.isChecked()
    assert widget.face_bound_min.value() == pytest.approx(-6.0)
    assert widget.face_bound_max.value() == pytest.approx(6.0)
    assert widget.speed_factor_min.value() == pytest.approx(0.75)
    assert widget.speed_factor_max.value() == pytest.approx(1.30)
    assert widget.build_experiment() == request

    widget.face_bound_min.setValue(-5.5)
    widget.speed_factor_max.setValue(1.25)
    edited = widget.build_experiment()
    assert edited.match_face_bounds_deg == (-5.5, 6.0)
    assert edited.match_speed_factors == (0.75, 1.25)
    assert widget.cleanup()


def test_legacy_config_without_nominal_bounds_uses_visible_defaults(
    qapp, tmp_path: Path
) -> None:  # noqa: ANN001
    widget = _study(qapp)
    path = tmp_path / "legacy.json"
    widget.export_config(path)
    payload = json.loads(path.read_text())
    del payload["match_face_bounds_deg"]
    del payload["match_speed_factors"]
    path.write_text(json.dumps(payload))

    widget.load_config(path)

    assert not widget.use_absolute_face_bounds.isChecked()
    assert widget.speed_factor_min.value() == pytest.approx(0.75)
    assert widget.speed_factor_max.value() == pytest.approx(1.25)
    request = widget.build_experiment()
    assert request.match_face_bounds_deg is None
    assert request.match_speed_factors == (0.75, 1.25)
    assert widget.cleanup()


def test_imported_valid_bounds_are_not_silently_clamped(qapp, tmp_path: Path) -> None:  # noqa: ANN001
    widget = _study(qapp)
    request = replace(
        widget.build_experiment(),
        match_face_bounds_deg=(-14.995, 14.995),
        match_speed_factors=(0.75, 12.0),
    )
    path = tmp_path / "wide.json"
    widget.export_config(path, request)
    widget.load_config(path)
    assert widget.build_experiment() == request
    assert widget.cleanup()


def test_imported_bound_beyond_control_precision_fails_explicitly(
    qapp, tmp_path: Path
) -> None:  # noqa: ANN001
    widget = _study(qapp)
    request = replace(
        widget.build_experiment(), match_speed_factors=(0.123456789, 1.25)
    )
    path = tmp_path / "precision.json"
    widget.export_config(path, request)
    with pytest.raises(ValueError, match="cannot display exactly"):
        widget.load_config(path)
    assert widget.cleanup()


def test_exact_supplement_config_preserves_target_and_implicit_identity_noise(
    qapp, tmp_path: Path
) -> None:  # noqa: ANN001
    from src.tools.shot_pattern_analysis.matched_matrix import matched_matrix_cells

    widget = _study(qapp)
    _, request = matched_matrix_cells(
        n_shots=32,
        common_targets={"driver": 178.94875782337786},
        profiles=("combined2",),
        selected_cell_names=("driver_lean5_curve2_rotation2_combined2",),
        match_face_bounds_deg=(-6.0, 6.0),
        match_speed_factors=(0.75, 1.30),
    )[0]
    path = tmp_path / "driver-supplement.json"
    widget.export_config(path, request)
    widget.load_config(path)

    assert widget.build_experiment() == request
    assert widget.noise_inputs["shaft_rotation_deg"].value() == pytest.approx(2.0)
    assert widget.target_distance.value() == request.target_x_m
    assert widget.cleanup()


def test_import_precision_rejection_leaves_existing_gui_request_unchanged(
    qapp, tmp_path: Path
) -> None:  # noqa: ANN001
    widget = _study(qapp)
    original = widget.build_experiment()
    request = replace(
        original,
        target_x_m=178.94875782337786,
        noise=replace(
            original.noise,
            standard_deviations=(1.234567, *original.noise.standard_deviations[1:]),
        ),
    )
    path = tmp_path / "precision.json"
    widget.export_config(path, request)

    with pytest.raises(ValueError, match="cannot display exactly"):
        widget.load_config(path)

    assert widget.build_experiment() == original
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


def _write_completed_study_summary(
    output_dir: Path,
    *,
    course_scoring: Mapping[str, Any] | None = None,
    club_id: str = "seven_iron",
    target_x_m: float = 180.0,
    config: Mapping[str, Any] | None = None,
    nominals: Mapping[str, Any] | None = None,
    nominal: Mapping[str, Any] | None = None,
    summary: Mapping[str, Any] | None = None,
) -> Path:
    from dataclasses import asdict

    from src.tools.shot_pattern_analysis.matched_config import matched_preset

    if config is None:
        preset = (
            matched_preset(club_id)
            if club_id in ("driver", "seven_iron", "pitching_wedge")
            else None
        )
        club_payload = asdict(preset) if preset is not None else {"club_id": club_id}
        config = {
            "club": club_payload,
            "target_x_m": target_x_m if club_id == "driver" else None,
        }

    if nominals is None:
        if nominal is None:
            nominal = {
                "face_deg": 1.0,
                "path_deg": -1.0,
                "speed_mps": 32.0,
                "residual_m": 0.01,
            }
        nominals = dict.fromkeys(("Straight", "Draw", "Fade"), nominal)

    if summary is None:
        summary = {
            name: {"total_n": 8, "supported_n": 8, "unsupported_reasons": {}}
            for name in ("Straight", "Draw", "Fade")
        }

    payload: dict[str, Any] = {
        "status": "completed",
        "config": config,
        "target_x_m": target_x_m,
        "nominals": nominals,
        "summary": summary,
    }
    if course_scoring is not None:
        payload["course_scoring"] = course_scoring

    summary_file = output_dir / "summary.json"
    summary_file.write_text(json.dumps(payload), encoding="utf-8")
    return summary_file


def test_completed_displays_saved_course_scoring_strokes_gained(
    qapp, tmp_path: Path
) -> None:  # noqa: ANN001
    widget = _study(qapp)
    summary = {
        name: {
            "total_n": 8,
            "supported_n": 8,
            "unsupported_reasons": {},
            "realized_face_path_correlation": -0.42,
        }
        for name in ("Straight", "Draw", "Fade")
    }
    course_scoring = {
        "status": "available",
        "scenario_id": "approach_green_rough",
        "scenario": "Historical PGA approach to a centered circular green with rough outside",
        "club_id": "seven_iron",
        "start_lie": "fairway",
        "start_distance_m": 180.0,
        "target_x_m": 180.0,
        "green_radius_m": 15.0,
        "paired_n": 8,
        "total_n": 8,
        "conditioning": "all_samples",
        "full_population_status": "available_under_hypothetical_course_assumptions",
        "patterns": {
            "Straight": {
                "n": 8,
                "mean_strokes_gained": 0.250,
                "green_fraction": 0.75,
            },
            "Draw": {"n": 8, "mean_strokes_gained": 0.310, "green_fraction": 0.875},
            "Fade": {"n": 8, "mean_strokes_gained": 0.210, "green_fraction": 0.625},
        },
        "paired_benefit_vs_straight": {
            "Draw": {"estimate": 0.060, "lower_95": 0.015, "upper_95": 0.105},
            "Fade": {"estimate": -0.040, "lower_95": -0.085, "upper_95": 0.005},
        },
        "limitations": [
            "2003-2010 PGA benchmark and approximate reconciled putting fit; not a current or player-specific estimate.",
            "Flat hypothetical geometry; landing is treated as final lie with no roll, hazards, or recovery obstruction.",
            "Bootstrap intervals include Monte Carlo sampling only, not club delivery, course, or model uncertainty.",
        ],
    }
    _write_completed_study_summary(
        tmp_path,
        target_x_m=180.0,
        summary=summary,
        course_scoring=course_scoring,
    )

    widget._completed({"output_dir": tmp_path})

    # Individual supported counts preserved in result_label
    assert "8/8 supported" in widget.result_label.text()

    # Course scoring display exists and displays strokes gained
    assert hasattr(widget, "scoring_label"), (
        "MatchedStudyWidget must have scoring_label"
    )
    scoring_text = widget.scoring_label.text()
    assert "Historical PGA approach to a centered circular green" in scoring_text
    assert "180.0 m" in scoring_text
    assert "15" in scoring_text  # green radius
    assert "8/8" in scoring_text  # common supported paired count
    assert "Straight: +0.250" in scoring_text
    assert "Draw: +0.310" in scoring_text
    assert "Fade: +0.210" in scoring_text
    assert "Paired Difference vs. Straight (95% Monte Carlo Interval)" in scoring_text
    assert "Draw: +0.060 [+0.015, +0.105] strokes" in scoring_text
    assert "Fade: -0.040 [-0.085, +0.005] strokes" in scoring_text
    assert "carry-only" in scoring_text.lower() or "carry" in scoring_text.lower()
    assert "approximate" in scoring_text.lower() or "benchmark" in scoring_text.lower()
    assert widget.cleanup()


def test_completed_displays_saved_driver_tee_course_scoring(
    qapp, tmp_path: Path
) -> None:  # noqa: ANN001
    widget = _study(qapp)
    summary = {
        name: {
            "total_n": 10,
            "supported_n": 10,
            "unsupported_reasons": {},
            "realized_face_path_correlation": -0.35,
        }
        for name in ("Straight", "Draw", "Fade")
    }
    course_scoring = {
        "status": "available",
        "scenario_id": "tee_fairway_rough",
        "scenario": "Historical PGA tee shot on a flat 400 m hole with a 30 m fairway corridor",
        "club_id": "driver",
        "start_lie": "tee",
        "start_distance_m": 400.0,
        "target_x_m": 250.0,
        "hole_distance_m": 400.0,
        "fairway_half_width_m": 15.0,
        "paired_n": 10,
        "total_n": 10,
        "conditioning": "all_samples",
        "full_population_status": "available_under_hypothetical_course_assumptions",
        "patterns": {
            "Straight": {
                "n": 10,
                "mean_strokes_gained": 0.400,
                "fairway_fraction": 0.70,
            },
            "Draw": {
                "n": 10,
                "mean_strokes_gained": 0.450,
                "fairway_fraction": 0.80,
            },
            "Fade": {
                "n": 10,
                "mean_strokes_gained": 0.380,
                "fairway_fraction": 0.60,
            },
        },
        "paired_benefit_vs_straight": {
            "Draw": {"estimate": 0.050, "lower_95": 0.010, "upper_95": 0.090},
            "Fade": {"estimate": -0.020, "lower_95": -0.060, "upper_95": 0.020},
        },
        "limitations": [
            "2003-2010 PGA benchmark and approximate reconciled putting fit; not a current or player-specific estimate.",
            "Flat hypothetical geometry; landing is treated as final lie with no roll, hazards, or recovery obstruction.",
            "Bootstrap intervals include Monte Carlo sampling only, not club delivery, course, or model uncertainty.",
        ],
    }
    _write_completed_study_summary(
        tmp_path,
        club_id="driver",
        target_x_m=250.0,
        summary=summary,
        course_scoring=course_scoring,
    )

    widget._completed({"output_dir": tmp_path})

    scoring_text = widget.scoring_label.text()
    assert "tee shot" in scoring_text.lower() or "tee" in scoring_text.lower()
    assert "400" in scoring_text  # hole distance
    assert "30" in scoring_text or "15" in scoring_text  # fairway corridor width
    assert "Straight: +0.400" in scoring_text
    assert "Draw: +0.450" in scoring_text
    assert "Fade: +0.380" in scoring_text
    assert "Paired Difference vs. Straight (95% Monte Carlo Interval)" in scoring_text
    assert "Draw: +0.050 [+0.010, +0.090] strokes" in scoring_text
    assert "Fade: -0.020 [-0.060, +0.020] strokes" in scoring_text
    assert "Common supported pairs: 10/10" in scoring_text or "10/10" in scoring_text
    assert widget.cleanup()


def test_completed_handles_conditional_only_scoring(qapp, tmp_path: Path) -> None:  # noqa: ANN001
    widget = _study(qapp)
    summary = {
        "Straight": {"total_n": 10, "supported_n": 10, "unsupported_reasons": {}},
        "Draw": {
            "total_n": 10,
            "supported_n": 7,
            "unsupported_reasons": {"ground": 3},
        },
        "Fade": {
            "total_n": 10,
            "supported_n": 9,
            "unsupported_reasons": {"loft": 1},
        },
    }
    course_scoring = {
        "status": "conditional_only",
        "scenario_id": "approach_green_rough",
        "scenario": "Historical PGA approach to a centered circular green with rough outside",
        "club_id": "seven_iron",
        "start_lie": "fairway",
        "start_distance_m": 175.0,
        "target_x_m": 175.0,
        "green_radius_m": 15.0,
        "paired_n": 7,
        "total_n": 10,
        "conditioning": "common_supported_indices_only",
        "full_population_status": "unavailable_unsupported_terminal_lies",
        "patterns": {
            "Straight": {"n": 7, "mean_strokes_gained": 0.200},
            "Draw": {"n": 7, "mean_strokes_gained": 0.280},
            "Fade": {"n": 7, "mean_strokes_gained": 0.190},
        },
        "paired_benefit_vs_straight": {
            "Draw": {"estimate": 0.080, "lower_95": 0.020, "upper_95": 0.140},
            "Fade": {"estimate": -0.010, "lower_95": -0.060, "upper_95": 0.040},
        },
    }
    _write_completed_study_summary(
        tmp_path,
        target_x_m=175.0,
        summary=summary,
        course_scoring=course_scoring,
    )

    widget._completed({"output_dir": tmp_path})

    # Individual supported counts preserved in result_label
    result_text = widget.result_label.text()
    assert "Straight: " in result_text and "10/10 supported" in result_text
    assert "Draw: " in result_text and "7/10 supported" in result_text
    assert "Fade: " in result_text and "9/10 supported" in result_text

    # Scoring label explicitly states conditional status and denominator
    scoring_text = widget.scoring_label.text()
    assert "conditional" in scoring_text.lower()
    assert "7/10" in scoring_text  # common supported pairs
    assert "Straight: +0.200" in scoring_text
    assert "Draw: +0.280" in scoring_text
    assert "Fade: +0.190" in scoring_text
    assert widget.cleanup()


def test_completed_handles_unavailable_course_scoring(qapp, tmp_path: Path) -> None:  # noqa: ANN001
    widget = _study(qapp)
    summary = {
        name: {"total_n": 10, "supported_n": 1, "unsupported_reasons": {}}
        for name in ("Straight", "Draw", "Fade")
    }
    course_scoring = {
        "status": "unavailable",
        "reason": "fewer_than_two_common_supported_shots",
        "paired_n": 1,
        "total_n": 10,
    }
    _write_completed_study_summary(
        tmp_path,
        target_x_m=175.0,
        summary=summary,
        course_scoring=course_scoring,
    )

    widget._completed({"output_dir": tmp_path})

    scoring_text = widget.scoring_label.text()
    assert "unavailable" in scoring_text.lower()
    assert (
        "fewer_than_two_common_supported_shots" in scoring_text
        or "insufficient" in scoring_text.lower()
    )
    # Does not fabricate zeros or false claims
    assert "+0.000" not in scoring_text
    assert widget.cleanup()


def test_completed_handles_custom_club_no_context(qapp, tmp_path: Path) -> None:  # noqa: ANN001
    from dataclasses import asdict

    from src.tools.shot_pattern_analysis.matched_config import matched_preset

    widget = _study(qapp)
    club = asdict(matched_preset("seven_iron"))
    club["club_id"] = "custom"
    _write_completed_study_summary(
        tmp_path,
        config={"club": club, "target_x_m": 160.0},
        target_x_m=160.0,
        summary={
            name: {"total_n": 5, "supported_n": 5, "unsupported_reasons": {}}
            for name in ("Straight", "Draw", "Fade")
        },
        course_scoring={
            "status": "unavailable",
            "reason": "club_id must be a named illustrative preset",
        },
    )

    widget._completed({"output_dir": tmp_path})

    scoring_text = widget.scoring_label.text()
    assert "unavailable" in scoring_text.lower()
    assert (
        "custom" in scoring_text.lower()
        or "preset" in scoring_text.lower()
        or "baseline" in scoring_text.lower()
    )
    assert widget.cleanup()


def test_matched_study_with_scoring_label_respects_embedded_height(
    qapp, tmp_path: Path
) -> None:  # noqa: ANN001
    from src.tools.shot_pattern_analysis.gui import MainWidget

    main = MainWidget()
    try:
        main.mode_tabs.setCurrentIndex(1)
        matched = main.matched_study

        # Populate with completed scoring text
        course_scoring = {
            "status": "available",
            "scenario": "Historical PGA approach to a centered circular green with rough outside",
            "club_id": "seven_iron",
            "green_radius_m": 15.0,
            "paired_n": 8,
            "total_n": 8,
            "conditioning": "all_samples",
            "patterns": {
                "Straight": {"mean_strokes_gained": 0.25},
                "Draw": {"mean_strokes_gained": 0.31},
                "Fade": {"mean_strokes_gained": 0.21},
            },
            "paired_benefit_vs_straight": {
                "Draw": {"estimate": 0.06, "lower_95": 0.015, "upper_95": 0.105},
                "Fade": {"estimate": -0.04, "lower_95": -0.085, "upper_95": 0.005},
            },
        }
        _write_completed_study_summary(tmp_path, course_scoring=course_scoring)
        matched._completed({"output_dir": tmp_path})

        main.resize(1000, 760)
        main.show()
        qapp.processEvents()
        assert main.height() <= 760
    finally:
        main.cleanup()
        main.close()


def test_format_matched_course_scoring_pure_helper_edge_cases() -> None:
    from src.tools.shot_pattern_analysis.matched_gui import (
        format_matched_course_scoring,
    )

    # None handling
    assert "not available" in format_matched_course_scoring(None).lower()

    # Invalid type handling
    assert "invalid" in format_matched_course_scoring("invalid").lower()  # type: ignore[arg-type]

    # Insufficient pairs
    res_insuf = format_matched_course_scoring(
        {
            "status": "unavailable",
            "reason": "fewer_than_two_common_supported_shots",
            "paired_n": 1,
            "total_n": 100,
        }
    )
    assert "insufficient common supported shots" in res_insuf.lower()
    assert "1/100" in res_insuf
    assert "minimum 2" in res_insuf

    # Custom club
    res_custom = format_matched_course_scoring(
        {
            "status": "unavailable",
            "reason": "club_id must be a named illustrative preset",
        },
        club_id="custom",
    )
    assert "custom club" in res_custom.lower()
    assert "published baseline" in res_custom.lower()

    # Missing / non-finite values never turned into zeros
    res_nan = format_matched_course_scoring(
        {
            "status": "available",
            "paired_n": 10,
            "total_n": 10,
            "patterns": {
                "Straight": {"mean_strokes_gained": float("nan")},
                "Draw": {},
                "Fade": {"mean_strokes_gained": float("inf")},
            },
            "paired_benefit_vs_straight": {
                "Draw": {"estimate": float("nan"), "lower_95": 0.0, "upper_95": 0.1},
                "Fade": {"estimate": None, "lower_95": None, "upper_95": None},
            },
        }
    )
    assert "+0.000" not in res_nan
    assert "Straight: unavailable" in res_nan
    assert "Draw: unavailable" in res_nan
    assert "Fade: unavailable" in res_nan

    # Small mean strokes gained uses adaptive scientific notation
    res_small_mean = format_matched_course_scoring(
        {
            "status": "available",
            "paired_n": 10,
            "total_n": 10,
            "patterns": {
                "Straight": {"mean_strokes_gained": 7.46156e-5},
                "Draw": {"mean_strokes_gained": 0.0},
                "Fade": {"mean_strokes_gained": -7.10405e-5},
            },
        }
    )
    assert "+7.462e-05" in res_small_mean
    assert "+0.000" in res_small_mean
    assert "-7.104e-05" in res_small_mean


@pytest.mark.parametrize("scale", [1.0, 0.001])
def test_saved_small_sg_intervals_remain_numerically_visible(qapp, scale) -> None:  # noqa: ANN001
    """Display must preserve the sign and width of small simulated differences."""
    import re

    from src.tools.shot_pattern_analysis.matched_gui import (
        format_matched_course_scoring,
    )

    estimate, low, high = (
        scale * x for x in (0.0000746156, -0.0000710405, 0.000218961)
    )
    text = format_matched_course_scoring(
        {
            "status": "available",
            "club_id": "driver",
            "paired_n": 10000,
            "total_n": 10000,
            "paired_benefit_vs_straight": {
                "Draw": {"estimate": estimate, "lower_95": low, "upper_95": high}
            },
        }
    )
    match = re.search(r"Draw: ([^ ]+) \[([^,]+), ([^\]]+)\] strokes", text)
    assert match is not None
    shown = tuple(float(value) for value in match.groups())
    assert shown == pytest.approx((estimate, low, high), rel=0.01, abs=0.0)
    assert shown[1] < 0 < shown[2]


@pytest.mark.parametrize("event", ["run", "failure", "cancel"])
def test_new_study_lifecycle_does_not_present_previous_sg_as_current(
    qapp, tmp_path: Path, monkeypatch, event: str
) -> None:  # noqa: ANN001
    widget = _study(qapp)
    old_score = "Mean Strokes Gained: Previous Study Draw +0.250"
    widget.scoring_label.setText(old_score)
    started = []

    def capture_start(*args, **kwargs):
        started.append(widget.scoring_label.text())

    monkeypatch.setattr(widget.action_bar, "start", capture_start)
    try:
        if event == "run":
            widget.output_edit.setText(str(tmp_path / "new-study"))
            widget.run_analysis()
            assert len(started) == 1
            assert started[0] != old_score
        elif event == "failure":
            widget._failed("A deliberate non-simulation test failure")
        else:
            widget._cancelled()
        assert widget.scoring_label.text() != old_score
        assert "Previous Study Draw" not in widget.scoring_label.text()
    finally:
        widget.cleanup()


def test_shared_preview_scaling_preserves_source_and_handles_empty(qapp):
    from PyQt6 import QtCore, QtGui, QtWidgets

    from src.tools.shot_pattern_analysis.preview import scale_previews

    source = QtGui.QPixmap(400, 200)
    source.fill(QtGui.QColor("blue"))
    source_key = source.cacheKey()
    populated, absent, null = (QtWidgets.QLabel() for _ in range(3))
    populated.resize(100, 80)
    populated.setProperty("sourcePixmap", source)
    absent.setText("No Image")
    null.setProperty("sourcePixmap", QtGui.QPixmap())
    null.setText("Null Image")
    scale_previews([populated, absent, null])
    expected = source.scaled(
        populated.size(),
        QtCore.Qt.AspectRatioMode.KeepAspectRatio,
        QtCore.Qt.TransformationMode.SmoothTransformation,
    )
    assert populated.pixmap().toImage() == expected.toImage()
    assert populated.property("sourcePixmap").cacheKey() == source_key
    assert populated.pixmap().size() == QtCore.QSize(100, 50)
    assert absent.text() == "No Image"
    assert null.text() == "Null Image"
    populated.resize(200, 180)
    scale_previews([populated])
    assert populated.pixmap().size() == QtCore.QSize(200, 100)
    assert populated.property("sourcePixmap").cacheKey() == source_key
    scale_previews([])
