"""The final research report never invents unavailable model results."""

from __future__ import annotations

import pytest

from scripts.research.shot_pattern_analysis import build_final_report

pytestmark = pytest.mark.unit


def _cell(name: str, *, status: str = "completed") -> dict:
    config = {
        "club": {"club_id": "driver", "shaft_lean_deg": 0.0},
        "noise": {"standard_deviations": [1.0, 0, 0, 0, 0, 0, 0]},
        "curve_scale": 1.0,
        "n_shots": 10_000,
        "target_x_m": None,
    }
    result = {"cell": name, "status": status, "config": config}
    if status == "unavailable":
        return {**result, "reason": "nonairborne_nominal"}
    interval = {"estimate": 0.02, "lower_95": 0.01, "upper_95": 0.03}
    return {
        **result,
        "target_x_m": 179.0,
        "summary": {
            name: {
                "total_n": 10_000,
                "supported_n": 9_999,
                "lateral_sd_m": 5.0,
                "downrange_sd_m": 3.0,
                "target_rmse_m": 6.0,
                "target_containment_fraction": 0.8,
            }
            for name in ("Straight", "Draw", "Fade")
        },
        "paired_comparisons": {
            "paired_n": 9_999,
            "mean_radial_error_difference_vs_straight": {
                "Draw": interval,
                "Fade": interval,
            },
            "mean_squared_target_error_difference_m2": {
                "Draw": interval,
                "Fade": interval,
            },
        },
        "course_scoring": {
            "status": "conditional_only",
            "paired_n": 9_999,
            "paired_benefit_vs_straight": {"Draw": interval, "Fade": interval},
        },
    }


def _matrix(cells: list[dict], stage: str) -> dict:
    counts = {
        status: sum(cell["status"] == status for cell in cells)
        for status in ("completed", "unavailable", "not_run")
    }
    return {
        "schema": "matched-study-report/1",
        "stage": stage,
        "planned_cells": len(cells),
        "counts": counts,
        "cells": cells,
        "plan_sha256": "a" * 64,
    }


def test_report_shows_completed_and_unavailable_without_fabricating_effects():
    primary = _matrix(
        [
            _cell("driver_lean0_curve1_rotation1_shaft_only"),
            _cell("driver_lean10_curve1_rotation1_shaft_only", status="unavailable"),
        ],
        "final",
    )
    control = _matrix(
        [_cell("driver_lean0_curve1_rotation1_shaft_only")], "control_final"
    )
    lean = {
        "schema": "matched-across-lean/1",
        "pairs": [],
        "population": {"planned_pairs": 0},
    }
    text = build_final_report.render_final_report(primary, control, lean, {})
    assert "10,000" in text
    assert "1 / 2" in text
    assert "nonairborne_nominal" in text
    assert "0.0200 [0.0100, 0.0300]" in text
    assert "conditional" in text.lower()
    assert "Phase One" in text


def test_report_rejects_incomplete_or_wrong_study_population():
    completed = _cell("driver_lean0_curve1_rotation1_shaft_only")
    control = _matrix(
        [_cell("driver_lean0_curve1_rotation1_shaft_only")], "control_final"
    )
    lean = {
        "schema": "matched-across-lean/1",
        "pairs": [],
        "population": {"planned_pairs": 0},
    }
    with pytest.raises(ValueError, match="incomplete"):
        build_final_report.render_final_report(
            _matrix(
                [completed, {**completed, "cell": "later", "status": "not_run"}],
                "final",
            ),
            control,
            lean,
            {},
        )
    completed["summary"]["Draw"]["total_n"] = 32
    with pytest.raises(ValueError, match="10,000"):
        build_final_report.render_final_report(
            _matrix([completed], "final"), control, lean, {}
        )
    unavailable = _cell(
        "driver_lean10_curve1_rotation1_shaft_only", status="unavailable"
    )
    unavailable["config"]["n_shots"] = 32
    with pytest.raises(ValueError, match="10,000"):
        build_final_report.render_final_report(
            _matrix([unavailable], "final"), control, lean, {}
        )


def test_course_sensitivity_is_reported_only_when_supplied():
    cell = _cell("driver_lean0_curve1_rotation1_shaft_only")
    matrix = _matrix([cell], "final")
    lean = {
        "schema": "matched-across-lean/1",
        "pairs": [],
        "population": {"planned_pairs": 0},
    }
    text = build_final_report.render_final_report(matrix, matrix, lean, {})
    assert "Course-Dimension Sensitivity" in text
    assert "not computed" in text.lower()


def test_matrix_input_change_is_rejected_before_publication(tmp_path):
    import hashlib

    (tmp_path / "matrix_plan.json").write_text("fixed plan", encoding="utf-8")
    cell = tmp_path / "cell"
    cell.mkdir()
    (cell / "manifest.json").write_text("fixed manifest", encoding="utf-8")

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    report = {
        "plan_sha256": digest(tmp_path / "matrix_plan.json"),
        "cell_manifest_sha256": {"cell": digest(cell / "manifest.json")},
    }
    build_final_report.assert_matrix_inputs_unchanged(tmp_path, report)
    (cell / "manifest.json").write_text("changed manifest", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        build_final_report.assert_matrix_inputs_unchanged(tmp_path, report)


def test_final_builder_requires_full_324_and_72_cell_protocols():
    with pytest.raises(ValueError, match="324"):
        build_final_report.validate_final_plans(
            {"planned_cells": 1, "stage": "final_v1"},
            {"planned_cells": 72, "stage": "control_final_v1"},
        )
    build_final_report.validate_final_plans(
        {"planned_cells": 324, "stage": "final_v1"},
        {"planned_cells": 72, "stage": "control_final_v1"},
    )
