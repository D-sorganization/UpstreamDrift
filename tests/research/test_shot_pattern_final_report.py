"""The final research report never invents unavailable model results."""

from __future__ import annotations

import csv
import io

import pytest

from scripts.research.shot_pattern_analysis import build_final_report

pytestmark = pytest.mark.unit


def _cell(name: str, *, status: str = "completed") -> dict:
    config = {
        "club": {
            "club_id": "driver",
            "shaft_lean_deg": 0.0,
            "loft_deg": 10.0,
            "lie_deg": 58.5,
            "club_speed_mps": 45.0,
            "attack_angle_deg": -0.9,
            "clubhead_mass_kg": 0.2,
        },
        "noise": {"standard_deviations": [1.0, 0, 0, 0, 0, 0, 0], "correlation": None},
        "curve_scale": 1.0,
        "n_shots": 10_000,
        "target_x_m": None,
        "seed": 20261009,
        "match_tolerance_m": 0.05,
    }
    result = {"cell": name, "status": status, "config": config}
    if status == "unavailable":
        return {**result, "reason": "nonairborne_nominal"}
    interval = {"estimate": 0.02, "lower_95": 0.01, "upper_95": 0.03}
    return {
        **result,
        "target_x_m": 179.0,
        "nominals": {
            name: {"face_deg": 0.0, "path_deg": 0.0, "speed_mps": 45.0}
            for name in ("Straight", "Draw", "Fade")
        },
        "summary": {
            name: {
                "total_n": 10_000,
                "supported_n": 9_999,
                "lateral_sd_m": 5.0,
                "downrange_sd_m": 3.0,
                "target_rmse_m": 6.0,
                "target_containment_fraction": 0.8,
                "mean_carry_m": 179.0,
                "realized_face_path_correlation": 0.1,
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
    assert "Missing Metric in Completed Cells" in text
    assert "1 unavailable planned cell" in text
    assert "successful nominal matching under the stated bounds" in text
    assert "Infeasible cells" not in text
    assert "| Mean Radial Error Difference | Draw | 1 | 0 | 0 | 0 |" in text


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


def test_main_report_uses_compact_tables_and_separate_complete_inventory():
    cell = _cell("driver_lean0_curve1_rotation1_combined2")
    matrix = _matrix([cell], "final")
    lean = {
        "schema": "matched-across-lean/1",
        "pairs": [],
        "population": {"planned_pairs": 0},
    }
    text = build_final_report.render_final_report(matrix, matrix, lean, {})
    assert "REPORT_CELLS.csv" in text
    assert "9-profile" in text or "Nine-Profile" in text
    assert "common paired" in text.lower()
    assert "Mean Carry" in text
    assert "final_figures_v1/comparisons/driver/combined2" in text
    assert "### Complete Cell Inventory" not in text
    rows = list(
        csv.DictReader(
            io.StringIO(build_final_report.render_cell_inventory(matrix, matrix))
        )
    )
    assert len(rows) == 6
    assert {row["pattern"] for row in rows} == {"Straight", "Draw", "Fade"}
    assert rows[1]["nominal_speed_mps"] == "45.0"
    assert rows[1]["planned_n_per_pattern"] == "10000"
    assert rows[1]["attempted_n_per_pattern"] == "10000"


def test_unavailable_inventory_separates_planned_from_attempted_shots():
    unavailable = _cell(
        "driver_lean10_curve1_rotation1_shaft_only", status="unavailable"
    )
    matrix = _matrix([unavailable], "final")
    rows = list(
        csv.DictReader(
            io.StringIO(build_final_report.render_cell_inventory(matrix, matrix))
        )
    )
    assert len(rows) == 6
    assert all(row["planned_n_per_pattern"] == "10000" for row in rows)
    assert all(row["attempted_n_per_pattern"] == "0" for row in rows)
    assert all(row["supported_n"] == "" for row in rows)


def test_compact_comparison_table_does_not_wrap_full_optimizer_failure() -> None:
    unavailable = _cell("driver_lean0_curve2_rotation2_combined2", status="unavailable")
    unavailable["reason"] = (
        "nominal landing unreachable within bounds: residual=0.351838 m"
    )
    rendered = "\n".join(
        build_final_report._selected_comparison_lines(_matrix([unavailable], "final"))
    )
    assert "Nominal Match Unavailable (See Inventory)" in rendered
    assert "residual=0.351838" not in rendered
    inventory = build_final_report.render_cell_inventory(
        _matrix([unavailable], "final"), _matrix([], "final")
    )
    assert "residual=0.351838" in inventory


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
    from dataclasses import asdict
    import json

    from src.tools.shot_pattern_analysis.matched_matrix import matched_matrix_cells

    targets = {
        "driver": 178.94875782337786,
        "seven_iron": 127.88194166336235,
        "pitching_wedge": 77.40247724866299,
    }
    with pytest.raises(ValueError, match="324"):
        build_final_report.validate_final_plans(
            {"planned_cells": 1, "stage": "final", "cells": []},
            {"planned_cells": 72, "stage": "final", "cells": []},
            common_targets=targets,
        )

    def report(configs):
        return {
            "planned_cells": len(configs),
            "stage": "final",
            "cells": [
                {"cell": name, "config": json.loads(json.dumps(asdict(config)))}
                for name, config in configs
            ],
        }

    primary = report(matched_matrix_cells(n_shots=10_000, seed=20261009))
    control = report(
        matched_matrix_cells(
            n_shots=10_000,
            seed=20261009,
            profiles=("shaft_only", "combined2"),
            common_targets=targets,
        )
    )
    build_final_report.validate_final_plans(
        primary, control, common_targets=targets
    )
    altered = dict(primary)
    altered["cells"] = list(primary["cells"])
    altered["cells"][0] = {
        **altered["cells"][0],
        "config": {**altered["cells"][0]["config"], "seed": 2},
    }
    with pytest.raises(ValueError, match="reviewed protocol"):
        build_final_report.validate_final_plans(
            altered, control, common_targets=targets
        )
