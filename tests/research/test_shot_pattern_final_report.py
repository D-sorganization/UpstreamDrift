"""The final research report never invents unavailable model results."""

from __future__ import annotations

import csv
import io

import pytest

from scripts.research.shot_pattern_analysis import build_final_report
from src.tools.shot_pattern_analysis.matched_interpretation import (
    describe_cell,
    describe_effect_modification,
    describe_scoring,
)

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


def _detailed_cell(*, status="completed"):
    cell = _cell("driver_lean0_curve1_rotation2_combined2", status=status)
    if status != "completed":
        return cell
    for item in cell["summary"].values():
        item["supported_fraction_exact95"] = [0.999, 1.0]
        item["statistics"] = {
            "mean_downrange_error_m": -4.0,
            "mean_lateral_error_m": 0.5,
            "quadrant_fractions": {
                "left_long": 0.1,
                "left_short": 0.4,
                "right_long": 0.3,
                "right_short": 0.2,
            },
        }
        item["joint_delivery_outcome_diagnostics"] = {
            "fields": ["face_deg", "loft_deg"],
            "covariance": [[1.0, 0.6], [0.6, 1.0]],
        }
    cell["course_scoring"].update(
        scenario_id="tee_fairway_rough",
        start_lie="tee",
        start_distance_m=400.0,
        hole_distance_m=400.0,
        fairway_half_width_m=15.0,
        green_radius_m=None,
        total_n=10_000,
        patterns={
            name: {"n": 9_999, "mean_strokes_gained": -0.4}
            for name in ("Straight", "Draw", "Fade")
        },
    )
    return cell


def _qualified_interpretation():
    cell = _detailed_cell()
    pair = {
        "schema": "matched-explicit-source-pair/1",
        "comparison_kind": "shaft_rotation_sd",
        "status": "conditional_only",
        "conditioning": "same_indices_supported_in_all_six_populations",
        "common_supported_n": 9_998,
        "total_n_per_population": 10_000,
        "source_protocols": ["original_primary", "pw-primary-face-bounds/1"],
        "sources": [
            {
                "protocol_id": "original_primary",
                "cell": "before",
                "source_plan_sha256": "a" * 64,
                "source_cell_manifest_sha256": "b" * 64,
                "source_csv_sha256": "c" * 64,
            },
            {
                "protocol_id": "pw-primary-face-bounds/1",
                "cell": "after",
                "source_plan_sha256": "d" * 64,
                "source_cell_manifest_sha256": "e" * 64,
                "source_csv_sha256": "f" * 64,
            },
        ],
        "analysis_source_sha256": {"matched_lean_control.py": "1" * 64},
        "course_scoring_effect_modification": {
            "Draw": {"estimate": 0.012, "lower_95": 0.006, "upper_95": 0.018},
            "Fade": {"estimate": -0.003, "lower_95": -0.009, "upper_95": 0.003},
        },
    }
    return {
        "selected_cells": [describe_cell(cell)],
        "scores": [(cell["cell"], describe_scoring(cell))],
        "effects": [describe_effect_modification(pair)],
        "figure_links": [
            {
                "club": "driver",
                "kind": "density",
                "presentation": "academic",
                "source_protocol": "original_primary",
                "source_kind": "individual_supported_endpoints",
                "relative_path": "final_figures_v1/details/driver/density.png",
            },
            {
                "club": "driver",
                "kind": "density",
                "presentation": "social",
                "source_protocol": "original_primary",
                "source_kind": "individual_supported_endpoints",
                "relative_path": "final_figures_v1/social/details/driver/density.png",
            },
        ],
        "supplement": {
            "status": "qualified",
            "unique_executed_cells": 26,
            "verified_control_aliases": 4,
            "receipt_sha256": "9" * 64,
            "comparison_inventory": "supplement_report_v1/REPORT_SUPPLEMENT_COMPARISONS.csv",
            "figure_inventory": "supplement_report_v1/REPORT_SUPPLEMENT_FIGURES.json",
        },
    }


def test_final_interpretation_uses_qualified_populations_and_additive_protocol():
    cell = _detailed_cell()
    matrix = _matrix([cell], "final")
    lean = {
        "schema": "matched-across-lean/1",
        "pairs": [],
        "population": {"planned_pairs": 0},
    }
    rendered = build_final_report.render_final_report(
        matrix, matrix, lean, {}, interpretation=_qualified_interpretation()
    )
    assert "9,999 / 10,000" in rendered
    assert "Common-Three Scored" in rendered
    assert "Face–Loft Covariance" in rendered
    assert "Left-Long" in rendered
    assert "9,998 / 10,000" in rendered
    assert "pw-primary-face-bounds/1" in rendered
    assert "supplement_report_v1/REPORT_SUPPLEMENT_COMPARISONS.csv" in rendered
    assert "final_figures_v1/social/details/driver/density.png" in rendered
    assert "not a causal" in rendered


def test_final_interpretation_does_not_invent_unavailable_assets_or_shots():
    unavailable = _detailed_cell(status="unavailable")
    matrix = _matrix([unavailable], "final")
    lean = {
        "schema": "matched-across-lean/1",
        "pairs": [],
        "population": {"planned_pairs": 0},
    }
    interpretation = _qualified_interpretation()
    interpretation["selected_cells"] = [describe_cell(unavailable)]
    interpretation["scores"] = [(unavailable["cell"], describe_scoring(unavailable))]
    interpretation["figure_links"] = []
    rendered = build_final_report.render_final_report(
        matrix, matrix, lean, {}, interpretation=interpretation
    )
    assert "Not Attempted" in rendered
    assert (
        "final_figures_v1/details/driver_lean0_curve1_rotation2_combined2/overhead.png"
        not in rendered
    )


def test_verified_original_figures_require_both_presentations(tmp_path):
    import hashlib
    import json

    cell = _detailed_cell()["cell"]
    figure_root = tmp_path / "final_figures_v1"
    figure_root.mkdir()
    paths = (
        f"details/{cell}/density.png",
        f"details/{cell}/overhead.png",
        f"social/details/{cell}/density.png",
        f"social/details/{cell}/overhead.png",
        "availability.csv",
    )
    outputs = {}
    for relative in paths:
        path = figure_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
        outputs[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema": "matched-study-figure-production/1",
        "kind": "final",
        "source_stage": "final",
        "plan_sha256": "a" * 64,
        "cell_manifest_sha256": {cell: "b" * 64},
        "availability_ledger": "availability.csv",
        "outputs": outputs,
    }
    (figure_root / "figure_manifest.json").write_text(json.dumps(manifest))
    report = _matrix([_detailed_cell()], "final")
    report["cells"][0]["config"]["noise"]["standard_deviations"][0] = 2.0
    report["cell_manifest_sha256"] = {cell: "b" * 64}
    links = build_final_report.verified_original_figure_links(
        figure_root, report, report_root=tmp_path, protocol="original_primary"
    )
    assert {(item["kind"], item["presentation"]) for item in links} == {
        ("density", "academic"),
        ("overhead", "academic"),
        ("density", "social"),
        ("overhead", "social"),
    }
    assert (
        build_final_report.verified_original_figure_inventory(
            figure_root, report, report_root=tmp_path, protocol="original_primary"
        )
        == "final_figures_v1/availability.csv"
    )
    (figure_root / paths[-1]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="figure availability inventory changed"):
        build_final_report.verified_original_figure_inventory(
            figure_root, report, report_root=tmp_path, protocol="original_primary"
        )
    (figure_root / paths[0]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="figure artifact changed"):
        build_final_report.verified_original_figure_links(
            figure_root, report, report_root=tmp_path, protocol="original_primary"
        )


def test_control_final_stage_assets_keep_control_protocol(tmp_path):
    import hashlib
    import json

    cell = "driver_lean0_curve1_rotation2_combined2"
    root = tmp_path / "control_final_figures_v1"
    outputs = {}
    for prefix in ("", "social/"):
        for kind in ("density", "overhead"):
            relative = f"{prefix}details/{cell}/{kind}.png"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(relative.encode())
            outputs[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "figure_manifest.json").write_text(
        json.dumps(
            {
                "schema": "matched-study-figure-production/1",
                "kind": "final",
                "source_stage": "final",
                "plan_sha256": "a" * 64,
                "cell_manifest_sha256": {cell: "b" * 64},
                "outputs": outputs,
            }
        )
    )
    report = _matrix([_detailed_cell()], "final")
    report["cells"][0]["config"]["noise"]["standard_deviations"][0] = 2.0
    report["cell_manifest_sha256"] = {cell: "b" * 64}
    links = build_final_report.verified_original_figure_links(
        root, report, report_root=tmp_path, protocol="original_control"
    )
    assert {item["source_protocol"] for item in links} == {"original_control"}


@pytest.mark.slow
@pytest.mark.integration
def test_publication_boundary_rejects_mutated_png_and_supplement_receipt(
    tmp_path, monkeypatch
):
    import hashlib
    import json

    from src.tools.shot_pattern_analysis import matched_supplement_pipeline

    cell = _detailed_cell()
    cell["config"]["noise"]["standard_deviations"][0] = 2.0
    name = cell["cell"]
    reports = []
    figure_roots = []
    originals = []
    manifest_hashes = {}
    for label in ("primary", "control"):
        root = tmp_path / label
        (root / name).mkdir(parents=True)
        (root / "matrix_plan.json").write_bytes(b"plan")
        (root / name / "manifest.json").write_bytes(b"manifest")
        report = _matrix([cell], "final")
        report["plan_sha256"] = hashlib.sha256(b"plan").hexdigest()
        report["cell_manifest_sha256"] = {name: hashlib.sha256(b"manifest").hexdigest()}
        reports.append((root, report))
        figure_root = tmp_path / f"{label}_figures"
        outputs = {}
        for relative in (
            f"details/{name}/density.png",
            f"details/{name}/overhead.png",
            f"social/details/{name}/density.png",
            f"social/details/{name}/overhead.png",
            "availability.csv",
        ):
            path = figure_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(relative.encode())
            outputs[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        (figure_root / "figure_manifest.json").write_text(
            json.dumps(
                {
                    "schema": "matched-study-figure-production/1",
                    "kind": "final",
                    "source_stage": "final",
                    "plan_sha256": report["plan_sha256"],
                    "cell_manifest_sha256": report["cell_manifest_sha256"],
                    "availability_ledger": "availability.csv",
                    "outputs": outputs,
                }
            )
        )
        manifest_hashes[figure_root] = hashlib.sha256(
            (figure_root / "figure_manifest.json").read_bytes()
        ).hexdigest()
        figure_roots.append(figure_root)
        originals.append((figure_root / f"details/{name}/density.png").read_bytes())
    supplement_dir = tmp_path / "supplement"
    supplement_dir.mkdir()
    receipt_path = supplement_dir / "REPORT_SUPPLEMENT_PIPELINE.json"
    receipt_path.write_bytes(b"receipt")
    receipt_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        matched_supplement_pipeline, "load_verified_supplement_artifacts", lambda _: {}
    )
    arguments = {
        "primary_root": reports[0][0],
        "primary": reports[0][1],
        "control_root": reports[1][0],
        "control": reports[1][1],
        "primary_figures": figure_roots[0],
        "control_figures": figure_roots[1],
        "figure_manifest_sha256": manifest_hashes,
        "supplement_dir": supplement_dir,
        "supplement_receipt_sha256": receipt_sha,
        "source_sha256": build_final_report._report_source_hashes(),
        "report_root": tmp_path,
    }
    build_final_report.assert_publication_inputs_unchanged(**arguments)
    png = figure_roots[0] / f"details/{name}/density.png"
    png.write_bytes(b"replaced during aggregation")
    with pytest.raises(ValueError, match="figure artifact changed"):
        build_final_report.assert_publication_inputs_unchanged(**arguments)
    png.write_bytes(originals[0])
    receipt_path.write_bytes(b"coherently replaced receipt")
    with pytest.raises(RuntimeError, match="supplement receipt changed"):
        build_final_report.assert_publication_inputs_unchanged(**arguments)
    assert not (tmp_path / "REPORT.md").exists()


def test_final_publication_receipt_binds_report_and_supplement(tmp_path):
    report = tmp_path / "REPORT.md"
    cells = tmp_path / "REPORT_CELLS.csv"
    lean = tmp_path / "REPORT_LEAN.csv"
    for path in (report, cells, lean):
        path.write_text(path.name, encoding="utf-8")
    receipt = build_final_report.final_publication_receipt(
        output_paths=(report, cells, lean),
        source_sha256={"builder.py": "a" * 64},
        primary_plan_sha256="b" * 64,
        control_plan_sha256="c" * 64,
        figure_manifest_sha256={"primary": "d" * 64, "control": "e" * 64},
        supplement_receipt_sha256="f" * 64,
    )
    assert receipt["schema"] == "matched-final-publication-receipt/1"
    assert receipt["source_sha256"]["builder.py"] == "a" * 64
    assert receipt["supplement_receipt_sha256"] == "f" * 64
    assert set(receipt["output_sha256"]) == {
        "REPORT.md",
        "REPORT_CELLS.csv",
        "REPORT_LEAN.csv",
    }


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
    build_final_report.validate_final_plans(primary, control, common_targets=targets)
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
