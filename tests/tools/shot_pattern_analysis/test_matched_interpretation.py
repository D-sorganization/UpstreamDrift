"""Evidence-bound interpretation of saved matched-shot results."""

import inspect

import pytest

from src.tools.shot_pattern_analysis.matched_interpretation import (
    describe_cell,
    describe_effect_modification,
    describe_saved_cell,
    describe_scoring,
    render_cell_table,
    render_effect_table,
    render_geometry_table,
    render_scoring_table,
    select_verified_assets,
)

pytestmark = pytest.mark.unit


def test_describe_cell_respects_function_budget():
    assert len(inspect.getsourcelines(describe_cell)[0]) <= 100


def _cell(status="completed", supported_n=9_000):
    cell = {
        "cell": "driver_lean0_curve1_rotation2_combined2",
        "status": status,
        "config": {
            "n_shots": 10_000,
            "club": {"club_id": "driver", "shaft_lean_deg": 0.0},
        },
    }
    if status == "completed":
        pattern = {
            "total_n": 10_000,
            "supported_n": supported_n,
            "supported_fraction_exact95": [0.89, 0.91],
            "mean_carry_m": 179.0,
            "downrange_sd_m": 12.0,
            "lateral_sd_m": 8.0,
            "target_rmse_m": 15.0,
            "statistics": {
                "mean_downrange_error_m": 0.5,
                "mean_lateral_error_m": -0.4,
                "quadrant_fractions": {
                    "left_long": 0.1,
                    "left_short": 0.4,
                    "right_long": 0.3,
                    "right_short": 0.2,
                },
            },
            "joint_delivery_outcome_diagnostics": {
                "fields": ["face_deg", "loft_deg"],
                "covariance": [[4.0, 1.5], [1.5, 2.0]],
                "n": 9000,
            },
        }
        cell.update(
            target_x_m=178.5,
            summary=dict.fromkeys(("Straight", "Draw", "Fade"), pattern),
            nominals={
                name: {"face_deg": 1.5, "path_deg": 3.0, "speed_mps": 45.0}
                for name in ("Straight", "Draw", "Fade")
            },
        )
    return cell


def _pair():
    return {
        "schema": "matched-explicit-source-pair/1",
        "comparison_kind": "curve_strength",
        "status": "conditional_only",
        "conditioning": "same_indices_supported_in_all_six_populations",
        "common_supported_n": 8_000,
        "total_n_per_population": 10_000,
        "source_protocols": ["original", "pw-primary-face-bounds/1"],
        "sources": [
            {
                "protocol_id": "original",
                "cell": "first",
                "source_plan_sha256": "a" * 64,
                "source_cell_manifest_sha256": "b" * 64,
                "source_csv_sha256": "c" * 64,
            },
            {
                "protocol_id": "pw-primary-face-bounds/1",
                "cell": "second",
                "source_plan_sha256": "d" * 64,
                "source_cell_manifest_sha256": "e" * 64,
                "source_csv_sha256": "f" * 64,
            },
        ],
        "analysis_source_sha256": {"matched_lean_control.py": "1" * 64},
        "course_scoring_effect_modification": {
            "Draw": {"estimate": 0.02, "lower_95": -0.01, "upper_95": 0.05},
            "Fade": {"estimate": -0.01, "lower_95": -0.03, "upper_95": 0.01},
        },
    }


def test_completed_cell_reports_spread_bias_quadrants_and_optional_geometry():
    rows = {
        name: [{"face_deg": 1.0, "loft_deg": 11.0}, {"face_deg": 2.0, "loft_deg": 12.0}]
        for name in ("Straight", "Draw", "Fade")
    }
    result = describe_cell(_cell(supported_n=2), delivered_rows=rows)
    draw = result["patterns"]["Draw"]
    assert result["status"] == "completed"
    assert draw["mean_carry_m"] == 179.0
    assert draw["mean_downrange_error_m"] == 0.5
    assert draw["supported_fraction_ci95"] == (0.89, 0.91)
    assert draw["face_loft_covariance_deg2"] == 1.5
    assert draw["mean_delivered_face_deg"] == 1.5
    assert draw["mean_delivered_loft_deg"] == 11.5
    assert draw["quadrant_fractions"]["left_short"] == 0.4


def test_delivered_mean_cannot_be_claimed_from_incomplete_supported_rows():
    rows = {
        name: [{"face_deg": 1.0, "loft_deg": 11.0}]
        for name in ("Straight", "Draw", "Fade")
    }
    with pytest.raises(ValueError, match="population differs"):
        describe_cell(_cell(), delivered_rows=rows)


def test_unavailable_cell_is_not_mislabeled_as_zero_supported_shots():
    result = describe_cell(_cell("unavailable"))
    assert result["attempted_n_per_pattern"] == 0
    assert result["patterns"]["Straight"]["status"] == "Not Attempted"
    assert result["patterns"]["Straight"]["supported_n"] is None


def test_completed_cell_with_one_supported_shot_has_no_estimable_spread():
    cell = _cell(supported_n=1)
    item = cell["summary"]["Straight"]
    item["statistics"] = None
    item["joint_delivery_outcome_diagnostics"] = None
    result = describe_cell(cell)
    assert result["patterns"]["Straight"]["status"] == "Insufficient Supported Shots"
    assert result["patterns"]["Straight"]["supported_n"] == 1
    table = render_cell_table([result])
    assert "1 / 10,000" in table
    assert "Insufficient Supported Shots" in table


def test_scoring_keeps_absolute_means_and_population_distinct():
    cell = _cell()
    cell["course_scoring"] = {
        "status": "conditional_only",
        "scenario_id": "tee_fairway_rough",
        "start_lie": "tee",
        "hole_distance_m": 400.0,
        "fairway_half_width_m": 15.0,
        "green_radius_m": None,
        "paired_n": 8500,
        "total_n": 10000,
        "patterns": {
            name: {"n": 8500, "mean_strokes_gained": -0.4}
            for name in ("Straight", "Draw", "Fade")
        },
    }
    result = describe_scoring(cell)
    assert result["status"] == "conditional_only"
    assert result["paired_n"] == 8500
    assert result["patterns"]["Draw"]["absolute_mean_strokes_gained"] == -0.4
    assert result["patterns"]["Draw"]["individual_endpoint_supported_n"] == 9000
    assert result["patterns"]["Draw"]["common_three_scored_n"] == 8500
    assert result["context"]["fairway_half_width_m"] == 15.0


def test_conditional_scoring_remains_reportable_with_its_population():
    cell = _cell()
    cell["course_scoring"] = {
        "status": "conditional_only",
        "scenario_id": "fairway_green",
        "start_lie": "fairway",
        "start_distance_m": 128.0,
        "hole_distance_m": None,
        "fairway_half_width_m": None,
        "green_radius_m": 15.0,
        "paired_n": 8_500,
        "total_n": 10_000,
        "patterns": {
            name: {"n": 8_500, "mean_strokes_gained": 0.1}
            for name in ("Straight", "Draw", "Fade")
        },
    }
    result = describe_scoring(cell)
    assert result["status"] == "conditional_only"
    assert result["patterns"]["Straight"]["absolute_mean_strokes_gained"] == 0.1


def test_effect_modification_requires_common_six_pair_and_explicit_scope():
    pair = _pair()
    result = describe_effect_modification(pair)
    assert result["common_six_n"] == 8000
    assert result["draw_minus_straight_change"]["estimate"] == 0.02
    assert "pointwise" in result["inference_limit"]
    with pytest.raises(ValueError, match="six-population"):
        describe_effect_modification({**pair, "conditioning": "individual means"})
    with pytest.raises(ValueError, match="source evidence"):
        describe_effect_modification({**pair, "sources": []})
    with pytest.raises(ValueError, match="population"):
        describe_effect_modification({**pair, "status": "available"})
    assert "first" in render_effect_table([result])
    assert "second" in render_effect_table([result])


def test_assets_must_be_in_manifest_and_exist(tmp_path):
    import hashlib

    (tmp_path / "social").mkdir()
    image = tmp_path / "social" / "density.png"
    image.write_bytes(b"png")
    entries = [
        {
            "club": "driver",
            "cell": "c",
            "png": "social/density.png",
            "status": "completed",
        }
    ]
    manifest = {
        "outputs": {"social/density.png": hashlib.sha256(b"png").hexdigest()},
        "cell_manifest_sha256": {"c": "a" * 64},
    }
    result = select_verified_assets(
        tmp_path,
        entries,
        manifest,
        club="driver",
        kind="density",
        presentation="social",
    )
    assert result == [{"kind": "density", "presentation": "social", "path": image}]
    with pytest.raises(ValueError, match="presentation"):
        select_verified_assets(
            tmp_path,
            entries,
            manifest,
            club="driver",
            kind="density",
            presentation="academic",
        )
    image.unlink()
    with pytest.raises(ValueError, match="missing"):
        select_verified_assets(
            tmp_path,
            entries,
            manifest,
            club="driver",
            kind="density",
            presentation="social",
        )


def test_asset_link_refuses_uninventoried_cell_or_changed_bytes(tmp_path):
    import hashlib

    image = tmp_path / "density.png"
    image.write_bytes(b"changed")
    entries = [
        {"club": "driver", "cell": "c", "png": "density.png", "status": "completed"}
    ]
    manifest = {
        "outputs": {"density.png": hashlib.sha256(b"png").hexdigest()},
        "cell_manifest_sha256": {"c": "a" * 64},
    }
    with pytest.raises(ValueError, match="hash"):
        select_verified_assets(
            tmp_path,
            entries,
            manifest,
            club="driver",
            kind="density",
            presentation="academic",
        )
    manifest["outputs"]["density.png"] = hashlib.sha256(b"changed").hexdigest()
    manifest["cell_manifest_sha256"] = {}
    with pytest.raises(ValueError, match="inventoried"):
        select_verified_assets(
            tmp_path,
            entries,
            manifest,
            club="driver",
            kind="density",
            presentation="academic",
        )


def test_saved_cell_means_use_only_supported_rows(monkeypatch, tmp_path):
    import csv
    import gzip

    from src.tools.shot_pattern_analysis import matched_interpretation

    monkeypatch.setattr(
        matched_interpretation, "load_validated_cell", lambda _: _cell(supported_n=1)
    )
    monkeypatch.setattr(matched_interpretation, "manifest_files_match", lambda _: True)
    (tmp_path / "manifest.json").write_text("original")
    with gzip.open(tmp_path / "shots.csv.gz", "wt", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("pattern", "status", "face_deg", "loft_deg")
        )
        writer.writeheader()
        for name in ("Straight", "Draw", "Fade"):
            writer.writerow(
                {
                    "pattern": name,
                    "status": "supported",
                    "face_deg": 1.0,
                    "loft_deg": 10.0,
                }
            )
            writer.writerow(
                {
                    "pattern": name,
                    "status": "off_face",
                    "face_deg": 500.0,
                    "loft_deg": 500.0,
                }
            )
    result = describe_saved_cell(tmp_path)
    assert result["patterns"]["Draw"]["mean_delivered_face_deg"] == 1.0
    assert result["patterns"]["Draw"]["delivered_mean_n"] == 1


def test_saved_cell_rejects_coherent_manifest_replacement_during_read(
    monkeypatch, tmp_path
):
    import csv
    import gzip

    from src.tools.shot_pattern_analysis import matched_interpretation

    monkeypatch.setattr(
        matched_interpretation, "load_validated_cell", lambda _: _cell(supported_n=1)
    )
    (tmp_path / "manifest.json").write_text("original")
    with gzip.open(tmp_path / "shots.csv.gz", "wt", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("pattern", "status", "face_deg", "loft_deg")
        )
        writer.writeheader()
        for name in ("Straight", "Draw", "Fade"):
            writer.writerow(
                {
                    "pattern": name,
                    "status": "supported",
                    "face_deg": 1.0,
                    "loft_deg": 10.0,
                }
            )

    def replace_at_final_check(_):
        (tmp_path / "manifest.json").write_text("replacement")
        return True

    monkeypatch.setattr(
        matched_interpretation, "manifest_files_match", replace_at_final_check
    )
    with pytest.raises(ValueError, match="changed during"):
        describe_saved_cell(tmp_path)


def test_saved_cell_rejects_replacement_inside_loader(monkeypatch, tmp_path):
    from src.tools.shot_pattern_analysis import matched_interpretation

    (tmp_path / "manifest.json").write_text("initial")
    (tmp_path / "shots.csv.gz").write_bytes(b"initial")

    def changed_loader(_):
        (tmp_path / "manifest.json").write_text("new coherent manifest")
        (tmp_path / "shots.csv.gz").write_bytes(b"new coherent csv")
        return _cell("unavailable")

    monkeypatch.setattr(matched_interpretation, "load_validated_cell", changed_loader)
    with pytest.raises(ValueError, match="changed during"):
        describe_saved_cell(tmp_path)


def test_saved_unavailable_cell_does_not_read_nonexistent_csv(monkeypatch, tmp_path):
    from src.tools.shot_pattern_analysis import matched_interpretation

    monkeypatch.setattr(
        matched_interpretation, "load_validated_cell", lambda _: _cell("unavailable")
    )
    (tmp_path / "manifest.json").write_text("unavailable")
    assert (
        describe_saved_cell(tmp_path)["patterns"]["Fade"]["status"] == "Not Attempted"
    )


def test_report_tables_keep_denominators_and_noncausal_scope():
    table = render_cell_table([describe_cell(_cell())])
    assert "9,000 / 10,000" in table
    assert "[89.0%, 91.0%]" in table
    assert "X Bias / SD" in table
    assert "Not Attempted" in render_cell_table([describe_cell(_cell("unavailable"))])
    effect_table = render_effect_table([describe_effect_modification(_pair())])
    assert "8,000 / 10,000" in effect_table
    assert "+0.020" in effect_table
    assert "not a causal" in effect_table


def test_scoring_table_states_absolute_context_and_individual_population():
    cell = _cell()
    cell["course_scoring"] = {
        "status": "conditional_only",
        "scenario_id": "tee_fairway_rough",
        "start_lie": "tee",
        "hole_distance_m": 400.0,
        "fairway_half_width_m": 15.0,
        "green_radius_m": None,
        "paired_n": 8_500,
        "total_n": 10_000,
        "patterns": {
            name: {"n": 8_500, "mean_strokes_gained": -0.4}
            for name in ("Straight", "Draw", "Fade")
        },
    }
    table = render_scoring_table([(cell["cell"], describe_scoring(cell))])
    assert "tee" in table
    assert "400.0 m" in table
    assert "15.0 m" in table
    assert "-0.400" in table
    assert "9,000" in table and "8,500" in table
    assert "Common-Three Scored" in table


def test_geometry_table_does_not_claim_face_causes_long_left():
    rows = {
        name: [{"face_deg": 1.0, "loft_deg": 11.0}] * 9_000
        for name in ("Straight", "Draw", "Fade")
    }
    table = render_geometry_table([describe_cell(_cell(), delivered_rows=rows)])
    assert "Face–Loft Covariance" in table
    assert "1.500" in table
    assert "Left-Long" in table
    assert "do not identify" in table
