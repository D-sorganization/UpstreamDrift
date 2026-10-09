"""Contracts for exporting validated summaries of the completed 24-cell matrix."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _make_matrix(root: Path) -> Path:
    from src.tools.shot_pattern_analysis.matrix import corrected_matrix

    baseline_hash = hashlib.sha256(b"same source-backed baseline").hexdigest()
    digest = hashlib.sha256(b"frozen sources").hexdigest()
    native_digest = hashlib.sha256(b"native library").hexdigest()
    for name, config_obj in corrected_matrix():
        config = asdict(config_obj)
        output = root / name
        output.mkdir(parents=True)
        start = {
            "config": config,
            "source_sha256": {"src/example.py": digest},
            "native_binary_sha256": native_digest,
        }
        _write_json(output / "run_start.json", start)
        receipt = {
            "source_unchanged_during_run": True,
            "execution_source_sha256": start["source_sha256"],
            "execution_native_binary_sha256": native_digest,
            "shots_per_pattern": 10_000,
        }
        _write_json(output / "receipt.json", receipt)
        driver = config["club_id"] == "driver"
        scenario_id = "tee_fairway_rough" if driver else "approach_green_rough"
        scoring = {
            "status": "available",
            "scenario_id": scenario_id,
            "club_id": config["club_id"],
            "source_backed_status": "available",
            "scored_shots": 30_000,
            "api_evaluated_states": 100,
            "interpolation_tolerance_strokes": 1e-12,
            "start_lie": "tee" if driver else "fairway",
            "start_distance_m": 400.0 if driver else 150.0,
            "target_x_m": 200.0 if driver else 150.0,
            "hole_distance_m": 400.0 if driver else None,
            "fairway_half_width_m": 15.0 if driver else None,
            "green_radius_m": None if driver else 15.0,
            "baseline": {"table_sha256": baseline_hash},
            "patterns": {
                pattern: {
                    "n": 10_000,
                    "mean_strokes_gained": 0.2,
                    "fairway_fraction" if driver else "green_fraction": 0.9,
                }
                for pattern in ("Straight", "Draw", "Fade")
            },
            "paired_benefit_vs_straight": {
                pattern: {"estimate": 0.1, "lower_95": 0.05, "upper_95": 0.15}
                for pattern in ("Draw", "Fade")
            },
        }
        summary = {
            "title": "Shot Pattern Analysis",
            "config": config,
            "target_x_m": 200.0 if driver else 150.0,
            "patterns": {
                pattern: {
                    "n": 10_000,
                    "aimed_lateral_sd_m": 4.0,
                    "mean_carry_m": 199.0,
                    "carry_sd_m": 0.3,
                    "aimed_target_rmse_m": 5.0,
                    "aimed_target_hit_fraction": 0.8,
                }
                for pattern in ("Straight", "Draw", "Fade")
            },
            "landing_dispersion": {
                pattern: {
                    "corr_lateral_downrange_error": -0.2,
                    "p95_radial_target_error_m": 8.0,
                }
                for pattern in ("Straight", "Draw", "Fade")
            },
            "equal_range_diagnostic": {
                "patterns": {
                    pattern: {"lateral_sd_m": 3.8}
                    for pattern in ("Straight", "Draw", "Fade")
                }
            },
            "course_scoring": scoring,
        }
        _write_json(output / "summary.json", summary)
        _write_json(output / "strokes_gained.json", scoring)
        _write_json(
            output / "strokes_gained_baseline.json",
            {"baseline": {"table_sha256": baseline_hash}},
        )
        for filename in (
            "shots.csv",
            "overhead_flight.png",
            "dispersion.png",
            "dispersion_equal_range.png",
        ):
            (output / filename).write_bytes(b"artifact")
        manifest_files = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in output.iterdir()
            if path.is_file()
        }
        _write_json(
            output / "manifest.json",
            {"scenario": name, "config": config, "files_sha256": manifest_files},
        )
    return root


@pytest.mark.slow
def test_full_matrix_exports_72_statistics_rows_and_four_artifacts(
    tmp_path: Path,
) -> None:
    from PIL import Image

    from src.tools.shot_pattern_analysis.overview import export_overview

    matrix_root = _make_matrix(tmp_path / "matrix")
    outputs = export_overview(matrix_root, tmp_path / "overview")

    assert set(outputs) == {
        "statistics_csv",
        "statistics_json",
        "fixed_loft_forest_png",
        "shaft_rotation_forest_png",
    }
    with outputs["statistics_csv"].open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 72
    assert {row["club_id"] for row in rows} == {
        "driver",
        "seven_iron",
        "pitching_wedge",
    }
    assert {row["pattern"] for row in rows} == {"Straight", "Draw", "Fade"}
    assert rows[0]["corr_lateral_downrange_error"] == "-0.2"
    payload = json.loads(outputs["statistics_json"].read_text(encoding="utf-8"))
    assert len(payload["rows"]) == 72
    assert payload["provenance"]["validated_matrix_cells"] == 24
    assert len(payload["provenance"]["manifest_sha256_by_scenario"]) == 24
    for key in ("fixed_loft_forest_png", "shaft_rotation_forest_png"):
        assert Image.open(outputs[key]).size == (1920, 1080)


@pytest.mark.slow
def test_partial_matrix_is_rejected(tmp_path: Path) -> None:
    from src.tools.shot_pattern_analysis.overview import load_validated_matrix

    matrix_root = _make_matrix(tmp_path / "matrix")
    (matrix_root / "driver_fixed_loft_sd1_scale1").rename(
        matrix_root / "driver_fixed_loft_sd1_scale1.partial"
    )
    with pytest.raises(ValueError, match="24|missing|complete"):
        load_validated_matrix(matrix_root)


def test_cli_accepts_matrix_root_and_output_directory(monkeypatch, capsys) -> None:  # noqa: ANN001
    from src.tools.shot_pattern_analysis import overview

    monkeypatch.setattr(
        overview,
        "export_overview",
        lambda matrix_root, output_dir: {
            "statistics_csv": Path(output_dir) / "overview_statistics.csv"
        },
    )
    result = overview.main(["matrix", "--output-dir", "overview"])

    assert result == 0
    assert capsys.readouterr().out.strip().endswith("overview_statistics.csv")


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param("file_hash", marks=pytest.mark.slow),
        pytest.param("superseded", marks=pytest.mark.slow),
        pytest.param("baseline", marks=pytest.mark.slow),
    ],
)
def test_untrusted_or_superseded_cell_is_rejected(tmp_path: Path, failure: str) -> None:
    from src.tools.shot_pattern_analysis.overview import load_validated_matrix

    matrix_root = _make_matrix(tmp_path / "matrix")
    bundle = matrix_root / "driver_fixed_loft_sd1_scale1"
    if failure == "file_hash":
        (bundle / "summary.json").write_text("{}\n", encoding="utf-8")
    elif failure == "superseded":
        summary_path = bundle / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["analysis_status"] = "superseded_impact_approximation"
        _write_json(summary_path, summary)
        manifest_path = bundle / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files_sha256"]["summary.json"] = hashlib.sha256(
            summary_path.read_bytes()
        ).hexdigest()
        _write_json(manifest_path, manifest)
    else:
        bundle = matrix_root / "driver_fixed_loft_sd2_scale1"
        other = bundle / "summary.json"
        summary = json.loads(other.read_text())
        summary["course_scoring"]["baseline"]["table_sha256"] = "changed baseline"
        _write_json(other, summary)
        _write_json(bundle / "strokes_gained.json", summary["course_scoring"])
        _write_json(
            bundle / "strokes_gained_baseline.json",
            {"baseline": {"table_sha256": "changed baseline"}},
        )
        manifest_path = bundle / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files_sha256"]["summary.json"] = hashlib.sha256(
            other.read_bytes()
        ).hexdigest()
        manifest["files_sha256"]["strokes_gained.json"] = hashlib.sha256(
            (bundle / "strokes_gained.json").read_bytes()
        ).hexdigest()
        manifest["files_sha256"]["strokes_gained_baseline.json"] = hashlib.sha256(
            (bundle / "strokes_gained_baseline.json").read_bytes()
        ).hexdigest()
        _write_json(manifest_path, manifest)

    expected_message = (
        "benchmarks|same source-backed" if failure == "baseline" else None
    )
    with pytest.raises(ValueError, match=expected_message):
        load_validated_matrix(matrix_root)
