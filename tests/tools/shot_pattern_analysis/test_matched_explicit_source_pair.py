"""Paired source comparisons use common support and centered primitive draws."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import numpy as np

from src.tools.shot_pattern_analysis.matched_config import matched_preset
from src.tools.shot_pattern_analysis.noise import NoiseSpec

pytestmark = pytest.mark.unit
PATTERNS = ("Straight", "Draw", "Fade")


def test_effect_modification_bootstraps_difference_of_within_condition_benefits() -> (
    None
):
    from src.tools.shot_pattern_analysis.matched_lean_control import (
        _paired_effect_modification,
    )

    result = _paired_effect_modification(
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([0.0, 1.0, 2.0]),
        np.array([2.0, 3.0, 4.0]),
        seed=7,
    )
    assert result["estimate"] == pytest.approx(1.0)
    assert result["lower_95"] == pytest.approx(1.0)
    assert result["upper_95"] == pytest.approx(1.0)


def test_rotation_sd_pairing_compares_standardized_rotation_draws() -> None:
    from src.tools.shot_pattern_analysis.matched_lean_control import (
        _normalized_rotation_errors,
    )

    one_degree = {pattern: {0: (0.25, 1.0, 2.0)} for pattern in PATTERNS}
    two_degree = {pattern: {0: (0.5, 1.0, 2.0)} for pattern in PATTERNS}
    assert _normalized_rotation_errors(one_degree, 1.0) == _normalized_rotation_errors(
        two_degree, 2.0
    )


def _write_cell(
    root: Path,
    *,
    lean: float,
    target_request: float | None,
    protocol: str,
    invalid: tuple[str, int] | None = None,
) -> dict:
    root.mkdir(parents=True)
    club = replace(matched_preset("pitching_wedge"), shaft_lean_deg=lean)
    config = {
        "club": asdict(club),
        "noise": asdict(NoiseSpec((1.0, 0.5, 0.44704, 0.00635, 0.00635, 0.0, 0.0))),
        "n_shots": 4,
        "seed": 20261009,
        "curve_scale": 2.0,
        "target_x_m": target_request,
        "match_tolerance_m": 0.05,
        "target_radius_m": 15.0,
        "match_face_bounds_deg": [-6.0, 6.0],
        "match_speed_factors": [0.75, 1.25],
    }
    cell = f"pitching_wedge_lean{lean:g}_curve2_rotation1_shaft_only"
    execution = {"source_sha256": {"frozen.py": "fixture"}}
    target = 77.40247724866299
    nominals = {}
    for index, pattern in enumerate(PATTERNS):
        face = (index - 1) * (2.0 + lean * 0.01)
        path = (index - 1) * (3.5 - lean * 0.01)
        speed = 30.0 + index * 0.2 + lean * 0.01
        nominals[pattern] = {
            "face_deg": face,
            "path_deg": path,
            "speed_mps": speed,
            "residual_m": 0.0,
            "outcome": {"carry_x_m": target, "carry_y_m": 0.0},
        }
    directory = root / cell
    directory.mkdir()
    (directory / "run_start.json").write_text(
        json.dumps({"config": config, "execution": execution})
    )
    summary = {
        "config": config,
        "status": "completed",
        "target_x_m": target,
        "nominals": nominals,
        "summary": {pattern: {"total_n": 4, "supported_n": 4} for pattern in PATTERNS},
    }
    (directory / "summary.json").write_text(json.dumps(summary))
    fields = (
        "pattern",
        "shot_index",
        "status",
        "shaft_rotation_deg",
        "path_deg",
        "speed_mps",
        "contact_horizontal_m",
        "contact_vertical_m",
        "shaft_lean_deg",
        "attack_angle_deg",
        "carry_x_m",
        "carry_y_m",
    )
    with gzip.open(directory / "shots.csv.gz", "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pattern in PATTERNS:
            for index in range(4):
                is_invalid = invalid == (pattern, index)
                writer.writerow(
                    {
                        "pattern": pattern,
                        "shot_index": index,
                        "status": "off_face" if is_invalid else "supported",
                        "shaft_rotation_deg": (index + 1) * 0.1,
                        "path_deg": nominals[pattern]["path_deg"] + (index + 1) * 0.05,
                        "speed_mps": nominals[pattern]["speed_mps"] + (index + 1) * 0.1,
                        "contact_horizontal_m": index * 0.001,
                        "contact_vertical_m": -index * 0.001,
                        "shaft_lean_deg": lean + (index + 1) * 0.02,
                        "attack_angle_deg": club.attack_angle_deg - (index + 1) * 0.03,
                        "carry_x_m": ""
                        if is_invalid
                        else target - index * 0.2 + lean * 0.01,
                        "carry_y_m": "" if is_invalid else (index - 1) * 0.4,
                    }
                )
    hashes = {
        filename: hashlib.sha256((directory / filename).read_bytes()).hexdigest()
        for filename in ("run_start.json", "summary.json", "shots.csv.gz")
    }
    (directory / "manifest.json").write_text(
        json.dumps({"schema": "matched-shot-study/1", "files_sha256": hashes})
    )
    plan = {
        "schema": "matched-shot-matrix/1",
        "stage": "supplement_final",
        "execution": execution,
        "supplement_protocol": protocol,
        "selected_cell_names": [cell],
        "cells": {cell: config},
    }
    plan_path = root / "matrix_plan.json"
    plan_path.write_text(json.dumps(plan))
    return {
        "cell": cell,
        "source_root": str(root),
        "source_protocol": protocol,
        "execution_kind": "unique",
        "source_plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "source_cell_manifest_sha256": hashlib.sha256(
            (directory / "manifest.json").read_bytes()
        ).hexdigest(),
        "source_csv_sha256": hashlib.sha256(
            (directory / "shots.csv.gz").read_bytes()
        ).hexdigest(),
    }


def _stub_scorer(endpoints, **kwargs):
    return {
        "scenario": "fixture",
        "baseline": {"table_sha256": "source"},
        "api_evaluated_states": 1,
        "patterns": {
            "Straight": {"mean_strokes_gained": 0.0},
            "Draw": {"mean_strokes_gained": 0.1},
        },
        "paired_benefit_vs_straight": {
            "Draw": {"estimate": 0.1, "lower_95": 0.0, "upper_95": 0.2}
        },
    }


def test_explicit_pair_centers_nominal_delivery_and_uses_six_population_intersection(
    tmp_path, monkeypatch
) -> None:
    from src.tools.shot_pattern_analysis import matched_lean_control as module

    monkeypatch.setattr(module, "score_course_endpoints", _stub_scorer)
    before = _write_cell(
        tmp_path / "pw0",
        lean=0,
        target_request=None,
        protocol="pw-primary-face-bounds/1",
        invalid=("Draw", 3),
    )
    after = _write_cell(
        tmp_path / "pw5",
        lean=5,
        target_request=77.40247724866299,
        protocol="pw-control-face-bounds/1",
        invalid=("Fade", 2),
    )

    result = module.analyze_explicit_source_pair(
        before,
        after,
        comparison_kind="across_lean",
        target_x_m=77.40247724866299,
    )

    assert result["schema"] == "matched-explicit-source-pair/1"
    assert result["status"] == "conditional_only"
    assert result["common_supported_n"] == 2
    assert result["conditioning"] == "same_indices_supported_in_all_six_populations"
    assert result["input_error_max_abs_difference"] <= 1e-12
    assert result["source_protocols"] == [
        "pw-primary-face-bounds/1",
        "pw-control-face-bounds/1",
    ]
    assert set(result["patterns"]) == set(PATTERNS)
    assert result["patterns"]["Draw"]["physical_pattern"] == "Draw"
    assert result["patterns"]["Draw"]["course_sg_difference"]["estimate"] == 0.1


def test_explicit_pair_rejects_unapproved_physical_config_difference(tmp_path) -> None:
    from src.tools.shot_pattern_analysis.matched_lean_control import (
        analyze_explicit_source_pair,
    )

    before = _write_cell(
        tmp_path / "pw0",
        lean=0,
        target_request=None,
        protocol="pw-primary-face-bounds/1",
    )
    after = _write_cell(
        tmp_path / "pw5",
        lean=5,
        target_request=77.40247724866299,
        protocol="pw-control-face-bounds/1",
    )
    plan_path = Path(after["source_root"]) / "matrix_plan.json"
    plan = json.loads(plan_path.read_text())
    cell_name = after["cell"]
    plan["cells"][cell_name]["club"]["loft_deg"] += 1
    (Path(after["source_root"]) / cell_name / "run_start.json").write_text(
        json.dumps({"config": plan["cells"][cell_name], "execution": plan["execution"]})
    )
    (Path(after["source_root"]) / cell_name / "summary.json").write_text(
        json.dumps(
            {
                "config": plan["cells"][cell_name],
                "status": "completed",
                "target_x_m": 77.40247724866299,
            }
        )
    )
    (Path(after["source_root"]) / cell_name / "manifest.json").write_text(
        json.dumps(
            {
                "files_sha256": {
                    filename: hashlib.sha256(
                        (Path(after["source_root"]) / cell_name / filename).read_bytes()
                    ).hexdigest()
                    for filename in ("run_start.json", "summary.json", "shots.csv.gz")
                }
            }
        )
    )
    plan_path.write_text(json.dumps(plan))
    after["source_plan_sha256"] = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    after["source_cell_manifest_sha256"] = hashlib.sha256(
        (Path(after["source_root"]) / cell_name / "manifest.json").read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="config differences"):
        analyze_explicit_source_pair(
            before, after, comparison_kind="across_lean", target_x_m=77.40247724866299
        )


def test_explicit_pair_rejects_centered_input_mismatch(tmp_path) -> None:
    from src.tools.shot_pattern_analysis.matched_lean_control import (
        analyze_explicit_source_pair,
    )

    before = _write_cell(
        tmp_path / "pw0",
        lean=0,
        target_request=None,
        protocol="pw-primary-face-bounds/1",
    )
    after = _write_cell(
        tmp_path / "pw5",
        lean=5,
        target_request=77.40247724866299,
        protocol="pw-control-face-bounds/1",
    )
    cell_dir = Path(after["source_root"]) / after["cell"]
    path = cell_dir / "shots.csv.gz"
    with gzip.open(path, "rt", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["path_deg"] = str(float(rows[0]["path_deg"]) + 1e-6)
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    manifest = json.loads((cell_dir / "manifest.json").read_text())
    manifest["files_sha256"]["shots.csv.gz"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    (cell_dir / "manifest.json").write_text(json.dumps(manifest))
    after["source_csv_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    after["source_cell_manifest_sha256"] = hashlib.sha256(
        (cell_dir / "manifest.json").read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="primitive input error mismatch"):
        analyze_explicit_source_pair(
            before, after, comparison_kind="across_lean", target_x_m=77.40247724866299
        )
