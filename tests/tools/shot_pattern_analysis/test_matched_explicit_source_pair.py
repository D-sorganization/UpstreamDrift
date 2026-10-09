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


def test_comparison_normalizes_only_documented_legacy_bound_defaults() -> None:
    from src.tools.shot_pattern_analysis.matched_lean_control import _comparison_config

    legacy = {"club": {"club_id": "driver"}}
    normalized = _comparison_config(legacy)
    assert "match_face_bounds_deg" not in legacy
    assert (
        normalized["match_face_bounds_deg"]
        == "legacy_reference_centered_plus_minus_3_deg"
    )
    assert normalized["match_speed_factors"] == [0.75, 1.25]


def test_rotation_sd_interaction_requires_ascending_source_order() -> None:
    from src.tools.shot_pattern_analysis.matched_lean_control import (
        _require_rotation_sd_order,
    )

    _require_rotation_sd_order(1.0, 2.0)
    with pytest.raises(ValueError, match="order 1 degree before 2"):
        _require_rotation_sd_order(2.0, 1.0)


def test_driver_speed_bound_exception_is_protocol_specific() -> None:
    from src.tools.shot_pattern_analysis.matched_lean_control import (
        _is_reviewed_driver_speed_control,
    )

    first = {
        "club": {"club_id": "driver", "shaft_lean_deg": 0.0},
        "match_speed_factors": [0.75, 1.25],
    }
    second = {
        "club": {"club_id": "driver", "shaft_lean_deg": 5.0},
        "match_speed_factors": [0.75, 1.30],
    }
    old_plan = {"supplement_protocol": None}
    supplement_plan = {"supplement_protocol": "driver-control-speed-bounds/1"}
    assert _is_reviewed_driver_speed_control(first, second, old_plan, supplement_plan)
    assert not _is_reviewed_driver_speed_control(
        second, first, supplement_plan, old_plan
    )
    second["match_speed_factors"] = [0.75, 1.35]
    assert not _is_reviewed_driver_speed_control(
        first, second, old_plan, supplement_plan
    )


def test_cross_snapshot_requires_hash_bound_accepted_transition(tmp_path) -> None:
    from src.tools.shot_pattern_analysis.matched_lean_control import (
        _execution_transition,
    )

    old = {
        "source_sha256": {"src/tools/shot_pattern_analysis/matched_runner.py": "a"},
        "native_binary_sha256": "same",
    }
    new = {
        "source_sha256": {"src/tools/shot_pattern_analysis/matched_runner.py": "b"},
        "native_binary_sha256": "same",
    }
    with pytest.raises(ValueError, match="requires an accepted source-transition"):
        _execution_transition(old, new, None, None)

    evidence = tmp_path / "review.json"
    evidence.write_text('{"approved": true}\n')
    receipt = tmp_path / "transition.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "matched-delivery-source-transition/1",
                "status": "independently_accepted",
                "qualified_scope": "matched-delivery-comparisons",
                "old_execution": old,
                "new_execution": new,
                "changed_sources": {
                    "src/tools/shot_pattern_analysis/matched_runner.py": {
                        "old_sha256": "a",
                        "new_sha256": "b",
                        "classification": "approved_nonprojection",
                    }
                },
                "evidence_files": {
                    "review.json": hashlib.sha256(evidence.read_bytes()).hexdigest()
                },
            }
        )
        + "\n"
    )
    receipt_sha = hashlib.sha256(receipt.read_bytes()).hexdigest()
    assert _execution_transition(old, new, receipt, receipt_sha) == receipt_sha

    tampered = json.loads(receipt.read_text())
    tampered["changed_sources"][
        "src/tools/shot_pattern_analysis/matched_physics.py"
    ] = {
        "old_sha256": "a",
        "new_sha256": "b",
        "classification": "approved_nonprojection",
    }
    receipt.write_text(json.dumps(tampered) + "\n")
    with pytest.raises(ValueError, match="receipt hash mismatch"):
        _execution_transition(old, new, receipt, receipt_sha)


def test_across_lean_pair_supports_5_to_10_degrees(tmp_path, monkeypatch) -> None:
    from src.tools.shot_pattern_analysis import matched_lean_control as module

    monkeypatch.setattr(module, "score_course_endpoints", _stub_scorer)
    before = _write_cell(
        tmp_path / "lean5",
        lean=5,
        target_request=77.40247724866299,
        protocol="pw-control-face-bounds/1",
    )
    after = _write_cell(
        tmp_path / "lean10",
        lean=10,
        target_request=77.40247724866299,
        protocol="pw-control-face-bounds/1",
    )
    result = module.analyze_explicit_source_pair(
        before,
        after,
        comparison_kind="across_lean",
        target_x_m=77.40247724866299,
    )
    assert result["patterns"]["Draw"]["first_condition"]["shaft_lean_deg"] == 5
    assert result["patterns"]["Draw"]["second_condition"]["shaft_lean_deg"] == 10


def test_rotation_sd_pair_reports_common_index_sg_difference_of_differences(
    tmp_path, monkeypatch
) -> None:
    from types import SimpleNamespace

    from src.tools.shot_pattern_analysis import matched_lean_control as module

    monkeypatch.setattr(module, "score_course_endpoints", _stub_scorer)
    first = _write_cell(
        tmp_path / "sd1",
        lean=0,
        target_request=None,
        protocol="pw-primary-face-bounds/1",
        rotation_sd=1.0,
    )
    second = _write_cell(
        tmp_path / "sd2",
        lean=0,
        target_request=None,
        protocol="pw-primary-face-bounds/1",
        rotation_sd=2.0,
    )
    vectors = [
        {
            "Straight": np.array([10.0, 11.0, 12.0, 13.0]),
            "Draw": np.array([11.0, 12.0, 13.0, 14.0]),
            "Fade": np.array([9.0, 10.0, 11.0, 12.0]),
        },
        {
            "Straight": np.array([10.0, 11.0, 12.0, 13.0]),
            "Draw": np.array([13.0, 14.0, 15.0, 16.0]),
            "Fade": np.array([8.0, 9.0, 10.0, 11.0]),
        },
    ]
    calls = 0

    def score_vectors(endpoints, **kwargs):
        nonlocal calls
        result = SimpleNamespace(
            scores=vectors[calls],
            scenario_id="approach_green_rough",
            baseline={"id": "fixture"},
            api_evaluated_states=10,
        )
        calls += 1
        return result

    monkeypatch.setattr(module, "score_course_endpoint_vectors", score_vectors)
    result = module.analyze_explicit_source_pair(
        first,
        second,
        comparison_kind="shaft_rotation_sd",
        target_x_m=77.40247724866299,
    )
    assert result["course_scoring_effect_modification"]["Draw"]["estimate"] == 2.0
    assert result["course_scoring_effect_modification"]["Fade"]["estimate"] == -1.0
    assert result["common_supported_n"] == 4
    assert "same six-population" in result["course_scoring_effect_modification_scope"]


def test_curve_strength_pair_reports_paired_sg_interaction(
    tmp_path, monkeypatch
) -> None:
    from types import SimpleNamespace

    from src.tools.shot_pattern_analysis import matched_lean_control as module

    monkeypatch.setattr(module, "score_course_endpoints", _stub_scorer)
    first = _write_cell(
        tmp_path / "curve1",
        lean=0,
        target_request=None,
        protocol="original-primary/1",
        curve_scale=1.0,
    )
    second = _write_cell(
        tmp_path / "curve2",
        lean=0,
        target_request=None,
        protocol="pw-primary-face-bounds/1",
        curve_scale=2.0,
    )
    vectors = [
        {
            "Straight": np.array([10.0, 10.0, 10.0, 10.0]),
            "Draw": np.array([11.0, 11.0, 11.0, 11.0]),
            "Fade": np.array([9.0, 9.0, 9.0, 9.0]),
        },
        {
            "Straight": np.array([10.0, 10.0, 10.0, 10.0]),
            "Draw": np.array([12.0, 12.0, 12.0, 12.0]),
            "Fade": np.array([8.0, 8.0, 8.0, 8.0]),
        },
    ]
    calls = 0

    def score_vectors(endpoints, **kwargs):
        nonlocal calls
        result = SimpleNamespace(
            scores=vectors[calls],
            scenario_id="approach_green_rough",
            baseline={"id": "fixture"},
            api_evaluated_states=10,
        )
        calls += 1
        return result

    monkeypatch.setattr(module, "score_course_endpoint_vectors", score_vectors)
    result = module.analyze_explicit_source_pair(
        first,
        second,
        comparison_kind="curve_strength",
        target_x_m=77.40247724866299,
    )
    assert result["course_scoring_effect_modification"]["Draw"]["estimate"] == 1.0
    assert result["course_scoring_effect_modification"]["Fade"]["estimate"] == -1.0


def _write_cell(
    root: Path,
    *,
    lean: float,
    target_request: float | None,
    protocol: str,
    invalid: tuple[str, int] | None = None,
    rotation_sd: float = 1.0,
    curve_scale: float = 2.0,
    club_id: str = "pitching_wedge",
    speed_factors: tuple[float, float] = (0.75, 1.25),
) -> dict:
    root.mkdir(parents=True)
    club = replace(matched_preset(club_id), shaft_lean_deg=lean)
    config = {
        "club": asdict(club),
        "noise": asdict(
            NoiseSpec((rotation_sd, 0.5, 0.44704, 0.00635, 0.00635, 0.0, 0.0))
        ),
        "n_shots": 4,
        "seed": 20261009,
        "curve_scale": curve_scale,
        "target_x_m": target_request,
        "match_tolerance_m": 0.05,
        "target_radius_m": 15.0,
        "match_face_bounds_deg": [-6.0, 6.0],
        "match_speed_factors": list(speed_factors),
    }
    cell = f"{club_id}_lean{lean:g}_curve{curve_scale:g}_rotation{rotation_sd:g}_shaft_only"
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
                        "shaft_rotation_deg": (index + 1) * 0.1 * rotation_sd,
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
        "supplement_protocol": None if protocol == "original_primary/1" else protocol,
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


def test_reviewed_driver_speed_bound_pair_passes_full_explicit_api(tmp_path) -> None:
    from src.tools.shot_pattern_analysis.matched_lean_control import (
        analyze_explicit_source_pair,
    )

    before = _write_cell(
        tmp_path / "driver0",
        lean=0,
        target_request=None,
        protocol="original_primary/1",
        club_id="driver",
        speed_factors=(0.75, 1.25),
    )
    after = _write_cell(
        tmp_path / "driver5",
        lean=5,
        target_request=77.40247724866299,
        protocol="driver-control-speed-bounds/1",
        club_id="driver",
        speed_factors=(0.75, 1.30),
    )
    result = analyze_explicit_source_pair(
        before,
        after,
        comparison_kind="across_lean",
        target_x_m=77.40247724866299,
    )
    assert result["comparison_kind"] == "across_lean"
    assert result["allowed_config_differences"] == [
        "club.shaft_lean_deg",
        "match_speed_factors",
        "target_x_m",
    ]


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


def test_explicit_pair_rechecks_source_bytes_after_analysis(
    tmp_path, monkeypatch
) -> None:
    from src.tools.shot_pattern_analysis import matched_lean_control as module

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
    original = module._paired_pattern_metrics
    changed = False

    def tampering_metrics(*args, **kwargs):
        nonlocal changed
        result = original(*args, **kwargs)
        if not changed:
            csv_path = Path(after["source_root"]) / after["cell"] / "shots.csv.gz"
            csv_path.write_bytes(csv_path.read_bytes() + b"t")
            changed = True
        return result

    monkeypatch.setattr(module, "_paired_pattern_metrics", tampering_metrics)
    with pytest.raises(ValueError, match="changed during analysis"):
        module.analyze_explicit_source_pair(
            before,
            after,
            comparison_kind="across_lean",
            target_x_m=77.40247724866299,
        )


def test_explicit_pair_rejects_coherently_valid_different_execution(tmp_path) -> None:
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
    root = Path(after["source_root"])
    cell_dir = root / after["cell"]
    plan_path = root / "matrix_plan.json"
    plan = json.loads(plan_path.read_text())
    execution = {"source_sha256": {"frozen.py": "different-but-coherent"}}
    plan["execution"] = execution
    run_start_path = cell_dir / "run_start.json"
    run_start = json.loads(run_start_path.read_text())
    run_start["execution"] = execution
    run_start_path.write_text(json.dumps(run_start))
    manifest_path = cell_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files_sha256"]["run_start.json"] = hashlib.sha256(
        run_start_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    plan_path.write_text(json.dumps(plan))
    after["source_plan_sha256"] = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    after["source_cell_manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="accepted source-transition receipt"):
        analyze_explicit_source_pair(
            before,
            after,
            comparison_kind="across_lean",
            target_x_m=77.40247724866299,
        )


@pytest.mark.parametrize("change", ["protected_source", "native_runtime"])
def test_full_pair_api_rejects_coherently_rehashed_protected_execution(
    tmp_path, change
) -> None:
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
    before_plan = json.loads(
        (Path(before["source_root"]) / "matrix_plan.json").read_text()
    )
    old_execution = before_plan["execution"]
    new_execution = json.loads(json.dumps(old_execution))
    changed_sources = {}
    if change == "protected_source":
        path = "src/shared/python/physics/contact.py"
        old_hash = new_execution["source_sha256"].get(path)
        new_hash = "coherently-rehashed-contact-source"
        new_execution["source_sha256"][path] = new_hash
        changed_sources[path] = {
            "old_sha256": old_hash,
            "new_sha256": new_hash,
            "classification": "approved_nonprojection",
        }
    else:
        new_execution["native_binary_sha256"] = "different-native-binary"

    root = Path(after["source_root"])
    plan_path = root / "matrix_plan.json"
    plan = json.loads(plan_path.read_text())
    plan["execution"] = new_execution
    plan_path.write_text(json.dumps(plan))
    cell_dir = root / after["cell"]
    run_start_path = cell_dir / "run_start.json"
    run_start = json.loads(run_start_path.read_text())
    run_start["execution"] = new_execution
    run_start_path.write_text(json.dumps(run_start))
    manifest_path = cell_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files_sha256"]["run_start.json"] = hashlib.sha256(
        run_start_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    after["source_plan_sha256"] = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    after["source_cell_manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()

    evidence = tmp_path / "review.json"
    evidence.write_text('{"accepted": true}\n')
    receipt = tmp_path / "transition.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "matched-delivery-source-transition/1",
                "status": "independently_accepted",
                "qualified_scope": "matched-delivery-comparisons",
                "old_execution": old_execution,
                "new_execution": new_execution,
                "changed_sources": changed_sources,
                "evidence_files": {
                    evidence.name: hashlib.sha256(evidence.read_bytes()).hexdigest()
                },
            }
        )
        + "\n"
    )
    receipt_hash = hashlib.sha256(receipt.read_bytes()).hexdigest()
    expected = (
        "unapproved source change"
        if change == "protected_source"
        else "runtime or backend"
    )
    with pytest.raises(ValueError, match=expected):
        analyze_explicit_source_pair(
            before,
            after,
            comparison_kind="across_lean",
            target_x_m=77.40247724866299,
            transition_receipt_path=receipt,
            transition_receipt_sha256=receipt_hash,
        )


def test_full_pair_api_rechecks_transition_evidence_after_calculation(
    tmp_path, monkeypatch
) -> None:
    from src.tools.shot_pattern_analysis import matched_lean_control as module

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
    old_execution = json.loads(
        (Path(before["source_root"]) / "matrix_plan.json").read_text()
    )["execution"]
    new_execution = json.loads(json.dumps(old_execution))
    source_path = "src/tools/shot_pattern_analysis/matched_runner.py"
    new_execution["source_sha256"][source_path] = "approved-new-hash"
    second_root = Path(after["source_root"])
    plan_path = second_root / "matrix_plan.json"
    plan = json.loads(plan_path.read_text())
    plan["execution"] = new_execution
    plan_path.write_text(json.dumps(plan))
    cell_dir = second_root / after["cell"]
    run_start_path = cell_dir / "run_start.json"
    run_start = json.loads(run_start_path.read_text())
    run_start["execution"] = new_execution
    run_start_path.write_text(json.dumps(run_start))
    manifest_path = cell_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files_sha256"]["run_start.json"] = hashlib.sha256(
        run_start_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    after["source_plan_sha256"] = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    after["source_cell_manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    evidence = tmp_path / "review.json"
    evidence.write_text("initial approved evidence\n")
    receipt = tmp_path / "transition.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "matched-delivery-source-transition/1",
                "status": "independently_accepted",
                "qualified_scope": "matched-delivery-comparisons",
                "old_execution": old_execution,
                "new_execution": new_execution,
                "changed_sources": {
                    source_path: {
                        "old_sha256": None,
                        "new_sha256": "approved-new-hash",
                        "classification": "approved_nonprojection",
                    }
                },
                "evidence_files": {
                    evidence.name: hashlib.sha256(evidence.read_bytes()).hexdigest()
                },
            }
        )
        + "\n"
    )
    receipt_hash = hashlib.sha256(receipt.read_bytes()).hexdigest()
    original_metrics = module._paired_pattern_metrics
    changed = False

    def mutate_evidence(*args, **kwargs):
        nonlocal changed
        result = original_metrics(*args, **kwargs)
        if not changed:
            evidence.write_text("evidence changed during analysis\n")
            changed = True
        return result

    monkeypatch.setattr(module, "_paired_pattern_metrics", mutate_evidence)
    with pytest.raises(ValueError, match="evidence file hash mismatch"):
        module.analyze_explicit_source_pair(
            before,
            after,
            comparison_kind="across_lean",
            target_x_m=77.40247724866299,
            transition_receipt_path=receipt,
            transition_receipt_sha256=receipt_hash,
        )


def test_full_pair_api_rechecks_course_score_helper_source_after_calculation(
    tmp_path, monkeypatch
) -> None:
    from src.tools.shot_pattern_analysis import matched_lean_control as module

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
    original_hashes = module._paired_scoring_source_hashes
    calls = 0

    def changed_hashes():
        nonlocal calls
        calls += 1
        values = original_hashes()
        if calls > 1:
            values["course_score_vectors.py"] = "changed-during-analysis"
        return values

    monkeypatch.setattr(module, "_paired_scoring_source_hashes", changed_hashes)
    with pytest.raises(RuntimeError, match="scoring source changed during analysis"):
        module.analyze_explicit_source_pair(
            before, after, comparison_kind="across_lean", target_x_m=77.40247724866299
        )
