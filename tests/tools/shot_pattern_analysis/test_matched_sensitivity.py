"""Saved-shot sensitivity retains only common supported pairs and provenance."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from dataclasses import asdict

import pytest

from src.tools.shot_pattern_analysis.matched_config import matched_preset
from src.tools.shot_pattern_analysis.matched_runner import ExperimentConfig
from src.tools.shot_pattern_analysis.noise import NoiseSpec
from src.tools.shot_pattern_analysis import matched_sensitivity as module

pytestmark = pytest.mark.unit


def saved_cell(root, club_id="driver"):
    request = ExperimentConfig(
        matched_preset(club_id), NoiseSpec((1, 0, 0, 0, 0, 0, 0)), n_shots=3
    )
    config = json.loads(json.dumps(asdict(request)))
    execution = {"source_sha256": {"frozen.py": "abc"}}
    name = f"{club_id}_cell"
    (root / "matrix_plan.json").write_text(
        json.dumps({"stage": "pilot", "execution": execution, "cells": {name: config}})
    )
    cell = root / name
    cell.mkdir()
    (cell / "run_start.json").write_text(
        json.dumps({"config": config, "execution": execution})
    )
    (cell / "summary.json").write_text(
        json.dumps({"status": "completed", "config": config, "target_x_m": 180.0})
    )
    with gzip.open(cell / "shots.csv.gz", "wt", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("pattern", "shot_index", "status", "carry_x_m", "carry_y_m"),
        )
        writer.writeheader()
        for pattern in ("Straight", "Draw", "Fade"):
            for index in range(3):
                supported = not (pattern == "Draw" and index == 2)
                writer.writerow(
                    {
                        "pattern": pattern,
                        "shot_index": index,
                        "status": "supported" if supported else "off_face",
                        "carry_x_m": 180 + index if supported else "",
                        "carry_y_m": (index - 1) * 2 if supported else "",
                    }
                )
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in cell.iterdir()
    }
    (cell / "manifest.json").write_text(json.dumps({"files_sha256": hashes}))
    return cell, root / "matrix_plan.json"


def test_driver_grid_scores_exact_common_pairs_and_inputs(tmp_path, monkeypatch):
    cell, plan = saved_cell(tmp_path)
    calls = []

    def score(endpoints, **kwargs):
        calls.append((endpoints, kwargs))
        return {
            "status": "available",
            "patterns": {
                name: {"n": 2, "mean_strokes_gained": 0.1} for name in endpoints
            },
            "paired_benefit_vs_straight": {
                "Draw": {"estimate": 0.0},
                "Fade": {"estimate": 0.0},
            },
            "baseline": {"table_sha256": "source"},
        }

    monkeypatch.setattr(module, "score_course_endpoints", score)
    report = module.analyze_saved_sensitivity(cell, plan)
    assert report["status"] == "conditional_only"
    assert report["population"]["total_per_pattern"] == 3
    assert report["population"]["common_supported_n"] == 2
    assert report["population"]["unsupported_by_pattern"]["Draw"] == 1
    assert report["population"]["supported_by_pattern"] == {
        "Straight": 3,
        "Draw": 2,
        "Fade": 3,
    }
    assert report["population"]["supported_but_unpaired_by_pattern"] == {
        "Straight": 1,
        "Draw": 0,
        "Fade": 1,
    }
    assert all(
        [i for i, _, _ in endpoints["Straight"]] == [0, 1] for endpoints, _ in calls
    )
    assert {
        (kwargs["fairway_half_width_m"], kwargs["hole_distance_m"])
        for _, kwargs in calls
    } == {(w, h) for w in (10.0, 15.0, 20.0) for h in (350.0, 400.0, 450.0)}
    assert len(calls) == 9
    assert report["input_sha256"]["manifest.json"]
    assert report["source_sha256"]["scenario_scoring.py"]
    assert report["scenarios"]["width15_hole400"]["paired_benefit_vs_straight"]


def test_iron_radius_cases_and_separate_output(tmp_path, monkeypatch):
    cell, plan = saved_cell(tmp_path, "seven_iron")
    calls = []

    def score(endpoints, **kwargs):
        calls.append(kwargs)
        return {
            "status": "available",
            "patterns": {},
            "paired_benefit_vs_straight": {},
            "baseline": {"table_sha256": "source"},
        }

    monkeypatch.setattr(module, "score_course_endpoints", score)
    output = module.write_saved_sensitivity(cell, plan, tmp_path / "separate")
    assert output.parent == tmp_path / "separate"
    assert {kwargs["green_radius_m"] for kwargs in calls} == {10.0, 15.0, 20.0}
    assert (
        json.loads(output.read_text())["scenarios"]["radius15"]["status"] == "available"
    )
    assert not (cell / "course_sensitivity.json").exists()


def test_manifest_and_plan_drift_fail_closed(tmp_path, monkeypatch):
    cell, plan = saved_cell(tmp_path)
    monkeypatch.setattr(
        module, "score_course_endpoints", lambda *a, **k: {"status": "available"}
    )
    (cell / "shots.csv.gz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="integrity"):
        module.analyze_saved_sensitivity(cell, plan)
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    cell, plan = saved_cell(fresh)
    data = json.loads(plan.read_text())
    data["execution"]["source_sha256"]["frozen.py"] = "changed"
    plan.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="source changed"):
        module.analyze_saved_sensitivity(cell, plan)


def test_saved_input_change_during_scoring_aborts(tmp_path, monkeypatch):
    cell, plan = saved_cell(tmp_path)

    def mutate_saved_input(endpoints, **kwargs):
        (cell / "summary.json").write_text("{}")
        return {"status": "available"}

    monkeypatch.setattr(module, "score_course_endpoints", mutate_saved_input)
    with pytest.raises(RuntimeError, match="saved input changed"):
        module.analyze_saved_sensitivity(cell, plan)


@pytest.mark.integration
def test_saved_iron_points_score_through_public_course_api(tmp_path):
    cell, plan = saved_cell(tmp_path, "seven_iron")
    report = module.analyze_saved_sensitivity(cell, plan)
    scenario = report["scenarios"]["radius15"]
    assert scenario["status"] == "available"
    assert scenario["scored_shots"] == 6
    assert scenario["patterns"]["Straight"]["n"] == 2
    assert set(scenario["paired_benefit_vs_straight"]) == {"Draw", "Fade"}


@pytest.mark.integration
def test_saved_driver_grid_scores_through_public_course_api(tmp_path):
    cell, plan = saved_cell(tmp_path, "driver")
    report = module.analyze_saved_sensitivity(cell, plan)
    assert len(report["scenarios"]) == 9
    assert all(case["status"] == "available" for case in report["scenarios"].values())
    assert all(case["scored_shots"] == 6 for case in report["scenarios"].values())
