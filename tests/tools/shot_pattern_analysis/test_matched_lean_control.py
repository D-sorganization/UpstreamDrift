"""Across-lean comparisons retain common indices and source-backed scoring."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from dataclasses import asdict, replace

import pytest

from src.tools.shot_pattern_analysis import matched_lean_control as module
from src.tools.shot_pattern_analysis.matched_config import matched_preset
from src.tools.shot_pattern_analysis.matched_runner import ExperimentConfig
from src.tools.shot_pattern_analysis.noise import NoiseSpec

pytestmark = pytest.mark.unit


def saved_lean_plan(root, *, unavailable_ten=True):
    execution = {"source_sha256": {"frozen.py": "abc"}}
    cells = {}
    for lean in (0, 5, 10):
        club = replace(matched_preset("seven_iron"), shaft_lean_deg=lean)
        request = ExperimentConfig(
            club, NoiseSpec((1, 0, 0, 0, 0, 0, 0)), n_shots=4, target_x_m=180.0
        )
        name = f"seven_iron_lean{lean}_curve1_rotation1_shaft_only"
        config = json.loads(json.dumps(asdict(request)))
        cells[name] = config
        directory = root / name
        directory.mkdir()
        (directory / "run_start.json").write_text(
            json.dumps({"config": config, "execution": execution})
        )
        if lean == 10 and unavailable_ten:
            summary = {
                "status": "unavailable",
                "config": config,
                "reason": "nominal_unreachable",
            }
        else:
            summary = {"status": "completed", "config": config, "target_x_m": 180.0}
            with gzip.open(directory / "shots.csv.gz", "wt", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "pattern",
                        "shot_index",
                        "status",
                        "carry_x_m",
                        "carry_y_m",
                    ),
                )
                writer.writeheader()
                for pattern in ("Straight", "Draw", "Fade"):
                    for index in range(4):
                        supported = not (lean == 5 and pattern == "Draw" and index == 3)
                        writer.writerow(
                            {
                                "pattern": pattern,
                                "shot_index": index,
                                "status": "supported" if supported else "off_face",
                                "carry_x_m": 180 + lean / 5 + index
                                if supported
                                else "",
                                "carry_y_m": index - 1 if supported else "",
                            }
                        )
        (directory / "summary.json").write_text(json.dumps(summary))
        hashes = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.iterdir()
        }
        (directory / "manifest.json").write_text(json.dumps({"files_sha256": hashes}))
    plan = root / "matrix_plan.json"
    plan.write_text(
        json.dumps({"stage": "pilot", "execution": execution, "cells": cells})
    )
    return plan


def test_joint_indices_and_unavailable_lean_pairs(tmp_path, monkeypatch):
    saved_lean_plan(tmp_path)
    calls = []

    def score(endpoints, **kwargs):
        calls.append((endpoints, kwargs))
        return {
            "status": "available",
            "scenario": "assumed flat course",
            "baseline": {"table_sha256": "source"},
            "api_evaluated_states": 3,
            "patterns": {
                "Straight": {"mean_strokes_gained": 0.0},
                "Draw": {"mean_strokes_gained": 0.1},
            },
            "paired_benefit_vs_straight": {
                "Draw": {"estimate": 0.1, "lower_95": 0.05, "upper_95": 0.15}
            },
        }

    monkeypatch.setattr(module, "score_course_endpoints", score)
    report = module.analyze_across_lean(tmp_path)
    assert report["schema"] == "matched-across-lean/1"
    assert report["population"]["planned_pairs"] == 3
    pairs = {tuple(pair["leans_deg"]): pair for pair in report["pairs"]}
    assert pairs[(0, 5)]["status"] == "conditional_only"
    assert pairs[(0, 5)]["common_supported_n"] == 3
    assert (
        pairs[(0, 5)]["patterns"]["Straight"]["course_sg_difference"]["estimate"] == 0.1
    )
    assert pairs[(0, 5)]["patterns"]["Straight"]["physical_pattern"] == "Straight"
    assert pairs[(0, 5)]["patterns"]["Straight"]["earlier_lean_deg"] == 0
    assert pairs[(0, 5)]["patterns"]["Straight"]["later_lean_deg"] == 5
    assert pairs[(0, 10)]["status"] == "unavailable"
    assert pairs[(5, 10)]["status"] == "unavailable"
    assert len(calls) == 3
    assert all(
        [i for i, _, _ in endpoints["Straight"]] == [0, 1, 2] for endpoints, _ in calls
    )
    assert all(
        [i for i, _, _ in endpoints["Draw"]] == [0, 1, 2] for endpoints, _ in calls
    )
    assert report["input_sha256"]["matrix_plan.json"]


def test_common_target_is_required_and_manifest_drift_rejected(tmp_path, monkeypatch):
    plan = saved_lean_plan(tmp_path)
    data = json.loads(plan.read_text())
    name = next(iter(data["cells"]))
    (tmp_path / name / "summary.json").write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        module.analyze_across_lean(tmp_path)
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    plan = saved_lean_plan(fresh)
    data = json.loads(plan.read_text())
    for config in data["cells"].values():
        config["target_x_m"] = None
    plan.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="common target"):
        module.analyze_across_lean(fresh)


def test_separate_atomic_report_never_overwrites_frozen_cells(tmp_path, monkeypatch):
    saved_lean_plan(tmp_path)
    monkeypatch.setattr(
        module,
        "score_course_endpoints",
        lambda *a, **k: {
            "scenario": "flat",
            "baseline": {"table_sha256": "source"},
            "api_evaluated_states": 3,
            "patterns": {
                "Straight": {"mean_strokes_gained": 0.0},
                "Draw": {"mean_strokes_gained": 0.1},
            },
            "paired_benefit_vs_straight": {"Draw": {"estimate": 0.1}},
        },
    )
    frozen = next(tmp_path.glob("*/manifest.json"))
    before = hashlib.sha256(frozen.read_bytes()).hexdigest()
    destination = tmp_path / "reports" / "lean_comparison.json"
    assert module.write_across_lean_report(tmp_path, destination) == destination
    assert json.loads(destination.read_text())["schema"] == "matched-across-lean/1"
    assert hashlib.sha256(frozen.read_bytes()).hexdigest() == before
    with pytest.raises(FileExistsError):
        module.write_across_lean_report(tmp_path, destination)


@pytest.mark.integration
def test_real_public_course_scoring_uses_paired_lean_rows(tmp_path):
    import numpy as np
    from src.tools.shot_pattern_analysis.scoring import (
        YARDS_PER_METRE,
        build_broadie_approx_baseline,
    )
    from src.tools.shot_pattern_analysis.scoring_cache import source_backed_score_cache

    saved_lean_plan(tmp_path, unavailable_ten=False)
    report = module.analyze_across_lean(tmp_path)
    pair = next(pair for pair in report["pairs"] if pair["leans_deg"] == [0, 5])
    assert pair["patterns"]["Draw"]["course_sg_difference"]["lower_95"] is not None
    assert pair["patterns"]["Fade"]["course_sg_difference"]["upper_95"] is not None
    baseline = build_broadie_approx_baseline()
    cache = source_backed_score_cache(
        baseline,
        start_lie="fairway",
        start_distance_yards=180 * YARDS_PER_METRE,
        finish_lies=("green", "rough"),
    )
    before = np.array([1.0, 1.0, np.hypot(2.0, 1.0)]) * YARDS_PER_METRE
    after = np.array([np.hypot(1.0, 1.0), 2.0, np.hypot(3.0, 1.0)]) * YARDS_PER_METRE
    expected = float(
        np.mean(cache.score_many("green", after) - cache.score_many("green", before))
    )
    assert pair["patterns"]["Straight"]["course_sg_difference"][
        "estimate"
    ] == pytest.approx(expected, abs=1e-12)
