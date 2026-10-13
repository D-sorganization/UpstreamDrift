"""Second-pass review fixes for the shot-pattern study (#12409).

Covers publication races (check-then-replace overwrote files that appeared
in between), partial bundle scoring, late config validation and numpy
integer grid sizes.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.tools.shot_pattern_analysis import (
    matched_lean_control,
    matched_matrix,
    matched_sensitivity,
    provenance,
    scenario_scoring,
)
from src.tools.shot_pattern_analysis.density import shared_histogram_density
from src.tools.shot_pattern_analysis.matched_config import matched_preset
from src.tools.shot_pattern_analysis.matched_runner import ExperimentConfig
from src.tools.shot_pattern_analysis.noise import NoiseSpec

pytestmark = pytest.mark.unit

_NOISE = NoiseSpec((1, 0, 0, 0, 0, 0, 0))


# --- One exclusive, atomic publisher ------------------------------------------


def test_publish_new_text_writes_exact_lf_bytes(tmp_path):
    destination = tmp_path / "nested" / "report.json"
    assert provenance.publish_new_text(destination, "a\nb\n") == destination
    assert destination.read_bytes() == b"a\nb\n"
    assert [p.name for p in destination.parent.iterdir()] == ["report.json"]


def test_publish_new_text_never_replaces_an_existing_file(tmp_path):
    destination = tmp_path / "report.json"
    destination.write_text("original")
    with pytest.raises(FileExistsError):
        provenance.publish_new_text(destination, "replacement")
    assert destination.read_text() == "original"
    assert [p.name for p in tmp_path.iterdir()] == ["report.json"]


def test_lean_report_is_not_overwritten_by_a_late_arrival(tmp_path, monkeypatch):
    matrix_root = tmp_path / "matrix"
    matrix_root.mkdir()
    (matrix_root / "matrix_plan.json").write_text(json.dumps({"cells": {}}))
    destination = tmp_path / "reports" / "lean.json"

    def analyze(_root):
        destination.parent.mkdir(parents=True)
        destination.write_text("someone else's report")
        return {"status": "available"}

    monkeypatch.setattr(matched_lean_control, "analyze_across_lean", analyze)
    with pytest.raises(FileExistsError):
        matched_lean_control.write_across_lean_report(matrix_root, destination)
    assert destination.read_text() == "someone else's report"


def test_sensitivity_result_is_not_overwritten_by_a_late_arrival(tmp_path, monkeypatch):
    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()
    output_dir = tmp_path / "out"
    destination = output_dir / "cell.course_sensitivity.json"

    def analyze(_cell, _plan):
        output_dir.mkdir()
        destination.write_text("someone else's result")
        return {"status": "available"}

    monkeypatch.setattr(matched_sensitivity, "analyze_saved_sensitivity", analyze)
    with pytest.raises(FileExistsError):
        matched_sensitivity.write_saved_sensitivity(
            cell_dir, tmp_path / "plan.json", output_dir
        )
    assert destination.read_text() == "someone else's result"


# --- Matrix plan publication is race-safe -------------------------------------


def _cells():
    request = ExperimentConfig(matched_preset("seven_iron"), _NOISE, n_shots=4)
    return (("seven_iron_cell", request),)


def _publish(root, stage="pilot"):
    matched_matrix._publish_matrix_plan(
        root, _cells(), stage, {"source": "x"}, matched_matrix.MatrixSelection()
    )


def test_matrix_plan_is_published_once_and_reused_when_identical(tmp_path):
    _publish(tmp_path)
    first = (tmp_path / "matrix_plan.json").read_bytes()
    _publish(tmp_path)
    assert (tmp_path / "matrix_plan.json").read_bytes() == first
    assert b"\r\n" not in first


def test_matrix_plan_written_concurrently_with_other_content_is_rejected(
    tmp_path, monkeypatch
):
    real_publish = provenance.publish_new_text

    def racing_publish(destination, text):
        Path(destination).write_text(json.dumps({"stage": "other shard"}))
        return real_publish(destination, text)

    monkeypatch.setattr(matched_matrix, "publish_new_text", racing_publish)
    with pytest.raises(RuntimeError, match="differs"):
        _publish(tmp_path)


# --- Bundle scoring fails before it writes anything ---------------------------


def test_bundle_scoring_failure_leaves_the_bundle_untouched(tmp_path, monkeypatch):
    summary = {"config": {"club_id": "seven_iron", "seed": 1}, "target_x_m": 150}
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    (tmp_path / "receipt.json").write_text(json.dumps({"run": "frozen"}))
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}

    class Baseline:
        def model_dump(self, mode):
            return {"knots": []}

    def failing_git(*_args, **_kwargs):
        raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr(scenario_scoring, "load_pattern_endpoints", lambda _p: {})
    monkeypatch.setattr(
        scenario_scoring, "score_course_endpoints", lambda *a, **k: {"status": "ok"}
    )
    monkeypatch.setattr(scenario_scoring, "_add_sensitivities", lambda *a, **k: None)
    monkeypatch.setattr(
        scenario_scoring, "build_broadie_approx_baseline", lambda: Baseline()
    )
    monkeypatch.setattr(scenario_scoring.subprocess, "run", failing_git)

    with pytest.raises(subprocess.CalledProcessError):
        scenario_scoring.score_corrected_bundle(tmp_path)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


# --- Experiment requests fail when built, not mid-run -------------------------


@pytest.mark.parametrize("field", ["club", "noise"])
def test_experiment_config_rejects_wrong_component_types(field):
    kwargs = {"club": matched_preset("seven_iron"), "noise": _NOISE, field: {}}
    with pytest.raises(TypeError, match=field):
        ExperimentConfig(**kwargs)


def test_experiment_config_rejects_speed_bounds_beyond_the_club_limit():
    club = replace(matched_preset("seven_iron"), club_speed_mps=60.0)
    with pytest.raises(ValueError, match="100 m/s"):
        ExperimentConfig(club, _NOISE, match_speed_factors=(0.75, 1.8))


# --- Grid sizes accept numpy integers but not booleans ------------------------


def _points():
    rng = np.random.default_rng(1)
    return {"Straight": rng.normal(size=(50, 2)), "Draw": rng.normal(size=(50, 2))}


def test_density_accepts_numpy_integer_bin_counts():
    result = shared_histogram_density(_points(), bins=np.int64(8))
    reference = shared_histogram_density(_points(), bins=8)
    np.testing.assert_array_equal(result.x_edges_m, reference.x_edges_m)
    np.testing.assert_array_equal(result.y_edges_m, reference.y_edges_m)
    assert result.density_vmax_per_m2 == reference.density_vmax_per_m2
    assert len(result.x_edges_m) == 9


def test_density_rejects_boolean_bin_counts():
    with pytest.raises(ValueError, match="bin count"):
        shared_histogram_density(_points(), bins=(True, 8))
