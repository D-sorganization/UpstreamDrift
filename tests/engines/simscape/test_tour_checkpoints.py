"""Durable warm-start checkpoints for the native first-prefix experiment."""

import json
from pathlib import Path

import pytest

from src.engines.Simscape_Multibody_Models.python.tour_checkpoints import (
    snapshot_prefix_report,
    read_prefix_candidate,
)

pytestmark = pytest.mark.unit


def test_snapshot_retains_best_candidate_and_is_immutable(tmp_path: Path) -> None:
    source = tmp_path / "report.json"
    report = {
        "qualification": "exploratory-first-prefix-only",
        "effort_scales": [200, 1500],
        "evaluations": [
            {"number": 1, "rmse_m": 0.001, "efforts": [20, 300]},
            {"number": 2, "rmse_m": 0.003, "efforts": [40, 450]},
        ],
    }
    source.write_text(json.dumps(report), encoding="utf-8")
    saved = snapshot_prefix_report(source, tmp_path / "checkpoints")
    content = saved.read_bytes()
    data = json.loads(content)
    assert data["resume_parameters"] == pytest.approx([1.1, 1.2])
    assert data["best_evaluation"]["number"] == 1
    assert data["report"] == report
    assert snapshot_prefix_report(source, saved.parent) == saved
    assert saved.read_bytes() == content
    report["evaluations"].append({"number": 3, "rmse_m": 0.0005, "efforts": [10, 150]})
    source.write_text(json.dumps(report), encoding="utf-8")
    assert snapshot_prefix_report(source, saved.parent) != saved
    assert saved.read_bytes() == content


def test_partial_report_cannot_replace_checkpoint(tmp_path: Path) -> None:
    source = tmp_path / "report.json"
    source.write_text('{"evaluations":', encoding="utf-8")
    with pytest.raises(ValueError):
        snapshot_prefix_report(source, tmp_path / "checkpoints")
    assert not (tmp_path / "checkpoints").exists()


def test_resume_checks_capture_marker_order_and_effort_units(tmp_path: Path) -> None:
    source = tmp_path / "report.json"
    report = {
        "qualification": "exploratory-first-prefix-only",
        "source_sha256": "capture-a",
        "labels": ["waist", "club"],
        "effort_scales": [200],
        "evaluations": [{"number": 1, "rmse_m": 0.001, "efforts": [20]}],
    }
    source.write_text(json.dumps(report), encoding="utf-8")
    saved = snapshot_prefix_report(source, tmp_path / "checkpoints")
    assert read_prefix_candidate(saved, report) == pytest.approx([1.1])
    for changed in [
        {"source_sha256": "capture-b"},
        {"labels": ["club", "waist"]},
        {"effort_scales": [1500]},
    ]:
        with pytest.raises(ValueError, match="incompatible"):
            read_prefix_candidate(saved, report | changed)


def test_resume_rejects_changed_native_state_or_parameterization(
    tmp_path: Path,
) -> None:
    source = tmp_path / "report.json"
    identity = {
        "coordinate_names": ["HipInputX"],
        "q": [0.1],
        "qd": [0.2],
        "geometry_in": [14.5, 12],
        "body_names": ["Hip"],
        "offsets_m": [[0, 0, 0.1]],
        "basis": "constant-bernstein-6",
        "duration_s": 0.1,
    }
    report = {
        "qualification": "exploratory-first-prefix-only",
        "source_sha256": "capture-a",
        "labels": ["waist"],
        "effort_scales": [200],
        "fit_identity": identity,
        "evaluations": [{"number": 1, "rmse_m": 0.001, "efforts": [20]}],
    }
    source.write_text(json.dumps(report), encoding="utf-8")
    saved = snapshot_prefix_report(source, tmp_path / "checkpoints")
    assert read_prefix_candidate(saved, report) == pytest.approx([1.1])
    for key in identity:
        with pytest.raises(ValueError, match="fit_identity"):
            read_prefix_candidate(
                saved, report | {"fit_identity": identity | {key: None}}
            )
    legacy = {key: value for key, value in report.items() if key != "fit_identity"}
    with pytest.raises(ValueError, match="fit_identity"):
        read_prefix_candidate(saved, legacy)
    source.write_text(json.dumps(legacy), encoding="utf-8")
    old = snapshot_prefix_report(source, tmp_path / "checkpoints")
    with pytest.raises(ValueError, match="fit_identity"):
        read_prefix_candidate(old, report)
