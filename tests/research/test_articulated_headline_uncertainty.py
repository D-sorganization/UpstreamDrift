"""Contracts for the full-atlas articulated headline uncertainty campaign."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.research.proximal_distal_energy import (
    articulated_headline_uncertainty as study,
)

pytestmark = pytest.mark.scientific


def _fake_record(
    count: int, *, gates_passed: bool = True
) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        {
            "results": {
                "matched_load_work_cell_count": count,
                "matched_load_work_total_cell_count": 384,
                "all_registered_gates_passed": gates_passed,
            }
        },
        {},
    )


def test_registered_design_covers_every_required_constitutive_axis() -> None:
    config = study.HeadlineUncertaintyConfig(worker_count=1)
    names = {axis.name for axis in config.axes}
    assert names == {
        "grip_stiffness_scale",
        "grip_damping_scale",
        "shaft_bending_frequency_scale",
        "shaft_torsional_stiffness_scale",
        "shaft_damping_ratio",
        "ground_translation_stiffness_scale",
        "ground_translation_damping_scale",
        "ground_free_moment_stiffness_scale",
        "ground_free_moment_damping_scale",
    }
    corners = study.registered_corners(config)
    assert len(corners) == 1 + 2 * len(names)
    assert corners[0].corner_id == "nominal"
    with pytest.raises(ValueError, match="worker_count"):
        study.HeadlineUncertaintyConfig(worker_count=0)


def test_twenty_worker_campaign_limits_shaft_atlas_to_twelve_workers() -> None:
    config = study.HeadlineUncertaintyConfig(worker_count=20)
    corner = study.registered_corners(config)[0]

    assert study._shaft_config(corner, config).worker_count == 12


def test_campaign_reports_headline_movement_and_retains_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_shaft(config: study.ArticulatedShaftAtlasConfig):
        count = round(126 * config.bending_frequency_scale)
        return _fake_record(
            count,
            gates_passed=config.torsional_stiffness_scale != 1.44,
        )

    def fake_ground(config: study.ArticulatedGroundAtlasConfig, **_):
        if config.ground_free_moment_damping_scale == 1.5:
            raise RuntimeError("manufactured corner failure")
        count = round(4 * (config.ground_translation_stiffness_scale - 1.0))
        return _fake_record(max(0, count))

    monkeypatch.setattr(study, "run_articulated_shaft_atlas", fake_shaft)
    monkeypatch.setattr(study, "run_articulated_ground_atlas", fake_ground)
    record = study.run_headline_uncertainty(
        study.HeadlineUncertaintyConfig(worker_count=1)
    )

    assert record["design"]["corner_count"] == 19
    assert set(record["source_sha256"]) == {
        "scripts/research/proximal_distal_energy/articulated_headline_uncertainty.py",
        "scripts/research/proximal_distal_energy/articulated_shaft_atlas.py",
        "scripts/research/proximal_distal_energy/articulated_ground_atlas.py",
        "tests/research/test_articulated_headline_uncertainty.py",
    }
    assert all(len(digest) == 64 for digest in record["source_sha256"].values())
    nominal = record["corners"][0]
    assert nominal["shaft"]["matched_cell_count"] == 126
    assert nominal["ground"]["matched_cell_count"] == 0
    bending_high = next(
        row
        for row in record["corners"]
        if row["corner_id"] == "shaft_bending_frequency_scale-high"
    )
    assert bending_high["shaft"]["matched_cell_count_change_from_nominal"] > 0
    gate_failure = next(
        row
        for row in record["corners"]
        if row["corner_id"] == "shaft_torsional_stiffness_scale-high"
    )
    assert gate_failure["shaft"]["status"] == "failed_retained"
    assert gate_failure["shaft"]["failure_class"] == "RegisteredGateFailure"
    assert gate_failure["shaft"]["matched_cell_count"] is None
    failed = next(
        row
        for row in record["corners"]
        if row["corner_id"] == "ground_free_moment_damping_scale-high"
    )
    assert failed["ground"]["status"] == "failed_retained"
    assert failed["ground"]["matched_cell_count"] is None
    assert record["results"]["shaft"]["nominal_matched_cell_count"] == 126
    assert record["results"]["shaft"]["failed_corner_count"] == 1
    assert record["results"]["shaft"]["matched_cell_count_range"][0] < 126
    assert (
        "shaft_bending_frequency_scale-high"
        in record["results"]["shaft"]["nonzero_change_corner_ids"]
    )
    assert record["results"]["ground"]["nominal_matched_cell_count"] == 0
    assert record["results"]["ground"]["failed_corner_ids"] == [
        "ground_free_moment_damping_scale-high"
    ]


def test_pathway_fails_closed_when_source_changes_during_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = study.HeadlineUncertaintyConfig(worker_count=1)
    corner = study.registered_corners(config)[0]
    launch_sources = {"source.py": "a" * 64}
    monkeypatch.setattr(
        study, "run_articulated_shaft_atlas", lambda _: _fake_record(126)
    )
    monkeypatch.setattr(study, "_source_hashes", lambda: {"source.py": "b" * 64})

    result = study._run_pathway(
        "shaft",
        corner,
        config,
        execution_source_sha256=launch_sources,
    )

    assert result["status"] == "failed_retained"
    assert result["failure_class"] == "SourceDrift"
    assert result["computed_source_sha256"] == launch_sources
    assert result["observed_source_sha256"] == {"source.py": "b" * 64}


def test_campaign_checkpoint_resumes_without_repeating_completed_cells(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = {"shaft": 0, "ground": 0}

    def fake_shaft(_: study.ArticulatedShaftAtlasConfig):
        calls["shaft"] += 1
        return _fake_record(126)

    def fake_ground(_: study.ArticulatedGroundAtlasConfig, **__):
        calls["ground"] += 1
        return _fake_record(0)

    monkeypatch.setattr(study, "run_articulated_shaft_atlas", fake_shaft)
    monkeypatch.setattr(study, "run_articulated_ground_atlas", fake_ground)
    checkpoint = tmp_path / "headline.json"
    config = study.HeadlineUncertaintyConfig(worker_count=1)
    first = study.run_headline_uncertainty(config, checkpoint_path=checkpoint)
    first_calls = calls.copy()
    legacy = json.loads(checkpoint.read_text(encoding="utf-8"))
    for row in legacy["corners"]:
        for pathway in ("shaft", "ground"):
            row[pathway].pop("computed_source_sha256", None)
    legacy_sources = {"legacy/source.py": "a" * 64}
    legacy["source_sha256"] = legacy_sources
    checkpoint.write_text(json.dumps(legacy), encoding="utf-8")
    second = study.run_headline_uncertainty(config, checkpoint_path=checkpoint)

    assert first["status"] == second["status"] == "complete"
    assert calls == first_calls
    assert len(second["corners"]) == 19
    assert second["corners"][0]["shaft"]["computed_source_sha256"] == legacy_sources


def test_checkpoint_rejects_scientific_design_drift_but_not_worker_change(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        study, "run_articulated_shaft_atlas", lambda _: _fake_record(126)
    )
    monkeypatch.setattr(
        study, "run_articulated_ground_atlas", lambda _, **__: _fake_record(0)
    )
    checkpoint = tmp_path / "headline.json"
    study.run_headline_uncertainty(
        study.HeadlineUncertaintyConfig(worker_count=1), checkpoint_path=checkpoint
    )

    study.run_headline_uncertainty(
        study.HeadlineUncertaintyConfig(worker_count=2), checkpoint_path=checkpoint
    )
    record = json.loads(checkpoint.read_text(encoding="utf-8"))
    record["design"]["axes"][0]["low"] = 0.61
    checkpoint.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(RuntimeError, match="design does not match"):
        study.run_headline_uncertainty(
            study.HeadlineUncertaintyConfig(worker_count=2),
            checkpoint_path=checkpoint,
        )
