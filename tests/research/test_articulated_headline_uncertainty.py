"""Contracts for the full-atlas articulated headline uncertainty campaign."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.research.proximal_distal_energy import (
    articulated_headline_uncertainty as study,
)

pytestmark = pytest.mark.scientific


def _fake_record(count: int) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        {
            "results": {
                "matched_load_work_cell_count": count,
                "matched_load_work_total_cell_count": 384,
                "all_registered_gates_passed": True,
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


def test_campaign_reports_headline_movement_and_retains_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_shaft(config: study.ArticulatedShaftAtlasConfig):
        count = round(126 * config.bending_frequency_scale)
        return _fake_record(count)

    def fake_ground(config: study.ArticulatedGroundAtlasConfig):
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
    nominal = record["corners"][0]
    assert nominal["shaft"]["matched_cell_count"] == 126
    assert nominal["ground"]["matched_cell_count"] == 0
    bending_high = next(
        row
        for row in record["corners"]
        if row["corner_id"] == "shaft_bending_frequency_scale-high"
    )
    assert bending_high["shaft"]["matched_cell_count_change_from_nominal"] > 0
    failed = next(
        row
        for row in record["corners"]
        if row["corner_id"] == "ground_free_moment_damping_scale-high"
    )
    assert failed["ground"]["status"] == "failed_retained"
    assert failed["ground"]["matched_cell_count"] is None


def test_campaign_checkpoint_resumes_without_repeating_completed_cells(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = {"shaft": 0, "ground": 0}

    def fake_shaft(_: study.ArticulatedShaftAtlasConfig):
        calls["shaft"] += 1
        return _fake_record(126)

    def fake_ground(_: study.ArticulatedGroundAtlasConfig):
        calls["ground"] += 1
        return _fake_record(0)

    monkeypatch.setattr(study, "run_articulated_shaft_atlas", fake_shaft)
    monkeypatch.setattr(study, "run_articulated_ground_atlas", fake_ground)
    checkpoint = tmp_path / "headline.json"
    config = study.HeadlineUncertaintyConfig(worker_count=1)
    first = study.run_headline_uncertainty(config, checkpoint_path=checkpoint)
    first_calls = calls.copy()
    second = study.run_headline_uncertainty(config, checkpoint_path=checkpoint)

    assert first["status"] == second["status"] == "complete"
    assert calls == first_calls
    assert len(second["corners"]) == 19
