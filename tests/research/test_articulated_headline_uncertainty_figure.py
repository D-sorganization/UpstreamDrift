"""Figure contracts for the articulated headline uncertainty campaign."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy.make_articulated_headline_uncertainty_figure import (
    render_headline_uncertainty_figure,
)

pytestmark = pytest.mark.scientific


def _record() -> dict[str, object]:
    return {
        "schema_version": "articulated-headline-uncertainty/v1",
        "status": "complete",
        "design": {
            "method": "registered_nominal_plus_one_at_a_time_low_high_corners",
            "corner_count": 3,
            "axes": [
                {
                    "name": "grip_stiffness_scale",
                    "low": 0.6,
                    "nominal": 1.0,
                    "high": 1.4,
                    "pathways": ["shaft", "ground"],
                }
            ],
        },
        "corners": [
            {
                "corner_id": "nominal",
                "axis_name": "nominal",
                "level": "nominal",
                "value": 1.0,
                "shaft": {
                    "status": "completed",
                    "matched_cell_count": 126,
                    "matched_cell_count_change_from_nominal": 0,
                },
                "ground": {
                    "status": "completed",
                    "matched_cell_count": 0,
                    "matched_cell_count_change_from_nominal": 0,
                },
            },
            {
                "corner_id": "grip_stiffness_scale-low",
                "axis_name": "grip_stiffness_scale",
                "level": "low",
                "value": 0.6,
                "shaft": {
                    "status": "completed",
                    "matched_cell_count": 119,
                    "matched_cell_count_change_from_nominal": -7,
                },
                "ground": {
                    "status": "completed",
                    "matched_cell_count": 4,
                    "matched_cell_count_change_from_nominal": 4,
                },
            },
            {
                "corner_id": "grip_stiffness_scale-high",
                "axis_name": "grip_stiffness_scale",
                "level": "high",
                "value": 1.4,
                "shaft": {
                    "status": "failed_retained",
                    "matched_cell_count": None,
                    "matched_cell_count_change_from_nominal": None,
                },
                "ground": {
                    "status": "completed",
                    "matched_cell_count": 0,
                    "matched_cell_count_change_from_nominal": 0,
                },
            },
        ],
        "limitations": {
            "interaction_order": "one-at-a-time corners do not estimate interactions",
            "calibration": "bounds are engineering ranges",
            "human_inference": "survival does not promote a human claim",
        },
    }


def test_render_headline_uncertainty_figure(tmp_path: Path) -> None:
    record_path = tmp_path / "headline.json"
    record_path.write_text(json.dumps(_record()), encoding="utf-8")
    output = tmp_path / "headline"

    render_headline_uncertainty_figure(record_path, output)

    assert output.with_suffix(".pdf").stat().st_size > 1_000
    svg = output.with_suffix(".svg").read_text(encoding="utf-8")
    assert "Articulated Headline Sensitivity" in svg
    assert "Failed Corner Retained" in svg


def test_figure_rejects_incomplete_campaign(tmp_path: Path) -> None:
    record = _record()
    record["status"] = "in_progress"
    record_path = tmp_path / "headline.json"
    record_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="complete campaign"):
        render_headline_uncertainty_figure(record_path, tmp_path / "headline")
