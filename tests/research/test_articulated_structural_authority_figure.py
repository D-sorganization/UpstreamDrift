"""Figure contracts for the registered structural-authority corners."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy.make_articulated_structural_authority_figure import (
    render_articulated_structural_authority_figure,
)

pytestmark = pytest.mark.scientific


def _write_evidence(directory: Path) -> Path:
    corners = []
    for corner_id, status, failures, margin, clearance in (
        ("nominal", "feasible", 0, 0.10, 0.05),
        ("height_scale-low", "infeasible_retained", 1, 0.0, 0.06),
    ):
        artifact = f"{corner_id}.json"
        (directory / artifact).write_text(
            json.dumps(
                {
                    "results": {
                        "selected_feasible_sample_count": 52 - failures,
                        "selected_total_sample_count": 52,
                        "maximum_closure_error_m": 1.0e-10,
                        "minimum_joint_limit_margin_rad": margin,
                        "minimum_collision_clearance_m": clearance,
                    }
                }
            ),
            encoding="utf-8",
        )
        corners.append(
            {
                "corner_id": corner_id,
                "status": status,
                "failure_count": failures,
                "record_artifact": artifact,
            }
        )
    campaign = directory / "campaign.json"
    campaign.write_text(
        json.dumps(
            {
                "schema_version": "articulated-structural-authority-campaign/v1",
                "status": "complete",
                "corners": corners,
            }
        ),
        encoding="utf-8",
    )
    return campaign


def test_render_articulated_structural_authority_figure(tmp_path: Path) -> None:
    campaign = _write_evidence(tmp_path)
    output = tmp_path / "figure"

    render_articulated_structural_authority_figure(campaign, output)

    assert output.with_suffix(".pdf").stat().st_size > 1_000
    svg = output.with_suffix(".svg").read_text(encoding="utf-8")
    assert "Structural Authority Across Registered Engineering Corners" in svg
    assert "Height Scale Low" in svg
    assert "51/52" in svg


def test_figure_rejects_deleted_failure(tmp_path: Path) -> None:
    campaign = _write_evidence(tmp_path)
    record = json.loads(campaign.read_text(encoding="utf-8"))
    record["corners"][1]["failure_count"] = 0
    campaign.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="failure count"):
        render_articulated_structural_authority_figure(campaign, tmp_path / "figure")
