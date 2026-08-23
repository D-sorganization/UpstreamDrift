"""Publication contracts for the articulated structural sensitivity figure."""

from __future__ import annotations

import hashlib
import json

import pytest

from scripts.research.proximal_distal_energy.articulated_structural_figure import (
    render_structural_figure,
)
from scripts.research.proximal_distal_energy.articulated_structural_result import (
    AXIS_PATHWAYS,
    CORNER_PATHWAYS,
)

pytestmark = pytest.mark.scientific


def _record() -> dict:
    support = []
    outcomes = []
    for corner_id, pathway in CORNER_PATHWAYS:
        persistent = int(pathway == "shaft")
        support.append(
            {
                "corner_id": corner_id,
                "pathway": pathway,
                "planned_cell_count": 384,
                "feasible_cell_count": 384,
                "executed_cell_count": 384,
                "matched_cell_count": persistent,
                "common_executed_cell_count": 384,
                "nominal_only_executed_cell_count": 0,
                "corner_only_executed_cell_count": 0,
                "persistent_cell_count": persistent,
                "entered_cell_count": 0,
                "exited_cell_count": 0,
                "resolved_persistent_cell_count": int(
                    persistent and corner_id.endswith("-high")
                ),
            }
        )
        if persistent and corner_id != "nominal":
            resolved = corner_id.endswith("-high")
            outcomes.append(
                {
                    "corner_id": corner_id,
                    "pathway": pathway,
                    "cell_identity": f"{corner_id}-cell",
                    "change_m_s": 0.002 if resolved else 0.0005,
                    "resolution_threshold_m_s": 0.001,
                    "resolved": resolved,
                }
            )
    axes = []
    for axis_name, pathway in AXIS_PATHWAYS:
        supported = pathway == "shaft"
        axes.append(
            {
                "axis_name": axis_name,
                "pathway": pathway,
                "low_scale": 0.8,
                "nominal_scale": 1.0,
                "high_scale": 1.2,
                "shared_persistent_cell_count": int(supported),
                "summary_statistic": "unweighted median",
                "low_to_nominal_secant_m_s_per_unit_scale": (
                    -0.01 if supported else None
                ),
                "nominal_to_high_secant_m_s_per_unit_scale": (
                    0.02 if supported else None
                ),
                "low_to_nominal_secant_range_m_s_per_unit_scale": (
                    [-0.02, 0.0] if supported else None
                ),
                "nominal_to_high_secant_range_m_s_per_unit_scale": (
                    [0.01, 0.03] if supported else None
                ),
                "cell_classification_counts": (
                    {"resolved_opposing": int(supported)} if supported else {}
                ),
                "nonmonotonic_classification": (
                    "resolved_opposing_on_shared_support"
                    if supported
                    else "insufficient_shared_persistent_support"
                ),
            }
        )
    payload = {
        "schema_version": "articulated-structural-figure-data/v1",
        "result_sha256": "a" * 64,
        "support": support,
        "persistent_outcomes": outcomes,
        "axis_secants": axes,
        "retained_failures": [],
        "nominal_ground_matched_cell_count": 0,
        "interpretation": "synthetic engineering sensitivity; no causal, population, human, or coaching inference",
    }
    payload["figure_data_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def test_structural_figure_is_searchable_vector_and_exposes_boundaries(
    tmp_path,
) -> None:
    output = tmp_path / "structural-sensitivity.svg"
    record = _record()

    render_structural_figure(record, output)

    text = output.read_text(encoding="utf-8")
    assert text.startswith("<?xml")
    for label in (
        "Synthetic Engineering Sensitivity",
        "Nominal Ground Matching: 0/384",
        "Support Denominators",
        "Persistent Outcome Change",
        "Not Parameter Importance",
        "Retained Failures",
        "No Human Or Coaching Inference",
        "Support, persistent outcomes, secants, and retained failures",
    ):
        assert label in text
    assert record["figure_data_sha256"] in text
    assert record["result_sha256"] in text
    assert "Body_Mass" not in text


def test_structural_figure_fails_closed_before_writing(tmp_path) -> None:
    record = _record()
    record["nominal_ground_matched_cell_count"] = 1
    output = tmp_path / "invalid.svg"

    with pytest.raises(ValueError, match="nominal ground"):
        render_structural_figure(record, output)

    assert not output.exists()
