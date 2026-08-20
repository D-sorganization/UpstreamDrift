"""Evidence gates for committed structural authority corner artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy.articulated_scaled_authority import (
    load_scaled_authority,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"
RECORD = DATA / "articulated_structural_authority_campaign.json"
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
CHAPTER = ARTICLE / "chapters/_ch06caaa_structural_authority.qmd"
MANUSCRIPT = ARTICLE / "proximal_distal_energy_transfer.qmd"
QUESTION_PROGRAM = ARTICLE / "MOMENTUM_TRANSFER_QUESTION_PROGRAM.md"
FALSIFICATION_MATRIX = ARTICLE / "MODEL_COMPLETION_FALSIFICATION_MATRIX.md"
pytestmark = pytest.mark.scientific


def test_structural_authority_campaign_is_complete_and_registered() -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))

    assert record["schema_version"] == ("articulated-structural-authority-campaign/v1")
    assert record["status"] == "complete"
    assert [row["corner_id"] for row in record["corners"]] == [
        "nominal",
        "height_scale-low",
        "height_scale-high",
        "body_mass_scale-low",
        "body_mass_scale-high",
        "joint_limit_scale-low",
        "joint_limit_scale-high",
    ]
    results = record["results"]
    assert results["corner_count"] == 7
    assert (
        results["feasible_corner_count"]
        + results["infeasible_corner_count"]
        + results["failed_corner_count"]
        == 7
    )
    for relative, expected in record["source_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected


def test_every_generated_corner_loads_without_deleting_failures() -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))

    for row in record["corners"]:
        assert row["status"] in {
            "feasible",
            "infeasible_retained",
            "failed_retained",
        }
        if row["status"] == "failed_retained":
            assert row["failure_class"]
            assert row["record_artifact"] is None
            assert row["array_artifact"] is None
            continue
        authority = load_scaled_authority(
            DATA / row["record_artifact"],
            DATA / row["array_artifact"],
        )
        selected = authority.selected_case_indices
        observed_failures = int((~authority.feasible[selected]).sum())
        assert observed_failures == row["failure_count"]
        assert len(authority.authority_sha256) == 64


def test_article_retains_every_corner_and_propagation_boundary() -> None:
    chapter = CHAPTER.read_text(encoding="utf-8")
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")

    assert "{{< include chapters/_ch06caaa_structural_authority.qmd >}}" in manuscript
    for label in (
        "Nominal",
        "Height Scale 0.90",
        "Height Scale 1.10",
        "Body-Mass Scale 0.85",
        "Body-Mass Scale 1.15",
        "Joint-Limit Scale 0.85",
        "Joint-Limit Scale 1.15",
    ):
        assert f"| {label} |" in chapter
    assert "51/52" in chapter
    assert "case 0, phase\n12" in chapter
    assert "does not yet establish sensitivity of either\nheadline estimand" in chapter
    assert "nominal model instead of its bound scaled model" in chapter
    assert "figures/fig_articulated_structural_authority.pdf" in chapter


def test_question_program_distinguishes_authority_from_propagation() -> None:
    program = QUESTION_PROGRAM.read_text(encoding="utf-8")

    assert "six corners retain 52/52 feasible states" in program
    assert "low-height corner retains one case-0/phase-12 IK nonconvergence" in program
    assert (
        "authority regeneration propagates through both headline estimands" in program
    )
    assert "not a human feasibility or prevalence result" in program
    assert "Propagate every feasible #8800 authority" in program


def test_falsification_matrix_reconciles_closed_states_and_open_sensitivity() -> None:
    matrix = FALSIFICATION_MATRIX.read_text(encoding="utf-8")

    assert (
        "subject-scaled common states fail bilateral anatomical closure" not in matrix
    )
    assert "all 234 subject-scaled states closed" in matrix
    assert "seven structural corners have not yet propagated" in matrix
    assert "structural sensitivity of the headline estimands" in matrix
    assert "human transport remain unresolved" in matrix
    assert "original prescribed cases fail the 5 mm gate" in matrix
    assert "Retain the prescribed-pose mismatch as a negative control" in matrix
    assert "compare it with the solved 234-state closed atlas" in matrix
