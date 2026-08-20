from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = (
    ROOT
    / "docs"
    / "research"
    / "proximal_distal_energy_transfer"
    / "data"
    / "momentum_transfer_question_registry.json"
)

pytestmark = pytest.mark.unit


def test_momentum_transfer_registry_is_complete_and_falsifiable() -> None:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "momentum-transfer-question-registry/v2"
    assert payload["parent_epic"] == 8557
    assert payload["program_issue"] == 8595

    questions = payload["questions"]
    assert [question["id"] for question in questions] == [
        "Q1",
        "Q2",
        "Q3",
        "Q4",
        "Q5",
        "Q6",
        "Q7",
    ]
    assert len({question["slug"] for question in questions}) == len(questions)

    for question in questions:
        assert question["question"].endswith("?")
        assert question["status"]
        assert len(question["required_estimands"]) >= 3
        assert len(question["required_controls"]) >= 3
        assert len(question["falsifier"].split()) >= 10
        assert 8595 in question["issues"]
        assert all(isinstance(issue, int) and issue > 0 for issue in question["issues"])


def test_registry_preserves_critical_scientific_distinctions() -> None:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    by_id = {question["id"]: question for question in payload["questions"]}
    by_point = {point["id"]: point for point in payload["critical_points"]}

    assert "cancellation indicator" in by_id["Q1"]["required_controls"]
    assert "alternative casting criteria" in by_id["Q3"]["required_controls"]
    assert "observer-delay sensitivity" in by_id["Q4"]["required_estimands"]
    assert "attraction-region size" in by_id["Q5"]["required_estimands"]
    assert "interior-optimum search" in by_id["Q6"]["required_controls"]
    assert "one slack class at a time" in by_id["Q7"]["required_controls"]
    assert by_id["Q7"]["status"] == "partly_answered"
    geometry = by_point["MTQ-02"]
    assert "six corners retain 52/52 feasible states" in geometry["present_answer"]
    assert "case-0/phase-12 IK nonconvergence" in geometry["present_answer"]
    assert (
        "Propagate every feasible #8800 scaled authority"
        in (geometry["decisive_next_test"])
    )
    assert (
        "Complete #8800 authority regeneration" not in (geometry["decisive_next_test"])
    )
    assert (
        "docs/research/proximal_distal_energy_transfer/data/"
        "articulated_structural_authority_campaign.json"
        in geometry["evidence_artifacts"]
    )


def test_every_source_point_links_to_inspectable_evidence() -> None:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[2]

    for point in payload["critical_points"]:
        assert point["evidence_artifacts"]
        for artifact in point["evidence_artifacts"]:
            assert (root / artifact).is_file(), f"{point['id']}: {artifact}"
