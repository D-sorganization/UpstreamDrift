"""Contracts for the release-level proximal-distal claim review."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy.release_claim_review import (
    OPEN_AUDIT_STATES,
    REVIEW_SPECS,
    build_release_claim_review,
)

ROOT = Path(__file__).resolve().parents[3]
REGISTRY = (
    ROOT
    / "docs/research/proximal_distal_energy_transfer/data/claim_audit_registry.json"
)

pytestmark = pytest.mark.unit


def _registry() -> dict[str, object]:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def test_release_review_covers_every_release_claim_and_closes_only_review() -> None:
    updated, report = build_release_claim_review(_registry(), root=ROOT)

    inventory = updated["release_claim_inventory"]
    assert {item["release_claim_key"] for item in inventory} == set(REVIEW_SPECS)
    assert not {item["audit_state"] for item in inventory} & OPEN_AUDIT_STATES
    assert updated["audit_scope"]["completion_status"] == "complete"
    assert updated["audit_scope"]["completion_layer"] == "narrative_candidate_census"
    assert updated["audit_scope"]["release_review_completion_status"] == "complete"

    summary = report["summary"]
    expected_release_count = len(REVIEW_SPECS)
    assert summary["release_claim_count"] == expected_release_count
    assert summary["reviewed_release_claim_count"] == expected_release_count
    assert summary["open_release_review_count"] == 0
    assert summary["scientifically_open_gate_count"] == expected_release_count
    assert summary["atomic_claim_count"] == len(updated["claims"])

    rows = report["release_claim_reviews"]
    assert len(rows) == expected_release_count
    assert all(row["supporting_claim_ids"] for row in rows)
    assert all(row["evidence_artifacts"] for row in rows)
    assert all(row["negative_controls"] for row in rows)
    assert all(row["falsifiers"] for row in rows)
    assert all(row["remaining_scientific_gate"] for row in rows)


def test_release_review_preserves_untested_and_adverse_conclusions() -> None:
    _, report = build_release_claim_review(_registry(), root=ROOT)
    rows = {row["release_claim_key"]: row for row in report["release_claim_reviews"]}

    assert rows["human_self_stabilization"]["scientific_disposition"] == "untested"
    assert (
        rows["physical_bilateral_six_axis_device_validation"]["scientific_disposition"]
        == "untested"
    )
    assert (
        rows["high_proximal_velocity_universally_beneficial"]["scientific_disposition"]
        == "general_rule_rejected_at_declared_model_tiers"
    )
    assert (
        rows["distributed_modal_shaft_coupled_forward"]["scientific_disposition"]
        == "mechanism_supported_but_quantitative_screen_failed"
    )
    assert rows["global_event_topology_robustness"]["scientific_disposition"] == (
        "supported_for_registered_synthetic_topology_model_scenarios"
    )
    assert rows["global_event_topology_robustness"]["supporting_claim_ids"] == [
        "PD-CLAIM-321",
        "PD-CLAIM-322",
        "PD-CLAIM-323",
    ]


def test_release_review_fails_closed_on_missing_atomic_evidence() -> None:
    registry = _registry()
    broken = copy.deepcopy(registry)
    target = next(
        claim for claim in broken["claims"] if claim["claim_id"] == "PD-CLAIM-234"
    )
    target["evidence_artifacts"] = []

    with pytest.raises(ValueError, match="PD-CLAIM-234 lacks evidence_artifacts"):
        build_release_claim_review(broken, root=ROOT)


def test_release_review_fails_closed_on_release_inventory_drift() -> None:
    registry = _registry()
    broken = copy.deepcopy(registry)
    broken["release_claim_inventory"] = broken["release_claim_inventory"][:-1]

    with pytest.raises(ValueError, match="release claim keys do not match"):
        build_release_claim_review(broken, root=ROOT)
