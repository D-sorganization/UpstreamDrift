"""Prospective controls for the Pinocchio CRBA evidence requalification."""

from __future__ import annotations

import copy
import json

import pytest

from scripts.research.proximal_distal_energy.articulated_crba_requalification import (
    REGISTRATION_PATH,
    build_registration,
    validate_registration,
)


pytestmark = pytest.mark.unit


def test_checked_in_registration_freezes_complete_requalification_closure() -> None:
    registration = json.loads(REGISTRATION_PATH.read_text(encoding="utf-8"))

    summary = validate_registration(registration)

    assert summary == {
        "corrected_source_count": 4,
        "primary_artifact_count": 6,
        "figure_source_count": 7,
        "execution_phase_count": 8,
        "promotion_eligible": False,
    }
    assert registration == build_registration()
    assert registration["evidence_status"] == "prospective_no_requalified_outcome"
    assert registration["promotion_authority"] == "none_until_all_gates_pass"
    assert registration["primary_artifacts"] == [
        "articulated_inertia_cross_engine",
        "articulated_native_constraint_discrepancy",
        "articulated_contact_projection",
        "articulated_drift_contact_attribution",
        "articulated_forward_contact",
        "articulated_distributed_grip_atlas",
    ]


def test_registration_rejects_post_hoc_promotion_or_source_drift() -> None:
    registration = build_registration()
    promoted = copy.deepcopy(registration)
    promoted["promotion_authority"] = "qualified"
    with pytest.raises(ValueError, match="deterministic authority"):
        validate_registration(promoted)

    drifted = copy.deepcopy(registration)
    drifted["corrected_source_authorities"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="deterministic authority"):
        validate_registration(drifted)


def test_registration_requires_native_operator_identity_and_two_clean_replays() -> None:
    registration = build_registration()
    environment = registration["qualified_environment"]
    replay = registration["replay_contract"]

    assert environment["pinocchio_distribution"] == "pin"
    assert environment["pinocchio_version"] == "3.8.0"
    assert environment["mujoco_version"] == "3.8.0"
    assert environment["native_binary_compatibility_pins"] == {
        "cmeel-urdfdom": "4.0.1",
        "cmeel-tinyxml2": "10.0.0",
    }
    assert environment["pinocchio_operator_probe"] == ["Model", "crba", "rnea"]
    assert environment["maximum_workers"] == 1
    assert replay["clean_execution_count"] == 2
    assert replay["json_comparison"] == "canonical_exact"
    assert replay["npz_comparison"] == "memberwise_exact_equal_nan"
    assert replay["outcome_inspection_before_second_replay"] is False
