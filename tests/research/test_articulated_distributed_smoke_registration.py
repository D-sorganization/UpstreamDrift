"""Prospective controls for the distributed event-attribution smoke study."""

from __future__ import annotations

import copy
import json

import pytest

from scripts.research.proximal_distal_energy.articulated_distributed_smoke_registration import (
    EVALUATOR_REVISION,
    REGISTRATION_PATH,
    build_registration,
    registered_smoke_cases,
    validate_registration,
)


pytestmark = pytest.mark.scientific


def test_checked_in_registration_is_deterministic_and_prospective() -> None:
    registration = json.loads(REGISTRATION_PATH.read_text(encoding="utf-8"))

    summary = validate_registration(registration)

    assert registration == build_registration()
    assert summary == {
        "case_count": 6,
        "engine_count": 2,
        "time_step_count": 3,
        "promotion_eligible": False,
    }
    assert registration["evaluator_authority"]["revision"] == EVALUATOR_REVISION
    assert registration["execution_status"] == "not_started"
    assert registration["evidence_status"] == "prospective_no_smoke_outcome"
    assert registration["promotion_eligible"] is False
    assert registration["retained_outcomes"] == []


def test_registration_binds_source_data_and_current_event_scope() -> None:
    registration = build_registration()

    assert registration["input_authority"] == {
        "path": (
            "docs/research/proximal_distal_energy_transfer/data/"
            "subject_scaled_closed_contact.npz"
        ),
        "sha256": "9fa4364571ba5535995c63226289c0711ee1ebf37c58b7a3b4e4d14a98561779",
        "bytes": 35568,
    }
    assert registration["event_contract"]["supported_event_kinds"] == [
        "opening",
        "reattachment",
    ]
    assert registration["event_contract"]["prohibited_inferences"] == [
        "friction_limit_entry_or_exit",
        "static_stick_or_slip_transition",
        "discrete_impact_from_compliant_transition",
    ]


def test_registered_cases_cover_both_engines_and_three_refinement_levels() -> None:
    cases = registered_smoke_cases(build_registration())

    assert len(cases) == 6
    assert {case["engine"] for case in cases} == {"mujoco", "pinocchio"}
    assert {case["time_step_s"] for case in cases} == {0.001, 0.0005, 0.00025}
    assert len({case["case_id"] for case in cases}) == 6
    assert all(case["source_case_index"] == 0 for case in cases)
    assert all(case["source_sample_index"] == 6 for case in cases)
    assert all(case["checkpoint_policy"] == "atomic_per_case" for case in cases)


def test_registration_fails_closed_on_scope_or_outcome_drift() -> None:
    registration = build_registration()
    friction_drift = copy.deepcopy(registration)
    friction_drift["event_contract"]["supported_event_kinds"].append(
        "friction_limit_entry"
    )
    with pytest.raises(ValueError, match="deterministic authority"):
        validate_registration(friction_drift)

    outcome_drift = copy.deepcopy(registration)
    outcome_drift["retained_outcomes"].append({"case_id": "post_hoc"})
    with pytest.raises(ValueError, match="deterministic authority"):
        validate_registration(outcome_drift)

    launch_drift = copy.deepcopy(registration)
    launch_drift["execution_status"] = "completed"
    with pytest.raises(ValueError, match="deterministic authority"):
        validate_registration(launch_drift)
