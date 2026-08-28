"""Contracts for the prospective nonlinear-controller comparison."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy.nonlinear_controller_registration import (
    REPORT_PATH,
    build_registration,
    validate_registration,
)

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.scientific


def test_registration_is_committed_and_deterministic() -> None:
    expected = build_registration(ROOT)
    committed = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert committed == expected
    assert validate_registration(committed, ROOT) == {
        "controller_family_count": 9,
        "evaluation_trial_count": 24,
        "ranking_eligible_count": 0,
    }


def test_registration_binds_current_parent_authorities() -> None:
    report = build_registration(ROOT)
    assert [item["path"] for item in report["parent_authorities"]] == [
        "docs/research/proximal_distal_energy_transfer/data/trajectory_control_authority.json",
        "docs/research/proximal_distal_energy_transfer/data/bounded_event_reachability.json",
        "docs/research/proximal_distal_energy_transfer/data/event_topology_channel_matrix.json",
    ]
    assert all(len(item["sha256"]) == 64 for item in report["parent_authorities"])


def test_comparison_contract_is_matched_and_fail_closed() -> None:
    report = build_registration(ROOT)
    contract = report["matched_comparison_contract"]
    assert len(contract["evaluation_trial_ids"]) == 24
    assert contract["control_lower_nm"] == [-60.0, -15.0]
    assert contract["control_upper_nm"] == [60.0, 15.0]
    assert contract["failure_types"] == [
        "retained_event",
        "event_lost",
        "integration_failure",
        "solver_failure",
        "gate_failure",
    ]
    assert contract["ranking_rule"] == (
        "suppress all controller rankings when any comparability, adequacy, "
        "replay, convergence, optimality, event, or held-out gate fails"
    )
    assert set(report["tuning_set"]["trial_ids"]).isdisjoint(
        contract["evaluation_trial_ids"]
    )


def test_no_family_is_ranking_eligible_before_held_out_execution() -> None:
    report = build_registration(ROOT)
    families = {item["name"]: item for item in report["controller_families"]}
    assert len(families) == 9
    assert families["bounded_nmpc_collocation"]["status"] == "not_implemented"
    assert families["first_order_ilqr"]["status"] == (
        "prospective_pending_current_parent_qualification"
    )
    assert families["second_order_ddp"]["status"] == "not_implemented"
    assert families["risk_sensitive_control"]["status"] == "not_implemented"
    assert not any(item["eligible_for_ranking"] for item in families.values())
    assert report["controller_evaluation_count"] == 0


def test_registration_requires_checkpointed_single_worker_execution() -> None:
    execution = build_registration(ROOT)["execution_contract"]
    assert execution["maximum_workers"] == 1
    assert execution["checkpoint_granularity"] == "one controller-trial pair"
    assert execution["resume_requires_exact_identity_match"] is True
    assert execution["launch_authority"] == (
        "separate operator-approved run after protected parent merge verification"
    )
