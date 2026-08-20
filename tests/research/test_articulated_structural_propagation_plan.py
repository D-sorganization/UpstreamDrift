"""Contracts for fail-closed structural-corner headline propagation planning."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy.articulated_structural_propagation_plan import (
    CAMPAIGN,
    build_structural_propagation_plan,
    validate_structural_propagation_plan,
    write_structural_propagation_plan,
)

pytestmark = pytest.mark.scientific
ROOT = Path(__file__).resolve().parents[2]
COMMITTED = (
    ROOT
    / "docs/research/proximal_distal_energy_transfer/data"
    / "articulated_structural_propagation_plan.json"
)


@pytest.fixture(scope="module")
def plan():
    return build_structural_propagation_plan()


def test_plan_binds_all_seven_authority_corners(plan) -> None:
    assert plan["status"] == "ready"
    assert len(plan["corners"]) == 7
    assert len(plan["authority_campaign_sha256"]) == 64
    assert len(plan["design_sha256"]) == 64
    assert len(plan["contract_sha256"]) == 64
    assert plan["design"]["case_indices"] == [0, 8, 9, 17]
    assert plan["design"]["phase_indices"] == [0, 6, 12]
    assert "worker_count" not in plan["design"]["shaft_configuration"]
    assert "worker_count" not in plan["design"]["ground_configuration"]
    assert plan["design"]["parallelism"].startswith("worker_count is operational")
    assert {
        "scripts/research/proximal_distal_energy/articulated_shaft_atlas.py",
        "scripts/research/proximal_distal_energy/articulated_ground_atlas.py",
        "scripts/research/proximal_distal_energy/articulated_atlas_authority.py",
        "scripts/research/proximal_distal_energy/articulated_shaft_forward.py",
        "scripts/research/proximal_distal_energy/articulated_ground_forward.py",
        "docs/research/proximal_distal_energy_transfer/data/subject_scaled_closed_contact.npz",
    } < set(plan["source_sha256"])


def test_plan_preregisters_falsification_and_interpretation_boundaries(plan) -> None:
    acceptance = plan["acceptance"]

    assert acceptance["nominal_reproduction"] == {
        "shaft_matched_cell_count": 126,
        "shaft_total_cell_count": 384,
        "ground_matched_cell_count": 0,
        "ground_total_cell_count": 384,
    }
    assert set(acceptance["required_controls"]) == {
        "both native engines",
        "velocity reversal",
        "time-step refinement",
        "pathway killswitches",
        "unchanged load-work matching",
        "inconsistent authority-model scaling must fail closed",
        "deliberately infeasible joint-limit state must remain classified",
    }
    assert "without imputation" in acceptance["failure_policy"]
    assert len(acceptance["invalidators"]) == 5
    assert "no population, human" in acceptance["interpretation"]


def test_plan_requires_common_support_and_rejects_count_as_benefit(plan) -> None:
    analysis = plan["analysis"]

    assert analysis["cell_identity_fields"] == [
        "case_index",
        "phase_index",
        "velocity_factor",
        "time_step_s",
        "engine",
        "horizon_s",
    ]
    assert "persistent common matching support" in analysis["support_rule"]
    assert "0/384" in analysis["zero_nominal_ground_rule"]
    assert (
        "not evidence of paired ground-pathway benefit"
        in analysis["zero_nominal_ground_rule"]
    )
    assert "not outcome direction or causal benefit" in analysis["count_rule"]
    assert "two-engine discrepancy" in analysis["resolution_rule"]
    assert "otherwise report unresolved, not no effect" in analysis["resolution_rule"]
    assert "do not label either a derivative" in analysis["oat_secant_rule"]
    assert "nonmonotonic engineering sensitivity" in analysis["nonmonotonicity_rule"]
    assert "do not estimate higher-order" in analysis["interaction_rule"]
    assert "do not select favorable corners" in analysis["multiplicity"]


def test_plan_binds_restart_and_cell_level_evidence_contract(plan) -> None:
    evidence = plan["evidence_contract"]

    assert evidence["schema_version"] == "articulated-structural-propagation/v1"
    assert set(evidence["checkpoint_identity_fields"]) == {
        "corner_id",
        "authority_sha256",
        "scales",
        "model_sha256",
        "atlas_source_sha256",
        "scientific_configuration_sha256",
        "state_slot",
        "state",
        "pathway",
        "branch_kind",
        "branch_slot",
    }
    for pathway in ("shaft", "ground"):
        required = set(evidence["required_cell_arrays"][pathway])
        assert {
            "cell_identity",
            "matched_load_work",
            "matched_final_speed_difference_m_s",
            "load_match_relative_error",
            "work_match_relative_error",
            "gate_status",
            "failure_class",
        } <= required
    semantics = evidence["matching_metric_semantics"]
    assert semantics["shaft"] == {
        "comparison": "coupled versus rigid",
        "load": "peak station force",
        "work": "terminal dissipated work",
    }
    assert semantics["ground"] == {
        "comparison": "coupled versus fixed",
        "load": "peak grip force",
        "work": "terminal total dissipated work",
    }
    assert "atomic" in evidence["write_policy"]
    assert "all seven corners" in evidence["completion_policy"]
    assert "must not qualify" in evidence["partial_record_policy"]


def test_nominal_plan_reproduces_registered_atlas_sizes(plan) -> None:
    nominal = plan["corners"][0]

    assert nominal["corner_id"] == "nominal"
    assert nominal["requested_state_count"] == 12
    assert nominal["feasible_state_count"] == 12
    assert nominal["expected_shaft_trajectory_count"] == 384
    assert nominal["expected_ground_trajectory_count"] == 576
    assert nominal["expected_shaft_headline_cell_count"] == 384
    assert nominal["expected_ground_headline_cell_count"] == 384


def test_low_height_plan_retains_failure_and_runs_other_states(plan) -> None:
    low_height = plan["corners"][1]

    assert low_height["corner_id"] == "height_scale-low"
    assert low_height["status"] == "ready_with_retained_failure"
    assert low_height["requested_state_count"] == 12
    assert low_height["feasible_state_count"] == 11
    assert low_height["retained_failures"] == [
        {"case_index": 0, "phase_index": 12, "failure_class": "ik_nonconvergence"}
    ]
    assert [0, 12] not in low_height["feasible_states"]
    assert low_height["expected_shaft_trajectory_count"] == 352
    assert low_height["expected_ground_trajectory_count"] == 528
    assert low_height["expected_shaft_headline_cell_count"] == 352
    assert low_height["expected_ground_headline_cell_count"] == 352


def test_every_corner_binds_models_and_accounts_for_states(plan) -> None:
    for corner in plan["corners"]:
        assert (
            corner["feasible_state_count"] + len(corner["retained_failures"])
            == corner["requested_state_count"]
        )
        authority = corner["authority"]
        assert len(authority["authority_sha256"]) == 64
        assert set(authority["model_sha256"]) == {"0", "8", "9", "17"}


def test_committed_plan_is_exactly_reproducible(plan) -> None:
    committed_bytes = COMMITTED.read_bytes()
    committed = json.loads(committed_bytes)

    assert committed == plan
    assert committed_bytes == (json.dumps(plan, indent=2) + "\n").encode("utf-8")
    assert validate_structural_propagation_plan() == plan


def test_plan_rejects_incomplete_campaign(tmp_path) -> None:
    campaign = json.loads(CAMPAIGN.read_text(encoding="utf-8"))
    campaign["status"] = "in_progress"
    candidate = tmp_path / "campaign.json"
    candidate.write_text(json.dumps(campaign), encoding="utf-8")

    with pytest.raises(RuntimeError, match="must be complete"):
        build_structural_propagation_plan(candidate)


def test_plan_rejects_campaign_authority_digest_mismatch(tmp_path) -> None:
    campaign = json.loads(CAMPAIGN.read_text(encoding="utf-8"))
    campaign["corners"][0]["authority_sha256"] = "b" * 64
    candidate = tmp_path / "campaign.json"
    candidate.write_text(json.dumps(campaign), encoding="utf-8")

    with pytest.raises(RuntimeError, match="digests do not match"):
        build_structural_propagation_plan(candidate)


def test_validation_rejects_tampered_committed_plan(tmp_path) -> None:
    plan = json.loads(COMMITTED.read_text(encoding="utf-8"))
    plan["corners"][0]["feasible_state_count"] = 11
    candidate = tmp_path / "plan.json"
    candidate.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(RuntimeError, match="stale or altered"):
        validate_structural_propagation_plan(candidate)


def test_plan_write_is_atomic_and_exact(tmp_path, plan) -> None:
    candidate = tmp_path / "plan.json"

    assert write_structural_propagation_plan(candidate) == plan
    assert json.loads(candidate.read_text(encoding="utf-8")) == plan
    assert not candidate.with_suffix(".json.tmp").exists()
