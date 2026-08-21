"""Contracts for fail-closed structural-corner headline propagation planning."""

from __future__ import annotations

import json
import hashlib
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


def _first_difference(left, right, path: str = "$") -> str | None:
    """Return the first exact structural mismatch for reproducibility failures."""

    if type(left) is not type(right):
        return f"{path}: type {type(left).__name__} != {type(right).__name__}"
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return f"{path}: keys differ"
        for key in left:
            mismatch = _first_difference(left[key], right[key], f"{path}.{key}")
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}: length {len(left)} != {len(right)}"
        for index, (left_value, right_value) in enumerate(
            zip(left, right, strict=True)
        ):
            mismatch = _first_difference(
                left_value,
                right_value,
                f"{path}[{index}]",
            )
            if mismatch is not None:
                return mismatch
        return None
    if left != right:
        return f"{path}: {left!r} != {right!r}"
    return None


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
        "scripts/research/proximal_distal_energy/articulated_structural_axis_evidence.py",
        "scripts/research/proximal_distal_energy/articulated_structural_common_support.py",
        "scripts/research/proximal_distal_energy/articulated_structural_cell_evidence.py",
        "scripts/research/proximal_distal_energy/articulated_structural_branch_contract.py",
        "scripts/research/proximal_distal_energy/articulated_structural_checkpoint.py",
        "scripts/research/proximal_distal_energy/articulated_structural_corner_evidence.py",
        "scripts/research/proximal_distal_energy/articulated_structural_figure_data.py",
        "scripts/research/proximal_distal_energy/articulated_structural_gate_status.py",
        "scripts/research/proximal_distal_energy/articulated_structural_result.py",
        "scripts/research/proximal_distal_energy/articulated_shaft_forward.py",
        "scripts/research/proximal_distal_energy/articulated_ground_forward.py",
        "tests/research/test_articulated_structural_axis_evidence.py",
        "tests/research/test_articulated_structural_common_support.py",
        "tests/research/test_articulated_structural_cell_evidence.py",
        "tests/research/test_articulated_structural_branch_contract.py",
        "tests/research/test_articulated_structural_checkpoint.py",
        "tests/research/test_articulated_structural_corner_evidence.py",
        "tests/research/test_articulated_structural_figure_data.py",
        "tests/research/test_articulated_structural_gate_status.py",
        "tests/research/test_articulated_structural_result.py",
        "docs/research/proximal_distal_energy_transfer/data/subject_scaled_closed_contact.npz",
    } < set(plan["source_sha256"])


def test_plan_publishes_exact_pathway_checkpoint_identities(plan) -> None:
    identities = plan["design"]["execution_identity"]
    for pathway in ("shaft", "ground"):
        identity = identities[pathway]
        paths = identity["atlas_source_paths"]
        assert len(paths) == len(set(paths))
        assert (
            "scripts/research/proximal_distal_energy/"
            "articulated_atlas_runtime_authority.py"
        ) in paths
        assert (
            "scripts/research/proximal_distal_energy/"
            "articulated_structural_execution_identity.py"
        ) in paths
        assert (
            "scripts/research/proximal_distal_energy/"
            "articulated_structural_checkpoint.py"
        ) in paths
        assert (
            "scripts/research/proximal_distal_energy/"
            "articulated_structural_branch_contract.py"
        ) in paths
        assert f"articulated_{pathway}_atlas.py" in "\n".join(paths)
        source_hashes = {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in paths
        }
        expected_source = hashlib.sha256(
            json.dumps(
                source_hashes,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        configuration = plan["design"][f"{pathway}_configuration"]
        expected_configuration = hashlib.sha256(
            json.dumps(
                configuration,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

        assert identity["atlas_source_sha256"] == expected_source
        assert identity["scientific_configuration_sha256"] == expected_configuration
        assert "worker_count" not in configuration


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
    assert "planned, feasible, and executed" in analysis["denominator_rule"]
    assert "two-engine discrepancy" in analysis["resolution_rule"]
    assert "otherwise report unresolved, not no effect" in analysis["resolution_rule"]
    assert analysis["outcome_absolute_resolution_tolerance_m_s"] == 0.001
    assert "do not label either a derivative" in analysis["oat_secant_rule"]
    assert "registered scale-factor span" in analysis["secant_definition"]
    assert "not comparable parameter-importance" in analysis["secant_boundary"]
    assert "nonmonotonic engineering sensitivity" in analysis["nonmonotonicity_rule"]
    assert (
        "identities persistent on both sides"
        in analysis["nonmonotonicity_decision_rule"]
    )
    assert (
        "sum of both numerical resolutions" in analysis["nonmonotonicity_decision_rule"]
    )
    assert "unweighted median and full range" in analysis["axis_summary_rule"]
    assert "emit null rather than pool" in analysis["axis_summary_rule"]
    assert "do not estimate higher-order" in analysis["interaction_rule"]
    assert "do not select favorable corners" in analysis["multiplicity"]


def test_plan_binds_restart_and_cell_level_evidence_contract(plan) -> None:
    evidence = plan["evidence_contract"]

    assert evidence["schema_version"] == "articulated-structural-propagation/v2"
    assert evidence["checkpoint_schema_version"] == (
        "articulated-structural-checkpoint/v1"
    )
    assert "missing, extra, or altered" in evidence["checkpoint_metadata_rule"]
    assert "fields, shapes, and dtypes" in evidence["checkpoint_payload_rule"]
    assert "pickle disabled" in evidence["checkpoint_payload_rule"]
    assert "partial restart state" in evidence["checkpoint_set_rule"]
    assert "not release evidence" in evidence["checkpoint_set_rule"]
    assert "exact remaining descriptor sequence" in evidence["checkpoint_resume_rule"]
    assert "never release evidence" in evidence["checkpoint_resume_rule"]
    assert "independently" in evidence["checkpoint_contract_generation_rule"]
    assert (
        "runner-local buffer schema" in evidence["checkpoint_contract_generation_rule"]
    )
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
            "two_engine_speed_difference_discrepancy_m_s",
            "time_step_speed_difference_discrepancy_m_s",
            "resolution_threshold_m_s",
            "corner_minus_nominal_speed_difference_m_s",
            "resolved_outcome_change",
            "comparison_status",
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
    gates = evidence["gate_derivation"]
    assert gates["shaft"] == {
        "compared_branches": ["rigid", "coupled"],
        "per_cell_components": [
            "numerical_gates_passed",
            "parity_gates_passed",
            "small_deflection_gate_passed",
            "twist_gate_passed",
        ],
    }
    assert gates["ground"] == {
        "compared_branches": ["fixed", "coupled"],
        "per_cell_components": ["primary_numerical", "primary_parity"],
    }
    assert "both compared branches" in gates["branch_rule"]
    assert "broadcast identically" in gates["parity_rule"]
    assert "every failed component" in gates["failure_rule"]
    assert "do not replace" in gates["release_rule"]
    storage = evidence["cell_evidence_storage"]
    assert storage["schema_version"] == "articulated-structural-cell-evidence/v2"
    assert "without pickle" in storage["identity_encoding"]
    assert "name, dtype, shape, and byte payload" in storage["digest_rule"]
    assert (
        "outside persistent common support" in storage["nonpersistent_resolution_rule"]
    )
    assert "nominal-only and corner-only" in storage["missing_execution_rule"]
    assert "rather than as common unmatched" in storage["missing_execution_rule"]
    assert "only on persistent common support" in storage["paired_outcome_rule"]
    assert "NaN and false resolved status" in storage["paired_outcome_rule"]
    assert "matching state must agree" in storage["support_consistency_rule"]
    assert "stored change and threshold" in storage["support_consistency_rule"]
    assert "detached copies" in storage["ownership_rule"]
    assert "cannot alter source headline cells" in storage["ownership_rule"]
    assert "atomic temporary replacement" in storage["write_policy"]
    assert "from one atlas mapping" in storage["assembly_rule"]
    assert "disjoint retained-failure states" in storage["corner_assembly_rule"]
    assert "global and per-cell gates" in storage["corner_assembly_rule"]
    assert "atomic" in evidence["write_policy"]
    assert "all seven corners" in evidence["completion_policy"]
    assert "must not qualify" in evidence["partial_record_policy"]
    assert "not device accuracy" in evidence["resolution_boundary"]
    assert set(evidence["required_axis_summary_fields"]) == {
        "axis_name",
        "low_scale",
        "nominal_scale",
        "high_scale",
        "low_to_nominal_secant_m_s_per_unit_scale",
        "nominal_to_high_secant_m_s_per_unit_scale",
        "nonmonotonic_classification",
    }
    assert "registered low/high corner pair" in evidence["axis_assembly_rule"]
    assert "emit null" in evidence["axis_assembly_rule"]
    assert "unique safe relative NPZ paths" in evidence["bundle_validation_rule"]
    assert "reconcile pathway, digest" in evidence["bundle_validation_rule"]
    assert "exact governed plan" in evidence["plan_reconciliation_rule"]
    assert "corner authority" in evidence["plan_reconciliation_rule"]
    assert set(evidence["required_corner_summary_fields"]) == {
        "corner_id",
        "pathway",
        "cell_evidence_artifact",
        "cell_evidence_sha256",
        "requested_state_count",
        "feasible_state_count",
        "retained_failures",
        "planned_headline_cell_count",
        "executed_headline_cell_count",
        "matched_cell_count",
        "matched_fraction_of_feasible",
        "all_registered_gates_passed",
        "authority",
    }


def test_plan_preregisters_nonmisleading_publication_figure(plan) -> None:
    figure = plan["figure_contract"]

    assert figure["data_schema_version"] == "articulated-structural-figure-data/v1"
    assert "exactly 14 digest-bound cell packs" in figure["data_rule"]
    assert "do not filter favorable" in figure["data_rule"]
    assert figure["required_panels"] == [
        "planned feasible executed and matched support",
        "entered exited and persistent common matching support",
        "persistent-support outcome changes with resolution status",
        "one-sided engineering secants with nonmonotonicity",
        "retained state branch and gate failures",
    ]
    assert "0/384" in figure["zero_ground_support_rule"]
    assert "paired benefit" in figure["zero_ground_support_rule"]
    assert "not parameter-importance" in figure["secant_label_rule"]
    assert "unresolved" in figure["resolution_display_rule"]
    assert "denominators" in figure["support_display_rule"]
    assert set(figure["accessibility"]) == {
        "vector-safe PDF or SVG",
        "embedded searchable text",
        "color-independent status encoding",
        "units and alt text",
    }


def test_plan_binds_paper_claim_and_release_integration(plan) -> None:
    integration = plan["integration_contract"]

    assert set(integration["required_surfaces"]) == {
        "proximal_distal_energy_transfer.qmd",
        "MODEL_COMPLETION_FALSIFICATION_MATRIX.md",
        "MOMENTUM_TRANSFER_QUESTION_PROGRAM.md",
        "data/model_completion_predictions.json",
        "data/momentum_transfer_question_registry.json",
        "data/claim_audit_registry.json",
        "DATA_DICTIONARY.md",
    }
    assert integration["claim_classification"] == "model-dependent sensitivity"
    assert "complete and validated" in integration["release_promotion_rule"]
    assert set(integration["prohibited_promotions"]) == {
        "population robustness",
        "causal parameter effect",
        "cross-parameter importance ranking",
        "human mechanism",
        "coaching recommendation",
    }


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

    assert committed == plan, _first_difference(committed, plan)
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
