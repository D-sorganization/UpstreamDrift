"""Prospective structural-factorial design contract for UpstreamDrift #9153."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy.articulated_structural_factorial_plan import (
    StructuralFactorialPlan,
)

ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = (
    ROOT
    / "docs"
    / "research"
    / "proximal_distal_energy_transfer"
    / "data"
    / "articulated_structural_factorial_plan.json"
)
AUTHORITY_HASHES = {
    "closed_state_npz": "1" * 64,
    "shaft_structural_basis_json": "2" * 64,
    "shaft_structural_basis_npz": "3" * 64,
    "shaft_atlas_json": "4" * 64,
    "shaft_atlas_npz": "5" * 64,
    "ground_atlas_json": "6" * 64,
    "ground_atlas_npz": "7" * 64,
}

pytestmark = pytest.mark.scientific


def _plan() -> StructuralFactorialPlan:
    return StructuralFactorialPlan(
        design_authority_revision="a" * 40,
        authority_sha256=AUTHORITY_HASHES,
    )


def test_plan_is_a_complete_outcome_blind_binary_factorial() -> None:
    manifest = _plan().to_manifest()

    assert manifest["schema_version"] == "articulated-structural-factorial-plan/1.1.0"
    assert "no outcome was produced" in manifest["preregistration"]["amendment"]
    design = manifest["design"]
    assert design["factors"] == [
        "shaft_bending",
        "shaft_torsion",
        "ground_translation",
        "ground_free_moment",
    ]
    assert len(design["factorial_cells"]) == 16
    assert len({cell["cell_id"] for cell in design["factorial_cells"]}) == 16
    assert design["factorial_cells"][0]["levels"] == [0, 0, 0, 0]
    assert design["factorial_cells"][-1]["levels"] == [1, 1, 1, 1]
    assert len(design["states"]) == 12
    assert design["velocity_factors"] == [1.0, -1.0]
    assert design["time_steps_s"] == [0.0002, 0.0001, 0.00005]
    assert design["horizons_s"] == [0.004, 0.01, 0.025, 0.05]
    assert design["registered_engine_attempt_count"] == 2304
    assert design["expected_native_attempt_count"] == 1152


def test_plan_freezes_paired_estimands_without_posthoc_outcome_matching() -> None:
    manifest = _plan().to_manifest()

    analysis = manifest["analysis"]
    assert analysis["blocking_key"] == [
        "source_case_index",
        "source_sample_index",
        "velocity_factor",
        "time_step_s",
        "engine",
        "horizon_s",
    ]
    assert len(analysis["primary_contrasts"]) == 10
    assert len(analysis["exploratory_higher_order_contrasts"]) == 5
    assert analysis["outcome_matching"] == "prohibited"
    assert analysis["mediators_not_eligibility_filters"] == [
        "peak_grip_force_n",
        "terminal_contact_dissipation_j",
        "terminal_shaft_dissipation_j",
        "terminal_ground_dissipation_j",
    ]
    assert manifest["falsification"]["sign_reversal_suppresses_universal_benefit"]
    assert manifest["promotion"]["human_or_coaching_claims"] is False


def test_plan_is_serial_interrupt_safe_and_not_yet_executable() -> None:
    manifest = _plan().to_manifest()

    execution = manifest["execution"]
    assert execution["worker_count"] == 1
    assert execution["maximum_python_process_count"] == 1
    assert execution["checkpoint_policy"] == "one_atomic_checkpoint_per_attempt"
    assert execution["launch_status"] == "blocked_pending_immutable_runner_revision"
    assert manifest["promotion"]["eligible"] is False


def test_plan_rejects_mutable_or_incomplete_authority() -> None:
    with pytest.raises(ValueError, match="design_authority_revision"):
        StructuralFactorialPlan(
            design_authority_revision="main",
            authority_sha256=AUTHORITY_HASHES,
        )
    with pytest.raises(ValueError, match="authority_sha256 keys"):
        StructuralFactorialPlan(
            design_authority_revision="a" * 40,
            authority_sha256={"closed_state_npz": "1" * 64},
        )
    with pytest.raises(ValueError, match="worker_count"):
        StructuralFactorialPlan(
            design_authority_revision="a" * 40,
            authority_sha256=AUTHORITY_HASHES,
            worker_count=2,
        )


def test_committed_plan_matches_generator() -> None:
    committed = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    generated = StructuralFactorialPlan(
        design_authority_revision=committed["identity"]["design_authority_revision"],
        authority_sha256=committed["identity"]["authority_sha256"],
    ).to_manifest()

    assert committed == generated
