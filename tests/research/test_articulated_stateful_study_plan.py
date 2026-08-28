"""Prospective stateful distributed campaign contract tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.research.proximal_distal_energy.articulated_forward_attribution_runner import (
    build_registered_cases,
)

ROOT = Path(__file__).resolve().parents[2]
PLAN = (
    ROOT / "docs/research/proximal_distal_energy_transfer/data/"
    "articulated_stateful_distributed_plan.json"
)
EVALUATOR_REVISION = "6124cf4026383b9846bc9b95ef509e4f9a1426a2"
LAUNCHER_REVISION = "92c68aa91ac53a00b9ba6383464151aac071fbd7"
RAW_SHA256 = "8106a78c425119f14f7824b15c21f6b08a4c8e6ac6fd07826dbab3aeb0155758"
CANONICAL_SHA256 = "ae01fa154c7b38a47f02fbe44b8456bbef48b6b3f22a3ecc451dd7fc329fc68d"


def _manifest() -> dict[str, object]:
    value = json.loads(PLAN.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_plan_binds_immutable_evaluator_and_atomic_serial_execution() -> None:
    manifest = _manifest()

    assert manifest["identity"]["source_revision"] == EVALUATOR_REVISION
    assert manifest["preregistration"]["bound_evaluator_revision"] == EVALUATOR_REVISION
    assert manifest["preregistration"]["bound_launcher_revision"] == (LAUNCHER_REVISION)
    assert manifest["preregistration"]["timing"] == (
        "before_registered_stateful_campaign"
    )
    assert manifest["execution"]["worker_count"] == 1
    assert manifest["execution"]["case_checkpointing"] == "atomic_per_case"
    assert manifest["execution"]["launch_status"] == "not_started"
    assert manifest["execution"]["native_preload_order"] == [
        "mujoco",
        "pinocchio",
        "numerical_stack",
    ]
    assert manifest["prior_execution"]["trajectory_evidence_created"] is False


def test_plan_expands_exact_adverse_and_killswitch_case_matrix() -> None:
    manifest = _manifest()
    cases = build_registered_cases(manifest)

    assert len(cases) == 54
    assert len({case.case_key for case in cases}) == 54
    assert {case.engine for case in cases} == {"mujoco", "pinocchio"}
    assert {case.time_step_s for case in cases} == {0.001, 0.0005, 0.00025}
    assert {case.variant for case in cases} == {
        "nominal",
        "frictionless_killswitch",
        "low_friction_slip_probe",
        "high_friction",
        "low_tangential_stiffness",
        "high_tangential_stiffness",
        "zero_preload",
        "velocity_reversed",
        "opening_probe",
    }


def test_plan_preserves_scientific_boundaries_and_complete_histories() -> None:
    manifest = _manifest()
    promotion = manifest["promotion"]
    reporting = manifest["reporting"]

    assert promotion["human_or_anatomical_claims"] is False
    assert promotion["human_or_coaching_claims"] is False
    assert promotion["synthetic_evidence_substitutes_for_human_validation"] is False
    assert promotion["cross_engine_parity_required"] is True
    assert promotion["typed_failures_retained"] is True
    assert reporting["node_and_interval_histories_separate"] is True
    assert reporting["full_state_and_energy_histories_retained"] is True


def test_plan_has_stable_raw_and_canonical_hashes() -> None:
    manifest = _manifest()
    raw_sha = hashlib.sha256(PLAN.read_bytes()).hexdigest()
    canonical_sha = hashlib.sha256(
        json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()

    assert raw_sha == RAW_SHA256
    assert canonical_sha == CANONICAL_SHA256
