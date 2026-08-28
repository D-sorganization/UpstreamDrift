"""Prospective rigid-refinement extension contract for #9153."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy.articulated_rigid_refinement_plan import (
    RigidRefinementExtensionPlan,
)
from scripts.research.proximal_distal_energy.articulated_forward_attribution_runner import (
    build_registered_cases,
)


ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = (
    ROOT
    / "docs"
    / "research"
    / "proximal_distal_energy_transfer"
    / "data"
    / "articulated_rigid_refinement_plan.json"
)


def _plan() -> RigidRefinementExtensionPlan:
    return RigidRefinementExtensionPlan(
        source_revision="a" * 40,
        source_data_sha256="b" * 64,
    )


def test_extension_is_disjoint_from_the_disclosed_pilot_and_runner_compatible() -> None:
    manifest = _plan().to_manifest()

    assert manifest["schema_version"] == "rigid-refinement-extension/1.0.0"
    assert manifest["execution"]["worker_count"] == 1
    assert manifest["execution"]["estimated_registered_case_count"] == 216
    assert manifest["execution"]["estimated_native_attempt_count"] == 108
    assert manifest["design"]["time_steps_s"] == [0.0002, 0.0001, 0.00005]
    assert manifest["design"]["variants"] == ["nominal", "damping_high"]
    assert len(manifest["design"]["smoke_states"]) == 18
    assert manifest["design"]["smoke_states"][0] == {
        "source_case_index": 0,
        "source_sample_index": 0,
        "source_time_s": 0.0,
        "role": "prospective_screening_state",
    }
    assert manifest["design"]["smoke_states"][-1] == {
        "source_case_index": 17,
        "source_sample_index": 12,
        "source_time_s": 0.24,
        "role": "prospective_screening_state",
    }
    assert manifest["preregistration"]["pilot_steps_s"] == [
        0.00025,
        0.000125,
        0.0000625,
    ]
    assert manifest["preregistration"]["confirmatory_steps_disjoint"] is True
    assert manifest["promotion"]["original_rigid_smoke_failure_erased"] is False
    assert manifest["promotion"]["human_or_coaching_claims"] is False
    cases = build_registered_cases(manifest)
    assert len(cases) == 216
    assert len({case.case_key for case in cases}) == 216


def test_extension_rejects_parallel_execution_and_unfrozen_identity() -> None:
    with pytest.raises(ValueError, match="worker_count"):
        RigidRefinementExtensionPlan(
            source_revision="a" * 40,
            source_data_sha256="b" * 64,
            worker_count=2,
        )
    with pytest.raises(ValueError, match="source_revision"):
        RigidRefinementExtensionPlan(
            source_revision="main",
            source_data_sha256="b" * 64,
        )


def test_committed_extension_matches_generator() -> None:
    committed = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    generated = RigidRefinementExtensionPlan(
        source_revision=committed["identity"]["source_revision"],
        source_data_sha256=committed["identity"]["source_data_sha256"],
    ).to_manifest()

    assert committed == generated
