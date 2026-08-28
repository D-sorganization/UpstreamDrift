"""Preregistered manifest and refinement gates for #9153."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy.articulated_forward_attribution_study import (
    ForwardAttributionStudyPlan,
    assess_closure_refinement,
)


ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = (
    ROOT
    / "docs"
    / "research"
    / "proximal_distal_energy_transfer"
    / "data"
    / "articulated_forward_attribution_plan.json"
)


def test_study_plan_emits_versioned_serial_manifest() -> None:
    plan = ForwardAttributionStudyPlan(
        source_revision="a" * 40,
        source_data_sha256="b" * 64,
    )

    manifest = plan.to_manifest()

    assert manifest["schema_version"] == "1.0.0"
    assert manifest["issue"] == 9153
    assert manifest["execution"]["worker_count"] == 1
    assert manifest["execution"]["case_checkpointing"] == "atomic_per_case"
    assert manifest["design"]["time_steps_s"] == [0.001, 0.0005, 0.00025]
    assert manifest["promotion"]["human_or_coaching_claims"] is False
    assert (
        manifest["estimands"]["same_trajectory_attribution"]
        != manifest["estimands"]["forward_counterfactual"]
    )


def test_committed_plan_matches_the_preregistered_generator() -> None:
    committed = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    generated = ForwardAttributionStudyPlan(
        source_revision=committed["identity"]["source_revision"],
        source_data_sha256=committed["identity"]["source_data_sha256"],
    ).to_manifest()

    assert committed == generated


def test_study_plan_rejects_parallel_local_execution_and_unfrozen_identity() -> None:
    with pytest.raises(ValueError, match="worker_count"):
        ForwardAttributionStudyPlan(
            source_revision="a" * 40,
            source_data_sha256="b" * 64,
            worker_count=2,
        )
    with pytest.raises(ValueError, match="source_revision"):
        ForwardAttributionStudyPlan(
            source_revision="main",
            source_data_sha256="b" * 64,
        )


def test_refinement_gate_accepts_contracting_closure_residuals() -> None:
    result = assess_closure_refinement(
        time_steps_s=(0.001, 0.0005, 0.00025),
        momentum_relative_residuals=(0.01, 0.0048, 0.0023),
        work_relative_residuals=(0.004, 0.0011, 0.00032),
        momentum_tolerance=0.02,
        work_tolerance=0.01,
        refinement_ratio_limit=0.8,
    )

    assert result.passes
    assert result.momentum_refinement_ratios == pytest.approx((0.48, 0.4791666667))
    assert result.work_refinement_ratios == pytest.approx((0.275, 0.2909090909))


def test_refinement_gate_retains_nonmonotonic_failure() -> None:
    result = assess_closure_refinement(
        time_steps_s=(0.001, 0.0005, 0.00025),
        momentum_relative_residuals=(0.01, 0.012, 0.006),
        work_relative_residuals=(0.004, 0.003, 0.002),
        momentum_tolerance=0.02,
        work_tolerance=0.01,
        refinement_ratio_limit=0.8,
    )

    assert not result.passes
    assert "momentum_refinement" in result.failure_codes
