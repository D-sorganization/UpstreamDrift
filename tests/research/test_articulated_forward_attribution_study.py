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

    assert manifest["schema_version"] == "1.4.0"
    assert manifest["issue"] == 9153
    assert manifest["preregistration"] == {
        "revision": 5,
        "amendment_timing": ("after_initial_smoke_diagnosis_before_corrected_rerun"),
        "amendment": (
            "retain the omitted one-half v-transpose Mdot v kinetic transport work term"
        ),
        "supersedes_plan_sha256": (
            "cdb4f61e3c6e814906f6c43b68d62ca601c044d0a35598ff9a35e86dee3331f1"
        ),
        "preserved_diagnostic_execution_revision": (
            "0ba50aee3ab1fe1d445cd003e2428e048685d4f0"
        ),
    }
    assert manifest["execution"]["worker_count"] == 1
    assert manifest["execution"]["case_checkpointing"] == "atomic_per_case"
    assert "execution_revision" in manifest["execution"]["resume_identity"]
    assert manifest["design"]["time_steps_s"] == [0.001, 0.0005, 0.00025]
    assert manifest["design"]["smoke_states"] == [
        {
            "source_case_index": 4,
            "source_sample_index": 6,
            "source_time_s": 0.12,
            "role": "runtime_and_pipeline_qualification_only",
        }
    ]
    assert manifest["design"]["screening_case_indices"] == [0, 4, 8, 9, 13, 17]
    assert manifest["design"]["screening_sample_indices"] == [0, 6, 12]
    assert manifest["design"]["contact_law"] == {
        "name": "bilateral_kelvin_voigt_point_attachment_always_active",
        "contact_stiffness_n_m": 1800.0,
        "contact_damping_n_s_m": 18.0,
        "initial_club_displacement_m": 0.001,
        "initial_club_velocity_m_s": 0.05,
        "unilateral_contact": False,
    }
    assert manifest["design"]["variant_parameters"][5] == {
        "name": "velocity_reversed",
        "stiffness_factor": 1.0,
        "damping_factor": 1.0,
        "displacement_factor": 1.0,
        "velocity_factor": -1.0,
    }
    assert manifest["reporting"]["world_frame"] == "source_authority_world_xyz"
    assert manifest["reporting"]["face_path_proxy"] == (
        "angle between club local +x axis and clubhead velocity in the world frame"
    )
    assert manifest["tolerances"]["pointwise_force_closure"] == 1.0e-10
    assert manifest["tolerances"]["trajectory_energy_relative"] == 2.0e-2
    assert "0.5*v_transpose*Mdot*v" in manifest["estimands"]["generalized_work"]
    assert manifest["promotion"]["kinetic_transport_work_retained"] is True
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
