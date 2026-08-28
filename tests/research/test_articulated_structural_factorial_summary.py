"""Deterministic factorial contrasts and promotion gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.articulated_structural_factorial_plan import (
    StructuralFactorialPlan,
)
from scripts.research.proximal_distal_energy.articulated_structural_factorial_launcher import (
    bind_execution_session,
)
from scripts.research.proximal_distal_energy.articulated_structural_factorial_corruption_audit import (
    audit_checkpoint_corruption,
)
from scripts.research.proximal_distal_energy.articulated_structural_factorial_runner import (
    StructuralCase,
    StructuralEvaluation,
    build_launch_manifest,
    run_serial_cases,
)
from scripts.research.proximal_distal_energy.articulated_structural_factorial_runtime_audit import (
    audit_structural_runtime,
    validate_runtime_audit,
)
from scripts.research.proximal_distal_energy.articulated_structural_factorial_summary import (
    summarize_structural_factorial,
)

pytestmark = pytest.mark.scientific
HASHES = {
    "closed_state_npz": "1" * 64,
    "shaft_structural_basis_json": "2" * 64,
    "shaft_structural_basis_npz": "3" * 64,
    "shaft_atlas_json": "4" * 64,
    "shaft_atlas_npz": "5" * 64,
    "ground_atlas_json": "6" * 64,
    "ground_atlas_npz": "7" * 64,
}


def _fixture_plan() -> dict[str, object]:
    plan = StructuralFactorialPlan(
        design_authority_revision="a" * 40,
        authority_sha256=HASHES,
    ).to_manifest()
    design = dict(plan["design"])  # type: ignore[arg-type]
    design["states"] = design["states"][:1]
    design["velocity_factors"] = design["velocity_factors"][:1]
    design["engines"] = ["mujoco"]
    design["horizons_s"] = [0.05]
    design["registered_engine_attempt_count"] = 48
    design["expected_native_attempt_count"] = 48
    plan["design"] = design
    return plan


def _evaluation(case: StructuralCase) -> StructuralEvaluation:
    levels = [int(value) for value in case.cell_id]
    value = 10.0 + 2.0 * levels[0] + 3.0 * levels[1]
    residual = 10.0 * case.time_step_s
    return StructuralEvaluation(
        result={
            "horizons": [
                {
                    "horizon_s": 0.05,
                    "final_club_translation_speed_m_s": value,
                    "club_linear_momentum_change_kg_m_s": value,
                    "signed_contact_impulse_n_s": value,
                    "signed_contact_work_j": value,
                    "terminal_contact_dissipation_j": value / 3.0,
                    "terminal_shaft_dissipation_j": value / 3.0,
                    "terminal_ground_dissipation_j": value / 3.0,
                    "peak_grip_force_n": value,
                }
            ],
            "numerical": {
                "normalized_work_energy_residual": residual,
                "maximum_virtual_power_residual_w": 0.0,
                "maximum_shaft_power_residual_w": 0.0,
                "maximum_ground_power_residual_w": 0.0,
                "maximum_small_deflection_ratio": 0.01,
                "maximum_twist_angle_rad": 0.01,
                "maximum_base_translation_m": 0.01,
                "maximum_base_pitch_rad": 0.01,
            },
        },
        parity_arrays={
            "time_s": np.array([0.0, 0.05]),
            "q": np.zeros((2, 20)),
            "qd": np.zeros((2, 20)),
            "elastic_coordinates": np.zeros((2, 0)),
            "base_coordinates": np.zeros((2, 0)),
            "net_club_force_n": np.zeros((2, 3)),
            "maximum_station_force_n": np.zeros(2),
            "active_station_count": np.zeros(2, dtype=int),
            "ground_force_n": np.zeros((2, 3)),
        },
    )


def test_summary_recovers_registered_walsh_coefficients_and_suppresses_parity(
    tmp_path: Path,
) -> None:
    plan = _fixture_plan()
    launch = build_launch_manifest(plan=plan, execution_revision="b" * 40)
    run_serial_cases(
        plan=plan,
        launch=launch,
        checkpoint_dir=tmp_path,
        evaluator=_evaluation,
    )

    summary = summarize_structural_factorial(
        plan=plan,
        launch=launch,
        checkpoint_dir=tmp_path,
    )

    assert summary["inventory"]["registered_case_count"] == 48
    assert summary["gates"]["all_individual_numerical_pass"] is True
    assert summary["gates"]["all_refinement_groups_pass"] is True
    assert summary["gates"]["cross_engine_parity_complete_and_passed"] is False
    assert summary["gates"]["runtime_session_qualified"] is False
    assert summary["gates"]["corruption_sentinel_passed"] is False
    assert summary["gates"]["promotion_eligible"] is False
    assert summary["identity"]["runtime_identity_sha256"] is None
    assert summary["identity"]["corruption_audit_sha256"] is None
    coefficients = [
        row["walsh_coefficient"]
        for row in summary["factorial_contrasts"]
        if row["contrast_id"] == "shaft_bending"
        and row["outcome"] == "final_club_translation_speed_m_s"
    ]
    assert coefficients == pytest.approx([1.0, 1.0, 1.0])
    effects = [
        row["high_minus_low_effect"]
        for row in summary["factorial_contrasts"]
        if row["contrast_id"] == "shaft_bending"
        and row["outcome"] == "final_club_translation_speed_m_s"
    ]
    assert effects == pytest.approx([2.0, 2.0, 2.0])
    assert summary["contrast_convention"] == {
        "walsh_coefficient": "mean(outcome * coded contrast sign)",
        "high_minus_low_effect": "two times the Walsh coefficient",
        "sign_counts": "exact algebraic sign with no tolerance",
    }
    aggregate = next(
        row
        for row in summary["contrast_aggregates"]
        if row["contrast_id"] == "shaft_bending"
        and row["outcome"] == "final_club_translation_speed_m_s"
    )
    assert aggregate == {
        "contrast_id": "shaft_bending",
        "estimand_class": "primary",
        "order": 1,
        "outcome": "final_club_translation_speed_m_s",
        "expected_block_count": 3,
        "eligible_block_count": 3,
        "missing_block_count": 0,
        "support_fraction": 1.0,
        "exact_sign_counts": {"negative": 0, "zero": 0, "positive": 3},
        "sign_reversal": False,
        "walsh_coefficient": {"minimum": 1.0, "median": 1.0, "maximum": 1.0},
        "high_minus_low_effect": {
            "minimum": 2.0,
            "median": 2.0,
            "maximum": 2.0,
        },
    }
    assert len(summary["contrast_aggregates"]) == 75


def test_summary_rejects_a_stalled_final_refinement_step(tmp_path: Path) -> None:
    plan = _fixture_plan()
    launch = build_launch_manifest(plan=plan, execution_revision="b" * 40)

    def stalled_evaluation(case: StructuralCase) -> StructuralEvaluation:
        evaluation = _evaluation(case)
        result = dict(evaluation.result)
        numerical = dict(result["numerical"])
        numerical["normalized_work_energy_residual"] = {
            0.0002: 0.04,
            0.0001: 0.0316,
            0.00005: 0.0316,
        }[case.time_step_s]
        result["numerical"] = numerical
        return StructuralEvaluation(
            result=result,
            parity_arrays=evaluation.parity_arrays,
        )

    run_serial_cases(
        plan=plan,
        launch=launch,
        checkpoint_dir=tmp_path,
        evaluator=stalled_evaluation,
    )

    summary = summarize_structural_factorial(
        plan=plan,
        launch=launch,
        checkpoint_dir=tmp_path,
    )

    record = summary["refinement"][0]
    assert record["fine_to_coarse_ratio"] == pytest.approx(0.79)
    assert record["successive_refinement_ratios"] == pytest.approx([0.79, 1.0])
    assert record["maximum_successive_refinement_ratio"] == pytest.approx(1.0)
    assert record["passes"] is False
    assert summary["gates"]["all_refinement_groups_pass"] is False


def test_summary_fails_closed_when_a_registered_checkpoint_is_missing(
    tmp_path: Path,
) -> None:
    plan = _fixture_plan()
    launch = build_launch_manifest(plan=plan, execution_revision="b" * 40)

    with pytest.raises(FileNotFoundError, match="registered checkpoint is missing"):
        summarize_structural_factorial(
            plan=plan,
            launch=launch,
            checkpoint_dir=tmp_path,
        )


def test_summary_rejects_an_invalid_supplied_runtime_audit(tmp_path: Path) -> None:
    plan = _fixture_plan()
    launch = build_launch_manifest(plan=plan, execution_revision="b" * 40)
    plan_path = tmp_path / "plan.json"
    launch_path = tmp_path / "launch.json"
    audit_path = tmp_path / "runtime-audit.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    launch_path.write_text(json.dumps(launch), encoding="utf-8")
    audit_path.write_text(json.dumps({"schema_version": "invalid"}), encoding="utf-8")

    with pytest.raises(ValueError, match="runtime audit schema"):
        summarize_structural_factorial(
            plan=plan,
            launch=launch,
            checkpoint_dir=tmp_path / "checkpoints",
            plan_path=plan_path,
            launch_path=launch_path,
            runtime_audit_path=audit_path,
        )


def test_summary_accepts_a_qualified_bound_runtime_but_retains_other_gates(
    tmp_path: Path,
) -> None:
    plan = _fixture_plan()
    launch = build_launch_manifest(plan=plan, execution_revision="b" * 40)
    plan_path = tmp_path / "plan.json"
    launch_path = tmp_path / "launch.json"
    audit_path = tmp_path / "runtime-audit.json"
    checkpoint_dir = tmp_path / "checkpoints"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    launch_path.write_text(json.dumps(launch), encoding="utf-8")

    def probe(name: str) -> dict[str, str]:
        return {"name": name, "version": "3.3.4", "operator": "native"}

    audit = audit_structural_runtime(
        plan=plan,
        launch=launch,
        source_checkout={
            "revision": "b" * 40,
            "tree_sha": "c" * 40,
            "tracked_clean": True,
        },
        engine_probe=probe,
        operator_probe=lambda _name: {"passes": True},
    )
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    runtime_identity = validate_runtime_audit(plan=plan, launch=launch, audit=audit)
    bind_execution_session(
        plan_path=plan_path,
        launch_path=launch_path,
        runtime_audit_path=audit_path,
        launch=launch,
        runtime_identity=runtime_identity,
        checkpoint_dir=checkpoint_dir,
    )
    run_serial_cases(
        plan=plan,
        launch=launch,
        checkpoint_dir=checkpoint_dir,
        evaluator=_evaluation,
    )
    corruption_audit_path = tmp_path / "corruption-audit.json"
    corruption_audit_path.write_text(
        json.dumps(
            audit_checkpoint_corruption(
                plan=plan,
                launch=launch,
                checkpoint_dir=checkpoint_dir,
                audit_revision="d" * 40,
            )
        ),
        encoding="utf-8",
    )

    summary = summarize_structural_factorial(
        plan=plan,
        launch=launch,
        checkpoint_dir=checkpoint_dir,
        plan_path=plan_path,
        launch_path=launch_path,
        runtime_audit_path=audit_path,
        corruption_audit_path=corruption_audit_path,
    )

    assert summary["gates"]["runtime_session_qualified"] is True
    assert summary["gates"]["corruption_sentinel_passed"] is True
    assert summary["identity"]["runtime_identity_sha256"] == runtime_identity
    assert (
        summary["identity"]["corruption_audit_sha256"]
        == hashlib.sha256(corruption_audit_path.read_bytes()).hexdigest()
    )
    assert summary["gates"]["promotion_eligible"] is False
