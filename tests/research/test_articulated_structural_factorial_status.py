"""Partial campaign status is identity-bound and evidence-neutral."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.articulated_structural_factorial_plan import (
    StructuralFactorialPlan,
)
from scripts.research.proximal_distal_energy.articulated_structural_factorial_runner import (
    NativeEngineUnavailable,
    StructuralCase,
    StructuralEvaluation,
    build_launch_manifest,
    build_registered_cases,
    checkpoint_path,
    run_serial_cases,
)
from scripts.research.proximal_distal_energy.articulated_structural_factorial_status import (
    structural_factorial_status,
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


def _tiny_plan() -> dict[str, object]:
    plan = StructuralFactorialPlan(
        design_authority_revision="a" * 40,
        authority_sha256=HASHES,
    ).to_manifest()
    design = dict(plan["design"])  # type: ignore[arg-type]
    design["states"] = design["states"][:1]
    design["factorial_cells"] = design["factorial_cells"][:2]
    design["velocity_factors"] = design["velocity_factors"][:1]
    design["engines"] = ["mujoco"]
    design["time_steps_s"] = design["time_steps_s"][:2]
    design["registered_engine_attempt_count"] = 4
    design["expected_native_attempt_count"] = 4
    plan["design"] = design
    return plan


def _evaluation(_case: object) -> StructuralEvaluation:
    return StructuralEvaluation(
        result={"retained": True},
        parity_arrays={"time_s": np.array([0.0, 0.05])},
    )


def test_partial_status_validates_retained_records_and_reports_orphan_sidecar(
    tmp_path: Path,
) -> None:
    plan = _tiny_plan()
    launch = build_launch_manifest(plan=plan, execution_revision="b" * 40)
    run_serial_cases(
        plan=plan,
        launch=launch,
        checkpoint_dir=tmp_path,
        evaluator=_evaluation,
    )
    missing_case = build_registered_cases(plan)[-1]
    missing_json = checkpoint_path(tmp_path, missing_case)
    missing_json.unlink()

    status = structural_factorial_status(
        plan=plan, launch=launch, checkpoint_dir=tmp_path
    )

    assert status["retained_checkpoint_count"] == 3
    assert status["validated_completed_sidecar_count"] == 3
    assert status["inflight_or_orphan_sidecar_count"] == 1
    assert status["missing_case_count"] == 1
    assert status["next_case_key"] == missing_case.case_key
    assert status["complete"] is False
    assert status["promotion_eligible"] is False


def test_partial_status_rejects_unregistered_case_files(tmp_path: Path) -> None:
    plan = _tiny_plan()
    launch = build_launch_manifest(plan=plan, execution_revision="b" * 40)
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "case-unregistered.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="unregistered case files"):
        structural_factorial_status(plan=plan, launch=launch, checkpoint_dir=tmp_path)


def test_partial_status_exposes_typed_engine_unavailability(tmp_path: Path) -> None:
    plan = _tiny_plan()
    design = dict(plan["design"])  # type: ignore[arg-type]
    design["factorial_cells"] = design["factorial_cells"][:1]
    design["time_steps_s"] = design["time_steps_s"][:1]
    design["engines"] = ["mujoco", "pinocchio"]
    design["registered_engine_attempt_count"] = 2
    design["expected_native_attempt_count"] = 1
    plan["design"] = design
    launch = build_launch_manifest(plan=plan, execution_revision="b" * 40)

    def evaluate(case: StructuralCase) -> StructuralEvaluation:
        if case.engine == "pinocchio":
            raise NativeEngineUnavailable(
                engine="pinocchio", detail="qualified robotics runtime absent"
            )
        return _evaluation(case)

    run_serial_cases(
        plan=plan,
        launch=launch,
        checkpoint_dir=tmp_path,
        evaluator=evaluate,
    )

    status = structural_factorial_status(
        plan=plan, launch=launch, checkpoint_dir=tmp_path
    )

    assert status["observed_engine_status_counts"] == {
        "mujoco": {"completed": 1},
        "pinocchio": {"unavailable": 1},
    }
    assert status["observed_failure_code_counts"] == {"native_engine_unavailable": 1}
