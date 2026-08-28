"""The planted corruption sentinel is explicit and non-destructive."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.articulated_structural_factorial_corruption_audit import (
    audit_checkpoint_corruption,
)
from scripts.research.proximal_distal_energy.articulated_structural_factorial_plan import (
    StructuralFactorialPlan,
)
from scripts.research.proximal_distal_energy.articulated_structural_factorial_runner import (
    StructuralEvaluation,
    build_launch_manifest,
    run_serial_cases,
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


def _single_case_plan() -> dict[str, object]:
    plan = StructuralFactorialPlan(
        design_authority_revision="a" * 40,
        authority_sha256=HASHES,
    ).to_manifest()
    design = dict(plan["design"])  # type: ignore[arg-type]
    for key in (
        "states",
        "factorial_cells",
        "velocity_factors",
        "engines",
        "time_steps_s",
    ):
        design[key] = design[key][:1]
    design["registered_engine_attempt_count"] = 1
    design["expected_native_attempt_count"] = 1
    plan["design"] = design
    return plan


def test_corruption_audit_rejects_only_the_copy_and_preserves_source(
    tmp_path: Path,
) -> None:
    plan = _single_case_plan()
    launch = build_launch_manifest(plan=plan, execution_revision="b" * 40)
    checkpoints = run_serial_cases(
        plan=plan,
        launch=launch,
        checkpoint_dir=tmp_path,
        evaluator=lambda _case: StructuralEvaluation(
            result={"retained": True},
            parity_arrays={"time_s": np.asarray([0.0, 0.001])},
        ),
    )
    source = checkpoints[0].path
    sidecar = source.with_suffix(".npz")
    before = (
        hashlib.sha256(source.read_bytes()).hexdigest(),
        hashlib.sha256(sidecar.read_bytes()).hexdigest(),
    )

    audit = audit_checkpoint_corruption(
        plan=plan,
        launch=launch,
        checkpoint_dir=tmp_path,
        audit_revision="c" * 40,
    )

    after = (
        hashlib.sha256(source.read_bytes()).hexdigest(),
        hashlib.sha256(sidecar.read_bytes()).hexdigest(),
    )
    assert after == before
    assert audit["identity"] == {
        "plan_sha256": launch["plan_sha256"],
        "execution_revision": "b" * 40,
        "audit_revision": "c" * 40,
    }
    assert audit["sentinel"] == {
        "operation": "flip_middle_byte_in_copied_parity_sidecar",
        "source_checkpoint_unchanged": True,
        "observed_rejection": "completed checkpoint parity sidecar is missing or corrupt",
        "passes": True,
    }


def test_corruption_audit_requires_a_completed_checkpoint(tmp_path: Path) -> None:
    plan = _single_case_plan()
    launch = build_launch_manifest(plan=plan, execution_revision="b" * 40)

    with pytest.raises(ValueError, match="requires one completed"):
        audit_checkpoint_corruption(
            plan=plan,
            launch=launch,
            checkpoint_dir=tmp_path,
            audit_revision="c" * 40,
        )
