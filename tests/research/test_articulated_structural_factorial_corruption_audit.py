"""The planted corruption sentinel is explicit and non-destructive."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.articulated_structural_factorial_corruption_audit import (
    audit_checkpoint_corruption,
    main,
    validate_corruption_audit,
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
    assert (
        len(
            validate_corruption_audit(
                plan=plan,
                launch=launch,
                checkpoint_dir=tmp_path,
                audit=audit,
            )
        )
        == 64
    )

    tampered = {**audit, "sentinel": {**audit["sentinel"], "passes": False}}
    with pytest.raises(ValueError, match="exact rejection gate"):
        validate_corruption_audit(
            plan=plan,
            launch=launch,
            checkpoint_dir=tmp_path,
            audit=tampered,
        )


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


def test_corruption_audit_cli_derives_a_clean_committed_source_revision(
    tmp_path: Path,
) -> None:
    plan = _single_case_plan()
    launch = build_launch_manifest(plan=plan, execution_revision="b" * 40)
    checkpoint_dir = tmp_path / "checkpoints"
    run_serial_cases(
        plan=plan,
        launch=launch,
        checkpoint_dir=checkpoint_dir,
        evaluator=lambda _case: StructuralEvaluation(
            result={"retained": True},
            parity_arrays={"time_s": np.asarray([0.0, 0.001])},
        ),
    )
    plan_path = tmp_path / "plan.json"
    launch_path = tmp_path / "launch.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    launch_path.write_text(json.dumps(launch), encoding="utf-8")

    source_root = tmp_path / "source"
    module = source_root / (
        "scripts/research/proximal_distal_energy/"
        "articulated_structural_factorial_corruption_audit.py"
    )
    module.parent.mkdir(parents=True)
    module.write_text("# committed audit source\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", source_root], check=True)
    subprocess.run(
        ["git", "-C", source_root, "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", source_root, "config", "user.name", "Test"], check=True
    )
    subprocess.run(["git", "-C", source_root, "add", "."], check=True)
    subprocess.run(
        ["git", "-C", source_root, "commit", "-q", "-m", "audit source"],
        check=True,
    )
    revision = subprocess.run(
        ["git", "-C", source_root, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    output = tmp_path / "corruption-audit.json"

    assert (
        main(
            [
                "--plan",
                str(plan_path),
                "--launch",
                str(launch_path),
                "--checkpoint-dir",
                str(checkpoint_dir),
                "--audit-source-root",
                str(source_root),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert (
        json.loads(output.read_text(encoding="utf-8"))["identity"]["audit_revision"]
        == revision
    )

    module.write_text("# dirty audit source\n", encoding="utf-8")
    with pytest.raises(ValueError, match="committed audit module"):
        main(
            [
                "--plan",
                str(plan_path),
                "--launch",
                str(launch_path),
                "--checkpoint-dir",
                str(checkpoint_dir),
                "--audit-source-root",
                str(source_root),
                "--output",
                str(tmp_path / "dirty-audit.json"),
            ]
        )
