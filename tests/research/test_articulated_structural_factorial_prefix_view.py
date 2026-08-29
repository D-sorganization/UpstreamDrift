"""Outcome-blind structural prefix views preserve exact source bytes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.articulated_structural_factorial_plan import (
    StructuralFactorialPlan,
)
from scripts.research.proximal_distal_energy.articulated_structural_factorial_prefix_view import (
    materialize_structural_prefix_view,
)
from scripts.research.proximal_distal_energy.articulated_structural_factorial_runner import (
    StructuralEvaluation,
    build_launch_manifest,
    load_available_checkpoints,
    run_serial_cases,
)

pytestmark = pytest.mark.scientific


def _plan() -> dict[str, object]:
    plan = StructuralFactorialPlan(
        design_authority_revision="a" * 40,
        authority_sha256={
            "closed_state_npz": "1" * 64,
            "shaft_structural_basis_json": "2" * 64,
            "shaft_structural_basis_npz": "3" * 64,
            "shaft_atlas_json": "4" * 64,
            "shaft_atlas_npz": "5" * 64,
            "ground_atlas_json": "6" * 64,
            "ground_atlas_npz": "7" * 64,
        },
    ).to_manifest()
    design = dict(plan["design"])  # type: ignore[arg-type]
    design["states"] = design["states"][:1]
    design["factorial_cells"] = design["factorial_cells"][:2]
    design["velocity_factors"] = design["velocity_factors"][:1]
    design["time_steps_s"] = design["time_steps_s"][:1]
    design["registered_engine_attempt_count"] = 4
    design["expected_native_attempt_count"] = 2
    plan["design"] = design
    return plan


def _evaluation(_case: object) -> StructuralEvaluation:
    return StructuralEvaluation(
        result={"retained": True}, parity_arrays={"x": np.array([1.0])}
    )


def _source(root: Path, *, plan: dict[str, object], launch: dict[str, object]) -> Path:
    root.mkdir()
    (root / "execution-session.json").write_text(
        json.dumps(
            {
                "schema_version": "articulated-structural-factorial-session/1.0.0",
                "execution_revision": launch["execution_revision"],
                "runtime_identity_sha256": "c" * 64,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    run_serial_cases(
        plan=plan,
        launch=launch,
        checkpoint_dir=root,
        evaluator=_evaluation,
    )
    return root


def _tree_digest(root: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.iterdir())
        if path.is_file()
    }


def test_prefix_view_is_atomic_exact_and_source_preserving(tmp_path: Path) -> None:
    plan = _plan()
    launch = build_launch_manifest(plan=plan, execution_revision="b" * 40)
    source = _source(tmp_path / "source", plan=plan, launch=launch)
    before = _tree_digest(source)
    output = tmp_path / "prefix"

    manifest = materialize_structural_prefix_view(
        plan=plan,
        launch=launch,
        source_dir=source,
        prefix_stop_exclusive=2,
        output_dir=output,
    )

    assert manifest["classification"] == (
        "operational_prefix_view_not_scientific_summary"
    )
    assert manifest["source_checkpoint_count"] == 4
    assert manifest["prefix_case_stop_exclusive"] == 2
    assert manifest["complete_source_exposed"] is False
    assert (
        len(load_available_checkpoints(plan=plan, launch=launch, checkpoint_dir=output))
        == 2
    )
    assert len(tuple(output.glob("case-*.json"))) == 2
    assert len(tuple(output.glob("case-*.npz"))) == 2
    assert _tree_digest(source) == before
    assert not tuple(tmp_path.glob("prefix.tmp-*"))


def test_prefix_view_rejects_a_gap_or_session_drift(
    tmp_path: Path,
) -> None:
    plan = _plan()
    launch = build_launch_manifest(plan=plan, execution_revision="b" * 40)
    source = _source(tmp_path / "source", plan=plan, launch=launch)
    first = sorted(source.glob("case-*.json"))[0]
    first.with_suffix(".npz").unlink()
    first.unlink()

    with pytest.raises(ValueError, match="contiguous prefix"):
        materialize_structural_prefix_view(
            plan=plan,
            launch=launch,
            source_dir=source,
            prefix_stop_exclusive=2,
            output_dir=tmp_path / "gap",
        )

    source = _source(tmp_path / "drifted", plan=plan, launch=launch)
    session_path = source / "execution-session.json"
    session = json.loads(session_path.read_text(encoding="utf-8"))
    session["execution_revision"] = "d" * 40
    session_path.write_text(json.dumps(session), encoding="utf-8")

    with pytest.raises(ValueError, match="session identity"):
        materialize_structural_prefix_view(
            plan=plan,
            launch=launch,
            source_dir=source,
            prefix_stop_exclusive=2,
            output_dir=tmp_path / "drift",
        )


@pytest.mark.parametrize("stop", [0, 5])
def test_prefix_view_rejects_an_invalid_requested_stop(
    tmp_path: Path, stop: int
) -> None:
    plan = _plan()
    launch = build_launch_manifest(plan=plan, execution_revision="b" * 40)
    source = _source(tmp_path / "source", plan=plan, launch=launch)

    with pytest.raises(ValueError, match="prefix stop"):
        materialize_structural_prefix_view(
            plan=plan,
            launch=launch,
            source_dir=source,
            prefix_stop_exclusive=stop,
            output_dir=tmp_path / "invalid",
        )
