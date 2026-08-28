"""Fail-closed collection contracts for hosted structural slices."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.articulated_structural_factorial_collection import (
    StructuralSliceSource,
    collect_structural_slices,
    main,
)
from scripts.research.proximal_distal_energy.articulated_structural_factorial_plan import (
    StructuralFactorialPlan,
)
from scripts.research.proximal_distal_energy.articulated_structural_factorial_runner import (
    StructuralEvaluation,
    build_launch_manifest,
    load_registered_checkpoints,
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


def _evaluate(_case: object) -> StructuralEvaluation:
    return StructuralEvaluation(
        result={"retained": True}, parity_arrays={"x": np.array([1.0])}
    )


def _slice(
    root: Path,
    *,
    plan: dict[str, object],
    launch: dict[str, object],
    start: int,
    stop: int,
) -> Path:
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
        evaluator=_evaluate,
        case_start=start,
        case_stop=stop,
    )
    return root


def _digest_tree(root: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.iterdir())
        if path.is_file()
    }


def test_collection_is_deterministic_complete_and_source_preserving(
    tmp_path: Path,
) -> None:
    plan = _plan()
    launch = build_launch_manifest(plan=plan, execution_revision="b" * 40)
    first = _slice(tmp_path / "first", plan=plan, launch=launch, start=0, stop=2)
    second = _slice(tmp_path / "second", plan=plan, launch=launch, start=2, stop=4)
    before = {first: _digest_tree(first), second: _digest_tree(second)}
    output = tmp_path / "combined"

    manifest = collect_structural_slices(
        plan=plan,
        launch=launch,
        sources=(
            StructuralSliceSource(202, "structural-checkpoints-202", second),
            StructuralSliceSource(101, "structural-checkpoints-101", first),
        ),
        output_dir=output,
    )

    assert manifest["complete"] is True
    assert manifest["combined_checkpoint_count"] == 4
    assert [source["run_id"] for source in manifest["sources"]] == [101, 202]
    assert json.loads((output / "collection-manifest.json").read_text()) == manifest
    assert (
        len(
            load_registered_checkpoints(plan=plan, launch=launch, checkpoint_dir=output)
        )
        == 4
    )
    assert {first: _digest_tree(first), second: _digest_tree(second)} == before
    assert not tuple(tmp_path.glob("combined.tmp-*"))


def test_collection_rejects_session_drift_and_overlap(tmp_path: Path) -> None:
    plan = _plan()
    launch = build_launch_manifest(plan=plan, execution_revision="b" * 40)
    first = _slice(tmp_path / "first", plan=plan, launch=launch, start=0, stop=2)
    drifted = _slice(tmp_path / "drifted", plan=plan, launch=launch, start=2, stop=4)
    (drifted / "execution-session.json").write_text(
        json.dumps(
            {
                "schema_version": "articulated-structural-factorial-session/1.0.0",
                "execution_revision": launch["execution_revision"],
                "runtime_identity_sha256": "d" * 64,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    sources = (
        StructuralSliceSource(101, "structural-checkpoints-101", first),
        StructuralSliceSource(202, "structural-checkpoints-202", drifted),
    )
    with pytest.raises(ValueError, match="execution-session bytes"):
        collect_structural_slices(
            plan=plan,
            launch=launch,
            sources=sources,
            output_dir=tmp_path / "session-drift",
        )

    overlap = _slice(tmp_path / "overlap", plan=plan, launch=launch, start=1, stop=3)
    sources = (
        StructuralSliceSource(101, "structural-checkpoints-101", first),
        StructuralSliceSource(303, "structural-checkpoints-303", overlap),
    )
    with pytest.raises(ValueError, match="overlapping checkpoint"):
        collect_structural_slices(
            plan=plan,
            launch=launch,
            sources=sources,
            output_dir=tmp_path / "overlap-output",
        )


def test_collection_rejects_corrupt_or_unexpected_slice_files(tmp_path: Path) -> None:
    plan = _plan()
    launch = build_launch_manifest(plan=plan, execution_revision="b" * 40)
    source = _slice(tmp_path / "source", plan=plan, launch=launch, start=0, stop=2)
    sidecar = next(source.glob("case-*.npz"))
    sidecar.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="sidecar is missing or corrupt"):
        collect_structural_slices(
            plan=plan,
            launch=launch,
            sources=(StructuralSliceSource(101, "artifact", source),),
            output_dir=tmp_path / "corrupt-output",
        )

    source = _slice(tmp_path / "unexpected", plan=plan, launch=launch, start=2, stop=4)
    (source / "partial.tmp").write_text("inflight", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected files"):
        collect_structural_slices(
            plan=plan,
            launch=launch,
            sources=(StructuralSliceSource(202, "artifact", source),),
            output_dir=tmp_path / "unexpected-output",
        )


def test_collection_cli_retains_explicit_workflow_provenance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = _plan()
    launch = build_launch_manifest(plan=plan, execution_revision="b" * 40)
    source = _slice(tmp_path / "source", plan=plan, launch=launch, start=0, stop=2)
    plan_path = tmp_path / "plan.json"
    launch_path = tmp_path / "launch.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    launch_path.write_text(json.dumps(launch), encoding="utf-8")
    output = tmp_path / "combined"

    result = main(
        (
            "--plan",
            str(plan_path),
            "--launch",
            str(launch_path),
            "--source",
            "33174130362",
            "structural-checkpoints-33174130362",
            str(source),
            "--output-dir",
            str(output),
        )
    )

    assert result == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["sources"][0]["run_id"] == 33174130362
    assert (output / "collection-manifest.json").is_file()
