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


def _write_receipt(
    path: Path,
    *,
    source: Path,
    launch: dict[str, object],
    run_id: int,
    artifact_name: str,
    conclusion: str,
    start: int,
    stop: int,
    schema_version: str = ("articulated-structural-factorial-artifact-receipt/1.3.0"),
) -> Path:
    head_sha = "e" * 40
    files = [
        {"name": name, "sha256": digest}
        for name, digest in sorted(_digest_tree(source).items())
    ]
    payload = {
        "schema_version": schema_version,
        "classification": "workflow_artifact_provenance_not_scientific_summary",
        "requested_case_range": [start, stop],
        "execution_revision": launch["execution_revision"],
        "execution_session_sha256": hashlib.sha256(
            (source / "execution-session.json").read_bytes()
        ).hexdigest(),
        "run": {
            "id": run_id,
            "status": "completed",
            "conclusion": conclusion,
            "head_sha": head_sha,
        },
        "job": {"run_id": run_id, "head_sha": head_sha},
        "artifact": {
            "name": artifact_name,
            "digest": f"sha256:{'d' * 64}",
            "workflow_run": {"id": run_id, "head_sha": head_sha},
        },
        "artifact_archive_sha256": "d" * 64,
        "checkpoint_pair_count": len(tuple(source.glob("case-*.json"))),
        "files": files,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _source_record(
    *,
    launch: dict[str, object],
    run_id: int,
    artifact_name: str,
    conclusion: str,
    start: int,
    stop: int,
    directory: Path,
    schema_version: str = ("articulated-structural-factorial-artifact-receipt/1.3.0"),
) -> StructuralSliceSource:
    receipt = _write_receipt(
        directory.parent / f"{directory.name}-{run_id}-receipt.json",
        source=directory,
        launch=launch,
        run_id=run_id,
        artifact_name=artifact_name,
        conclusion=conclusion,
        start=start,
        stop=stop,
        schema_version=schema_version,
    )
    return StructuralSliceSource(
        run_id,
        artifact_name,
        conclusion,
        start,
        stop,
        directory,
        receipt,
    )


def test_collection_rejects_source_metadata_or_bytes_not_bound_by_receipt(
    tmp_path: Path,
) -> None:
    plan = _plan()
    launch = build_launch_manifest(plan=plan, execution_revision="b" * 40)
    source = _slice(tmp_path / "source", plan=plan, launch=launch, start=0, stop=2)
    receipt = _write_receipt(
        tmp_path / "receipt.json",
        source=source,
        launch=launch,
        run_id=101,
        artifact_name="artifact",
        conclusion="success",
        start=0,
        stop=2,
    )

    with pytest.raises(ValueError, match="receipt run ID"):
        collect_structural_slices(
            plan=plan,
            launch=launch,
            sources=(
                StructuralSliceSource(
                    202, "artifact", "success", 0, 2, source, receipt
                ),
            ),
            output_dir=tmp_path / "metadata-mismatch",
        )

    next(source.glob("case-*.json")).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="receipt files"):
        collect_structural_slices(
            plan=plan,
            launch=launch,
            sources=(
                StructuralSliceSource(
                    101, "artifact", "success", 0, 2, source, receipt
                ),
            ),
            output_dir=tmp_path / "byte-mismatch",
        )


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
            _source_record(
                launch=launch,
                run_id=202,
                artifact_name="structural-checkpoints-202",
                conclusion="success",
                start=2,
                stop=4,
                directory=second,
            ),
            _source_record(
                launch=launch,
                run_id=101,
                artifact_name="structural-checkpoints-101",
                conclusion="success",
                start=0,
                stop=2,
                directory=first,
                schema_version=(
                    "articulated-structural-factorial-artifact-receipt/1.1.0"
                ),
            ),
        ),
        output_dir=output,
    )

    assert manifest["complete"] is True
    assert manifest["combined_checkpoint_count"] == 4
    assert manifest["next_missing_case_index"] == 4
    assert [source["run_id"] for source in manifest["sources"]] == [101, 202]
    assert [source["artifact_receipt_schema"] for source in manifest["sources"]] == [
        "articulated-structural-factorial-artifact-receipt/1.1.0",
        "articulated-structural-factorial-artifact-receipt/1.3.0",
    ]
    assert manifest["sources"][0]["observed_case_range"] == [0, 2]
    assert json.loads((output / "collection-manifest.json").read_text()) == manifest
    assert (
        len(
            load_registered_checkpoints(plan=plan, launch=launch, checkpoint_dir=output)
        )
        == 4
    )
    assert {first: _digest_tree(first), second: _digest_tree(second)} == before
    assert not tuple(tmp_path.glob("combined.tmp-*"))


@pytest.mark.parametrize(
    "schema_version",
    [
        "articulated-structural-factorial-artifact-receipt/1.1.0",
        "articulated-structural-factorial-artifact-receipt/1.2.0",
        "articulated-structural-factorial-artifact-receipt/1.3.0",
    ],
)
def test_collection_accepts_each_governed_receipt_schema(
    tmp_path: Path, schema_version: str
) -> None:
    plan = _plan()
    launch = build_launch_manifest(plan=plan, execution_revision="b" * 40)
    source = _slice(tmp_path / "source", plan=plan, launch=launch, start=0, stop=2)

    manifest = collect_structural_slices(
        plan=plan,
        launch=launch,
        sources=(
            _source_record(
                launch=launch,
                run_id=101,
                artifact_name="structural-checkpoints-101",
                conclusion="success",
                start=0,
                stop=2,
                directory=source,
                schema_version=schema_version,
            ),
        ),
        output_dir=tmp_path / "combined",
    )

    assert manifest["sources"][0]["artifact_receipt_schema"] == schema_version


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
        _source_record(
            launch=launch,
            run_id=101,
            artifact_name="structural-checkpoints-101",
            conclusion="success",
            start=0,
            stop=2,
            directory=first,
        ),
        _source_record(
            launch=launch,
            run_id=202,
            artifact_name="structural-checkpoints-202",
            conclusion="success",
            start=2,
            stop=4,
            directory=drifted,
        ),
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
        _source_record(
            launch=launch,
            run_id=101,
            artifact_name="structural-checkpoints-101",
            conclusion="success",
            start=0,
            stop=2,
            directory=first,
        ),
        _source_record(
            launch=launch,
            run_id=303,
            artifact_name="structural-checkpoints-303",
            conclusion="success",
            start=1,
            stop=3,
            directory=overlap,
        ),
    )
    with pytest.raises(ValueError, match="overlapping checkpoint"):
        collect_structural_slices(
            plan=plan,
            launch=launch,
            sources=sources,
            output_dir=tmp_path / "overlap-output",
        )


@pytest.mark.parametrize("defect", ["job", "artifact"])
def test_collection_rejects_cross_run_receipt_identity(
    tmp_path: Path, defect: str
) -> None:
    plan = _plan()
    launch = build_launch_manifest(plan=plan, execution_revision="b" * 40)
    source = _slice(tmp_path / "source", plan=plan, launch=launch, start=0, stop=2)
    source_record = _source_record(
        launch=launch,
        run_id=101,
        artifact_name="structural-checkpoints-101",
        conclusion="success",
        start=0,
        stop=2,
        directory=source,
    )
    receipt = json.loads(source_record.receipt_path.read_text(encoding="utf-8"))
    if defect == "job":
        receipt["job"]["run_id"] = 202
    else:
        receipt["artifact"]["workflow_run"]["head_sha"] = "0" * 40
    source_record.receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="not bound to the retained run"):
        collect_structural_slices(
            plan=plan,
            launch=launch,
            sources=(source_record,),
            output_dir=tmp_path / "cross-run-output",
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
            sources=(
                _source_record(
                    launch=launch,
                    run_id=101,
                    artifact_name="artifact",
                    conclusion="success",
                    start=0,
                    stop=2,
                    directory=source,
                ),
            ),
            output_dir=tmp_path / "corrupt-output",
        )

    source = _slice(tmp_path / "unexpected", plan=plan, launch=launch, start=2, stop=4)
    (source / "partial.tmp").write_text("inflight", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected files"):
        collect_structural_slices(
            plan=plan,
            launch=launch,
            sources=(
                _source_record(
                    launch=launch,
                    run_id=202,
                    artifact_name="artifact",
                    conclusion="success",
                    start=2,
                    stop=4,
                    directory=source,
                ),
            ),
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
    receipt = _write_receipt(
        tmp_path / "receipt.json",
        source=source,
        launch=launch,
        run_id=33174130362,
        artifact_name="structural-checkpoints-33174130362",
        conclusion="success",
        start=0,
        stop=2,
    )

    result = main(
        (
            "--plan",
            str(plan_path),
            "--launch",
            str(launch_path),
            "--source",
            "33174130362",
            "structural-checkpoints-33174130362",
            "success",
            "0",
            "2",
            str(source),
            str(receipt),
            "--output-dir",
            str(output),
        )
    )

    assert result == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["sources"][0]["run_id"] == 33174130362
    assert (output / "collection-manifest.json").is_file()


def test_collection_accepts_only_a_contiguous_cancelled_prefix(tmp_path: Path) -> None:
    plan = _plan()
    launch = build_launch_manifest(plan=plan, execution_revision="b" * 40)
    initial = _slice(tmp_path / "initial", plan=plan, launch=launch, start=0, stop=1)
    prefix = _slice(tmp_path / "prefix", plan=plan, launch=launch, start=1, stop=3)

    manifest = collect_structural_slices(
        plan=plan,
        launch=launch,
        sources=(
            _source_record(
                launch=launch,
                run_id=101,
                artifact_name="initial",
                conclusion="success",
                start=0,
                stop=1,
                directory=initial,
            ),
            _source_record(
                launch=launch,
                run_id=404,
                artifact_name="partial",
                conclusion="cancelled",
                start=1,
                stop=4,
                directory=prefix,
            ),
        ),
        output_dir=tmp_path / "accepted-prefix",
    )
    assert manifest["sources"][1]["observed_case_range"] == [1, 3]
    assert manifest["sources"][1]["run_conclusion"] == "cancelled"

    incomplete_success = _slice(
        tmp_path / "incomplete-success", plan=plan, launch=launch, start=1, stop=3
    )
    with pytest.raises(ValueError, match="successful slice is incomplete"):
        collect_structural_slices(
            plan=plan,
            launch=launch,
            sources=(
                _source_record(
                    launch=launch,
                    run_id=101,
                    artifact_name="initial",
                    conclusion="success",
                    start=0,
                    stop=1,
                    directory=initial,
                ),
                _source_record(
                    launch=launch,
                    run_id=505,
                    artifact_name="incomplete",
                    conclusion="success",
                    start=1,
                    stop=4,
                    directory=incomplete_success,
                ),
            ),
            output_dir=tmp_path / "rejected-success",
        )


def test_collection_requires_one_gap_free_prefix_from_case_zero(tmp_path: Path) -> None:
    plan = _plan()
    launch = build_launch_manifest(plan=plan, execution_revision="b" * 40)
    first = _slice(tmp_path / "first", plan=plan, launch=launch, start=0, stop=2)
    after_gap = _slice(
        tmp_path / "after-gap", plan=plan, launch=launch, start=3, stop=4
    )

    with pytest.raises(ValueError, match="gap-free prefix from case zero"):
        collect_structural_slices(
            plan=plan,
            launch=launch,
            sources=(
                _source_record(
                    launch=launch,
                    run_id=101,
                    artifact_name="first",
                    conclusion="success",
                    start=0,
                    stop=2,
                    directory=first,
                ),
                _source_record(
                    launch=launch,
                    run_id=202,
                    artifact_name="after-gap",
                    conclusion="success",
                    start=3,
                    stop=4,
                    directory=after_gap,
                ),
            ),
            output_dir=tmp_path / "gapped-output",
        )
