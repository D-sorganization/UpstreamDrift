"""Governed aggregation of atomic #9153 smoke checkpoints."""

from __future__ import annotations

from pathlib import Path
import json

import pytest

from scripts.research.proximal_distal_energy.articulated_forward_attribution_runner import (
    NativeEngineUnavailable,
    StudyCase,
    run_serial_cases,
)
from scripts.research.proximal_distal_energy.articulated_forward_attribution_study import (
    ForwardAttributionStudyPlan,
)
from scripts.research.proximal_distal_energy.articulated_forward_attribution_summary import (
    aggregate_rigid_smoke,
    publish_rigid_smoke_evidence,
)


EXECUTION_REVISION = "c" * 40
REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = (
    REPO_ROOT
    / "docs"
    / "research"
    / "proximal_distal_energy_transfer"
    / "data"
    / "articulated_forward_attribution_smoke"
)


def _manifest() -> dict[str, object]:
    return ForwardAttributionStudyPlan(
        source_revision="a" * 40,
        source_data_sha256="b" * 64,
    ).to_manifest()


def _manufactured_result(case: StudyCase) -> dict[str, object]:
    if case.engine == "pinocchio":
        raise NativeEngineUnavailable(
            engine="pinocchio",
            detail="manufactured qualified runtime absence",
        )
    step_slot = {0.001: 0, 0.0005: 1, 0.00025: 2}[case.time_step_s]
    momentum = (0.01, 0.005, 0.0025)[step_slot]
    if case.variant == "nominal":
        work = (0.004, 0.0036, 0.0018)[step_slot]
    else:
        work = (0.004, 0.002, 0.001)[step_slot]
    return {
        "closure": {
            "momentum_relative_residual": momentum,
            "work_relative_residual": work,
            "pointwise_force_residual": 0.0,
            "trajectory_energy_relative_residual": 0.001,
            "failure_codes": [],
            "passes_registered_tolerances": True,
        }
    }


def test_aggregator_retains_native_absence_and_refinement_failure(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    run_serial_cases(
        manifest=manifest,
        execution_revision=EXECUTION_REVISION,
        checkpoint_dir=tmp_path,
        evaluator=_manufactured_result,
    )

    summary = aggregate_rigid_smoke(
        manifest=manifest,
        execution_revision=EXECUTION_REVISION,
        checkpoint_dir=tmp_path,
    )

    assert summary["counts"] == {
        "registered": 42,
        "completed": 21,
        "unavailable": 21,
        "failed": 0,
    }
    assert summary["promotion"]["eligible"] is False
    assert summary["promotion"]["failure_codes"] == [
        "native_engine_unavailable",
        "cross_engine_parity_unavailable",
        "refinement_failure",
    ]
    nominal = next(
        row
        for row in summary["groups"]
        if row["engine"] == "mujoco" and row["variant"] == "nominal"
    )
    assert nominal["failure_codes"] == ["work_refinement"]
    assert nominal["work_refinement_ratios"] == pytest.approx([0.9, 0.5])
    pinocchio = next(
        row
        for row in summary["groups"]
        if row["engine"] == "pinocchio" and row["variant"] == "nominal"
    )
    assert pinocchio["status"] == "unavailable"
    assert summary["claim_boundary"]["human_or_coaching_inference"] is False
    inventory = summary["checkpoint_inventory"]
    assert inventory["count"] == 42
    assert len(inventory["files"]) == 42
    assert len(inventory["checkpoint_set_sha256"]) == 64


def test_aggregator_never_promotes_before_parity_is_implemented(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    run_serial_cases(
        manifest=manifest,
        execution_revision=EXECUTION_REVISION,
        checkpoint_dir=tmp_path,
        evaluator=lambda case: (
            {
                **_manufactured_result(case),
            }
            if case.engine == "mujoco"
            else {
                **_manufactured_result(
                    StudyCase(
                        source_case_index=case.source_case_index,
                        source_sample_index=case.source_sample_index,
                        source_time_s=case.source_time_s,
                        engine="mujoco",
                        variant=case.variant,
                        time_step_s=case.time_step_s,
                        case_key=case.case_key,
                    )
                ),
            }
        ),
    )

    summary = aggregate_rigid_smoke(
        manifest=manifest,
        execution_revision=EXECUTION_REVISION,
        checkpoint_dir=tmp_path,
    )

    assert summary["counts"]["completed"] == 42
    assert summary["promotion"] == {
        "eligible": False,
        "failure_codes": ["cross_engine_parity_not_evaluated", "refinement_failure"],
    }


def test_publisher_copies_exact_checkpoints_and_writes_summary(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "published"
    run_serial_cases(
        manifest=manifest,
        execution_revision=EXECUTION_REVISION,
        checkpoint_dir=source_dir,
        evaluator=_manufactured_result,
    )

    summary_path = publish_rigid_smoke_evidence(
        manifest=manifest,
        execution_revision=EXECUTION_REVISION,
        checkpoint_dir=source_dir,
        output_dir=output_dir,
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    published = sorted((output_dir / "checkpoints").glob("case-*.json"))
    assert len(published) == 42
    assert summary_path == output_dir / "summary.json"
    assert summary["checkpoint_inventory"]["count"] == 42
    for item in summary["checkpoint_inventory"]["files"]:
        assert (output_dir / "checkpoints" / item["name"]).read_bytes() == (
            source_dir / item["name"]
        ).read_bytes()


def test_committed_smoke_evidence_is_fresh() -> None:
    manifest_path = EVIDENCE_ROOT.parent / "articulated_forward_attribution_plan.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    retained = json.loads((EVIDENCE_ROOT / "summary.json").read_text(encoding="utf-8"))

    regenerated = aggregate_rigid_smoke(
        manifest=manifest,
        execution_revision=retained["identity"]["execution_revision"],
        checkpoint_dir=EVIDENCE_ROOT / "checkpoints",
    )

    assert retained == regenerated
