"""Fail-closed stateful campaign aggregation and publication tests."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.research.proximal_distal_energy.articulated_forward_attribution_runner import (
    NativeEngineUnavailable,
    StudyCase,
    run_serial_cases,
)
from scripts.research.proximal_distal_energy.articulated_stateful_summary import (
    aggregate_stateful_smoke,
    publish_stateful_smoke_evidence,
)

ROOT = Path(__file__).resolve().parents[2]
PLAN = (
    ROOT / "docs/research/proximal_distal_energy_transfer/data/"
    "articulated_stateful_distributed_plan.json"
)
EXECUTION = "d" * 40


def _manifest() -> dict[str, object]:
    value = json.loads(PLAN.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _synthetic_result(case: StudyCase) -> dict[str, object]:
    if case.engine == "pinocchio":
        raise NativeEngineUnavailable(
            engine="pinocchio", detail="planted robotics runtime absence"
        )
    energy = case.time_step_s
    coupling = (
        1.0e-4
        if case.variant == "frictionless_killswitch"
        else (0.0 if case.variant == "opening_probe" else case.time_step_s**2)
    )
    regimes = {"elastic_stick": 1}
    if case.variant == "low_friction_slip_probe":
        regimes["coulomb_slip"] = 1
    if case.variant == "opening_probe":
        regimes = {"open": 1}
    return {
        "closure": {
            "trajectory_energy_relative_residual": energy,
            "tangential_coupling_work_relative_residual": coupling,
            "failure_codes": [],
            "passes_registered_tolerances": True,
        },
        "regimes": regimes,
        "outcomes": {
            "clubhead_speed_m_s": 1.0 + len(case.variant) / 100.0,
            "total_frictional_dissipation_j": 0.0,
            "total_release_dissipation_j": 0.0,
        },
    }


def _checkpoints(tmp_path: Path) -> tuple[dict[str, object], Path]:
    manifest = _manifest()
    checkpoint_dir = tmp_path / "run"
    run_serial_cases(
        manifest=manifest,
        execution_revision=EXECUTION,
        checkpoint_dir=checkpoint_dir,
        evaluator=_synthetic_result,
    )
    return manifest, checkpoint_dir


def test_aggregate_retains_native_absence_and_refinement_failure(
    tmp_path: Path,
) -> None:
    manifest, checkpoint_dir = _checkpoints(tmp_path)

    summary = aggregate_stateful_smoke(
        manifest=manifest,
        execution_revision=EXECUTION,
        checkpoint_dir=checkpoint_dir,
    )

    assert summary["counts"] == {
        "registered": 54,
        "completed": 27,
        "unavailable": 27,
        "failed": 0,
    }
    assert summary["promotion"] == {
        "eligible": False,
        "failure_codes": [
            "native_engine_unavailable",
            "cross_engine_parity_unavailable",
            "refinement_failure",
        ],
    }
    frictionless = next(
        group
        for group in summary["groups"]
        if group["engine"] == "mujoco" and group["variant"] == "frictionless_killswitch"
    )
    assert frictionless["coupling_work_refinement_ratios"] == [1.0, 1.0]
    assert frictionless["failure_codes"] == ["coupling_work_refinement"]
    opening = next(
        group
        for group in summary["groups"]
        if group["engine"] == "mujoco" and group["variant"] == "opening_probe"
    )
    assert opening["coupling_work_refinement_ratios"] == [0.0, 0.0]
    assert opening["passes"] is True
    assert len(summary["counterfactuals"]) == 8


def test_publish_copies_exact_checkpoint_set_and_is_deterministic(
    tmp_path: Path,
) -> None:
    manifest, checkpoint_dir = _checkpoints(tmp_path)
    output_dir = tmp_path / "published"

    summary_path = publish_stateful_smoke_evidence(
        manifest=manifest,
        execution_revision=EXECUTION,
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
    )
    first = summary_path.read_bytes()
    second_path = publish_stateful_smoke_evidence(
        manifest=manifest,
        execution_revision=EXECUTION,
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
    )

    assert second_path.read_bytes() == first
    assert len(tuple((output_dir / "checkpoints").glob("case-*.json"))) == 54
    summary = json.loads(first)
    assert summary["checkpoint_inventory"]["count"] == 54
