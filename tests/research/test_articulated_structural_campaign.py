"""Top-level orchestration contracts for structural headline propagation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.articulated_structural_atlas_execution import (
    StructuralAtlasExecution,
)
from scripts.research.proximal_distal_energy.articulated_structural_campaign import (
    StructuralCampaignDependencies,
    run_structural_propagation_campaign,
)

ROOT = Path(__file__).resolve().parents[2]
PLAN = (
    ROOT
    / "docs/research/proximal_distal_energy_transfer/data"
    / "articulated_structural_propagation_plan.json"
)
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"


def _execution(pathway: str) -> StructuralAtlasExecution:
    return StructuralAtlasExecution(
        record={
            "pathway": pathway,
            "results": {"all_registered_gates_passed": True},
        },
        arrays={"sentinel": np.asarray([1.0])},
        checkpoint_audit={
            "status": "complete",
            "checkpoint_count": 1,
            "release_evidence": False,
        },
    )


def test_campaign_runs_all_fourteen_paths_before_release_promotion(
    tmp_path,
) -> None:
    calls: list[tuple[str, str, int]] = []

    def executor(pathway: str):
        def run(_authority, *, corner_id, config, **_kwargs):
            calls.append((corner_id, pathway, config.worker_count))
            return _execution(pathway)

        return run

    def release_builder(*, completed, **_kwargs):
        assert len(completed) == 14
        return {"result_sha256": "a" * 64}

    status = run_structural_propagation_campaign(
        checkpoint_directory=tmp_path / "checkpoints",
        output_directory=tmp_path / "output",
        figure_directory=tmp_path / "figures",
        worker_count=2,
        plan_path=PLAN,
        data_directory=DATA,
        dependencies=StructuralCampaignDependencies(
            shaft_executor=executor("shaft"),
            ground_executor=executor("ground"),
            release_builder=release_builder,
        ),
    )

    assert len(calls) == 14
    assert all(worker_count == 2 for _, _, worker_count in calls)
    assert status["state"] == "complete"
    assert status["release_evidence"] is True
    persisted = json.loads(
        (tmp_path / "output/articulated_structural_campaign_status.json").read_text()
    )
    assert persisted == status
    assert len(persisted["completed"]) == 14


def test_campaign_persists_dynamic_failure_without_release_promotion(tmp_path) -> None:
    calls = 0

    def fail_on_second(_authority, *, corner_id, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("manufactured branch failure")
        return _execution("shaft")

    with pytest.raises(RuntimeError, match="corner=nominal, pathway=ground"):
        run_structural_propagation_campaign(
            checkpoint_directory=tmp_path / "checkpoints",
            output_directory=tmp_path / "output",
            figure_directory=tmp_path / "figures",
            worker_count=1,
            plan_path=PLAN,
            data_directory=DATA,
            dependencies=StructuralCampaignDependencies(
                shaft_executor=fail_on_second,
                ground_executor=fail_on_second,
            ),
        )

    persisted = json.loads(
        (tmp_path / "output/articulated_structural_campaign_status.json").read_text()
    )
    assert persisted["state"] == "failed_retained"
    assert persisted["release_evidence"] is False
    assert len(persisted["completed"]) == 1
    assert persisted["retained_execution_failures"] == [
        {
            "corner_id": "nominal",
            "pathway": "ground",
            "exception_type": "RuntimeError",
            "message": "manufactured branch failure",
        }
    ]


def test_campaign_rejects_nonpositive_worker_count(tmp_path) -> None:
    with pytest.raises(ValueError, match="worker_count"):
        run_structural_propagation_campaign(
            checkpoint_directory=tmp_path / "checkpoints",
            output_directory=tmp_path / "output",
            figure_directory=tmp_path / "figures",
            worker_count=0,
            plan_path=PLAN,
            data_directory=DATA,
        )
