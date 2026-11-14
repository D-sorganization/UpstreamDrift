"""Issue #14941: workspace optimization placeholder must not masquerade as verified."""

from __future__ import annotations

import pytest

from src.shared.python.workspace.optimization_training_workspace import (
    InvalidOptimizationConfigError,
    OptimizationJobConfig,
    OptimizationObjective,
    OptimizationResult,
    OptimizationTrainingWorkspaceCoordinator,
)
from src.shared.python.workspace.project_store import SessionProjectStore

pytestmark = pytest.mark.unit


def test_run_optimization_synchronous_is_placeholder_not_success(tmp_path) -> None:
    store = SessionProjectStore(tmp_path / "proj")
    store.create_project("p", "P")
    store.add_subject("s", "S")
    store.create_session("sess", "s", "S")
    coord = OptimizationTrainingWorkspaceCoordinator(project_store=store)
    config = OptimizationJobConfig(
        job_name="demo",
        session_id="sess",
        objectives=[OptimizationObjective("carry_distance", 180.0)],
    )
    res = coord.run_optimization_synchronous(config)
    assert res.success is False
    assert res.status == "placeholder"
    assert res.provenance.get("synthetic") is True


def test_success_true_without_backend_invoked_raises() -> None:
    with pytest.raises(InvalidOptimizationConfigError):
        OptimizationResult(
            job_id="x",
            success=True,
            optimal_speed=1.0,
            optimal_carry=1.0,
            objective_value=0.0012,
            metrics={"objective_value": 0.0012, "iterations": 14.0},
        )


def test_verified_result_requires_backend_and_non_placeholder_metrics() -> None:
    OptimizationResult(
        job_id="fixture",
        success=True,
        optimal_speed=45.0,
        optimal_carry=180.0,
        objective_value=2.5,
        provenance={"backend_invoked": True},
        metrics={"objective_value": 2.5, "iterations": 28.0},
    )


def test_placeholder_metrics_rejected_even_with_backend_invoked() -> None:
    with pytest.raises(InvalidOptimizationConfigError):
        OptimizationResult(
            job_id="bad",
            success=True,
            optimal_speed=45.0,
            optimal_carry=180.0,
            objective_value=0.0012,
            provenance={"backend_invoked": True},
            metrics={"objective_value": 0.0012, "iterations": 14.0},
        )
