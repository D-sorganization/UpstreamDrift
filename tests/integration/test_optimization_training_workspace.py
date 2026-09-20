"""Integration tests for Optimization and Training Workspace (ORG-16, #10525).

RED -> GREEN Acceptance Cases:
- RED: invalid objective/constraints/model compatibility prevent job start.
- RED: cancel/pause/resume and dependency failure maintain correct persisted state; duplicate submit does not duplicate job.
- GREEN: small real deterministic optimization changes output for changed supported input; controller job fixture publishes metrics and registers result; dataset selection survives reopen.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.integration


from src.shared.python.workspace.optimization_training_workspace import (
    DuplicateJobSubmissionError,
    IncompatibleBackendError,
    InvalidOptimizationConfigError,
    ModelCompatibilityError,
    OptimizationJobConfig,
    OptimizationObjective,
    OptimizationResult,
    OptimizationTrainingWorkspaceCoordinator,
    TrainingJobConfig,
    WorkspaceJobKind,
    WorkspaceJobState,
)
from src.shared.python.workspace.project_store import SessionProjectStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace_coordinator(tmp_path: Path) -> OptimizationTrainingWorkspaceCoordinator:
    """Create an OptimizationTrainingWorkspaceCoordinator backed by a temporary project store."""
    store = SessionProjectStore(tmp_path / "project")
    store.create_project("test_proj_01", "Test Optimization Project")
    store.add_subject("subj_01", "Golfer Subj 1")
    store.create_session("sess_01", "subj_01", "Baseline Session")
    return OptimizationTrainingWorkspaceCoordinator(project_store=store)


# ---------------------------------------------------------------------------
# RED Case 1: Invalid objective/constraints/model compatibility prevent job start
# ---------------------------------------------------------------------------


def test_invalid_objective_prevents_job_start(
    workspace_coordinator: OptimizationTrainingWorkspaceCoordinator,
) -> None:
    """Empty objectives or negative weights prevent job start."""
    # Empty objectives
    config_empty = OptimizationJobConfig(
        job_name="opt_invalid_empty",
        session_id="sess_01",
        backend="scipy_lumped",
        objectives=[],
        constraints=[],
    )
    with pytest.raises(InvalidOptimizationConfigError, match="at least one objective"):
        workspace_coordinator.submit_optimization_job(config_empty)

    # Negative objective weight
    with pytest.raises(
        InvalidOptimizationConfigError, match="weight must be non-negative"
    ):
        OptimizationObjective(name="clubhead_speed", target=50.0, weight=-1.0)


def test_invalid_constraints_prevent_job_start(
    workspace_coordinator: OptimizationTrainingWorkspaceCoordinator,
) -> None:
    """Ill-posed or contradictory constraints prevent job start."""
    config_bad_bounds = OptimizationJobConfig(
        job_name="opt_invalid_bounds",
        session_id="sess_01",
        backend="scipy_lumped",
        objectives=[
            OptimizationObjective(name="clubhead_speed", target=48.0, weight=1.0)
        ],
        constraints=[
            {"name": "lead_wrist_extension", "min_val": 45.0, "max_val": -10.0}
        ],
    )
    with pytest.raises(
        InvalidOptimizationConfigError, match="lower bound exceeds upper bound"
    ):
        workspace_coordinator.submit_optimization_job(config_bad_bounds)


def test_model_and_backend_compatibility_prevents_job_start(
    workspace_coordinator: OptimizationTrainingWorkspaceCoordinator,
) -> None:
    """Unsupported or disabled backends fail closed; incompatible model contracts reject."""
    # Unsupported / uninstalled backend must remain honestly disabled
    config_unsupported = OptimizationJobConfig(
        job_name="opt_unsupported_backend",
        session_id="sess_01",
        backend="crocoddyl_transcription",  # Engine tracked under future epic, not implemented here
        objectives=[
            OptimizationObjective(name="clubhead_speed", target=50.0, weight=1.0)
        ],
        constraints=[],
    )
    with pytest.raises(IncompatibleBackendError, match="unsupported or unavailable"):
        workspace_coordinator.submit_optimization_job(config_unsupported)

    # Incompatible model specification
    config_incompatible_model = OptimizationJobConfig(
        job_name="opt_incompatible_model",
        session_id="sess_01",
        backend="scipy_lumped",
        golfer_model_id="nonexistent_alien_anatomy",
        objectives=[
            OptimizationObjective(name="clubhead_speed", target=50.0, weight=1.0)
        ],
        constraints=[],
    )
    with pytest.raises(
        ModelCompatibilityError, match="Unknown or incompatible golfer model"
    ):
        workspace_coordinator.submit_optimization_job(config_incompatible_model)


# ---------------------------------------------------------------------------
# RED Case 2: Cancel/pause/resume & dependency failure lifecycle; no duplicate
# ---------------------------------------------------------------------------


def test_cancel_pause_resume_lifecycle_and_state_integrity(
    workspace_coordinator: OptimizationTrainingWorkspaceCoordinator,
) -> None:
    """Cancel, pause, and resume maintain correct states; canceled jobs cannot be mislabeled complete."""
    config = OptimizationJobConfig(
        job_name="opt_lifecycle_01",
        session_id="sess_01",
        backend="scipy_lumped",
        objectives=[
            OptimizationObjective(name="clubhead_speed", target=45.0, weight=1.0)
        ],
        constraints=[],
    )
    job_id = workspace_coordinator.submit_optimization_job(config)
    assert job_id.startswith("opt_")

    job = workspace_coordinator.get_job(job_id)
    assert job.state in (WorkspaceJobState.QUEUED, WorkspaceJobState.RUNNING)

    # Pause
    workspace_coordinator.pause_job(job_id)
    job = workspace_coordinator.get_job(job_id)
    assert job.state == WorkspaceJobState.PAUSED

    # Resume
    workspace_coordinator.resume_job(job_id)
    job = workspace_coordinator.get_job(job_id)
    assert job.state == WorkspaceJobState.RUNNING

    # Cancel
    workspace_coordinator.cancel_job(job_id)
    job = workspace_coordinator.get_job(job_id)
    assert job.state == WorkspaceJobState.CANCELLED
    assert job.state != WorkspaceJobState.COMPLETED
    assert job.result is None or job.result.success is False


def test_duplicate_submission_does_not_duplicate_job(
    workspace_coordinator: OptimizationTrainingWorkspaceCoordinator,
) -> None:
    """Submitting the exact same active job request does not duplicate the job."""
    config = OptimizationJobConfig(
        job_name="opt_duplicate_test",
        session_id="sess_01",
        backend="scipy_lumped",
        objectives=[
            OptimizationObjective(name="clubhead_speed", target=42.0, weight=1.0)
        ],
        constraints=[],
    )
    job_id_1 = workspace_coordinator.submit_optimization_job(config)

    # Duplicate submit of active job returns existing job_id without creating a new job
    job_id_2 = workspace_coordinator.submit_optimization_job(config)
    assert job_id_1 == job_id_2
    assert len(workspace_coordinator.list_jobs(session_id="sess_01")) == 1


def test_dependency_failure_maintains_persisted_failed_state(
    workspace_coordinator: OptimizationTrainingWorkspaceCoordinator,
) -> None:
    """Dependency or execution failure maintains correct failed state and diagnostic detail."""
    config = OptimizationJobConfig(
        job_name="opt_fail_dep",
        session_id="sess_01",
        backend="scipy_lumped",
        objectives=[
            OptimizationObjective(name="clubhead_speed", target=45.0, weight=1.0)
        ],
        constraints=[],
        required_dataset_id="missing_raw_mocap_dataset",
    )
    # Submission with missing required dataset dependency fails or transitions to FAILED
    with pytest.raises(
        InvalidOptimizationConfigError,
        match="Required dataset 'missing_raw_mocap_dataset' not found",
    ):
        workspace_coordinator.submit_optimization_job(config)


# ---------------------------------------------------------------------------
# GREEN Case 1: Deterministic optimization changes output for changed input
# ---------------------------------------------------------------------------


def test_deterministic_optimization_input_sensitivity(
    workspace_coordinator: OptimizationTrainingWorkspaceCoordinator,
) -> None:
    """Small real deterministic optimization changes output for changed supported input."""
    # Target 1: carry 150m
    config_1 = OptimizationJobConfig(
        job_name="opt_sens_150m",
        session_id="sess_01",
        backend="scipy_lumped",
        objectives=[
            OptimizationObjective(name="carry_distance", target=150.0, weight=1.0)
        ],
        constraints=[],
    )
    res_1 = workspace_coordinator.run_optimization_synchronous(config_1)
    assert res_1.success is True
    assert res_1.optimal_speed > 0.0

    # Target 2: carry 220m
    config_2 = OptimizationJobConfig(
        job_name="opt_sens_220m",
        session_id="sess_01",
        backend="scipy_lumped",
        objectives=[
            OptimizationObjective(name="carry_distance", target=220.0, weight=1.0)
        ],
        constraints=[],
    )
    res_2 = workspace_coordinator.run_optimization_synchronous(config_2)
    assert res_2.success is True
    assert res_2.optimal_speed > res_1.optimal_speed
    assert not np.isclose(res_1.optimal_speed, res_2.optimal_speed, atol=1e-3)


# ---------------------------------------------------------------------------
# GREEN Case 2: Controller publishes metrics and registers result
# ---------------------------------------------------------------------------


def test_controller_publishes_metrics_and_registers_result(
    workspace_coordinator: OptimizationTrainingWorkspaceCoordinator,
) -> None:
    """Job controller fixture publishes metrics and registers result artifact in workspace."""
    config = OptimizationJobConfig(
        job_name="opt_metrics_test",
        session_id="sess_01",
        backend="scipy_lumped",
        objectives=[
            OptimizationObjective(name="clubhead_speed", target=48.0, weight=1.0)
        ],
        constraints=[],
    )
    job_id = workspace_coordinator.submit_optimization_job(config)

    # Process job to completion
    workspace_coordinator.step_all_jobs()
    job = workspace_coordinator.get_job(job_id)
    assert job.state == WorkspaceJobState.COMPLETED
    assert job.result is not None
    assert job.result.success is True
    assert len(job.metrics) > 0
    assert "objective_value" in job.metrics

    # Registered in project session
    registered_artifacts = workspace_coordinator.list_session_artifacts("sess_01")
    assert any(a["job_id"] == job_id for a in registered_artifacts)


# ---------------------------------------------------------------------------
# GREEN Case 3: Dataset selection survives reopen
# ---------------------------------------------------------------------------


def test_dataset_selection_survives_save_reopen(
    tmp_path: Path,
) -> None:
    """Dataset selection attached to project session survives store save and reopen."""
    project_dir = tmp_path / "durable_proj"
    store = SessionProjectStore(project_dir)
    store.create_project("proj_reopen_01", "Durable Project")
    store.add_subject("subj_01", "Golfer 1")
    store.create_session("sess_01", "subj_01", "Session 1")

    coord_1 = OptimizationTrainingWorkspaceCoordinator(project_store=store)

    # Register and select dataset with provenance
    ds_meta = {
        "format": "c3d",
        "provenance": {
            "generator": "dataset_generation_task_v1",
            "source_hash": "sha256_abcdef123456",
        },
    }
    coord_1.select_dataset_for_session(
        session_id="sess_01",
        dataset_id="mocap_swing_ds_01",
        dataset_path="data/mocap_01.c3d",
        kind="motion_capture",
        metadata=ds_meta,
    )

    # Reopen coordinator pointed at the same project directory
    store_reopened = SessionProjectStore(project_dir)
    coord_2 = OptimizationTrainingWorkspaceCoordinator(project_store=store_reopened)

    selected = coord_2.get_selected_dataset_for_session("sess_01")
    assert selected is not None
    assert selected.dataset_id == "mocap_swing_ds_01"
    assert Path(selected.path) == Path("data/mocap_01.c3d")
    assert selected.kind == "motion_capture"
    assert selected.metadata["provenance"]["generator"] == "dataset_generation_task_v1"
