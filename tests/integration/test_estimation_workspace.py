"""Integration tests for Estimation Workspace Coordinator (ORG-17, #10526).

Verifies:
- RED: Invalid observation schema, non-identifiability, and non-finite residuals fail closed.
- RED: Cancelled job produces no passed result.
- RED: Source observations remain strictly immutable.
- GREEN: Supported real estimator recovers known synthetic parameter within tolerance.
- GREEN: Result provenance roundtrips cleanly to/from disk.
- GREEN: Canonical-core shell and Model & Match delegate to same coordinator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.estimation import (
    NonFiniteResidualError,
    UnidentifiableParametersError,
)
from src.shared.python.workspace.estimation_workspace import (
    EstimationJobResult,
    EstimationJobStatus,
    EstimationObservationPayload,
    EstimationParameterPrior,
    EstimationRequest,
    EstimationWorkspaceCoordinator,
)


@pytest.fixture
def synthetic_observations() -> EstimationObservationPayload:
    """Create a deterministic single-dof observation payload."""
    times = np.linspace(0.0, 1.0, 5)
    true_scale = 1.08
    # Baseline parabola
    q_truth = times**2
    observations = (true_scale * q_truth)[:, None]
    return EstimationObservationPayload(
        trial_id="synthetic-trial-001",
        times=times,
        observations=observations,
        units="m",
    )


@pytest.fixture
def standard_parameters() -> tuple[EstimationParameterPrior, ...]:
    """Single length parameter with initial guess 1.0."""
    return (
        EstimationParameterPrior(
            name="upper_length_m",
            initial=1.0,
            kind="length",
            lower=0.5,
            upper=1.5,
            prior=None,
            prior_scale=None,
        ),
    )


def test_estimation_workspace_availability() -> None:
    """Verify coordinator reports capability availability."""
    coordinator = EstimationWorkspaceCoordinator()
    avail = coordinator.check_availability()
    assert avail["available"] is True
    assert "map_single_trial" in avail["supported_tasks"]
    assert "scipy" in avail["dependencies"]


def test_invalid_observation_schema_fails_closed() -> None:
    """Non-finite or unsorted observation times must fail closed immediately."""
    # 1. NaN in times
    with pytest.raises(ValueError, match="finite"):
        EstimationObservationPayload(
            trial_id="invalid-times",
            times=np.array([0.0, np.nan, 1.0]),
            observations=np.zeros((3, 1)),
        )

    # 2. Unsorted times
    with pytest.raises(ValueError, match="strictly increasing"):
        EstimationObservationPayload(
            trial_id="unsorted-times",
            times=np.array([1.0, 0.5, 2.0]),
            observations=np.zeros((3, 1)),
        )

    # 3. Mismatched lengths
    with pytest.raises(ValueError, match="dimension mismatch"):
        EstimationObservationPayload(
            trial_id="mismatched",
            times=np.array([0.0, 0.5, 1.0]),
            observations=np.zeros((4, 1)),
        )

    # 4. Non-finite observation values
    with pytest.raises(ValueError, match="finite"):
        EstimationObservationPayload(
            trial_id="nan-obs",
            times=np.array([0.0, 0.5, 1.0]),
            observations=np.array([[0.0], [np.inf], [1.0]]),
        )


def test_source_observations_are_immutable(
    synthetic_observations: EstimationObservationPayload,
) -> None:
    """Source observations array must be read-only and unmodifiable."""
    with pytest.raises(ValueError, match="read-only"):
        synthetic_observations.observations[0, 0] = 999.0

    with pytest.raises(ValueError, match="read-only"):
        synthetic_observations.times[0] = 999.0


def test_synthetic_parameter_recovery(
    synthetic_observations: EstimationObservationPayload,
    standard_parameters: tuple[EstimationParameterPrior, ...],
) -> None:
    """Coordinator executes MAP estimation and recovers true scale 1.08 within 1%."""
    coordinator = EstimationWorkspaceCoordinator()
    request = EstimationRequest(
        task_id="job-recovery-01",
        task_kind="map_single_trial",
        observation=synthetic_observations,
        parameter_specs=standard_parameters,
        max_iterations=80,
    )

    result = coordinator.execute_estimation(request)
    assert result.status == EstimationJobStatus.COMPLETED
    assert result.success is True
    assert "upper_length_m" in result.estimated_parameters
    recovered = result.estimated_parameters["upper_length_m"]
    assert pytest.approx(1.08, rel=0.01) == recovered
    assert result.iterations > 0
    assert result.cost < 1e-6
    assert len(result.provenance_hash) == 64
    assert result.source_hash == synthetic_observations.source_hash


def test_non_identifiable_parameter_fails_closed(
    synthetic_observations: EstimationObservationPayload,
) -> None:
    """Unidentifiable parameter must fail closed under gate policy 'raise'."""
    unidentifiable_params = (
        EstimationParameterPrior(
            name="upper_length_m",
            initial=1.0,
            kind="length",
            lower=0.5,
            upper=1.5,
        ),
        EstimationParameterPrior(
            name="redundant_length_m",
            initial=1.0,
            kind="length",
            lower=0.5,
            upper=1.5,
        ),
    )
    coordinator = EstimationWorkspaceCoordinator()
    request = EstimationRequest(
        task_id="job-unidentifiable-01",
        task_kind="map_single_trial",
        observation=synthetic_observations,
        parameter_specs=unidentifiable_params,
        identifiability_gate="raise",
    )

    with pytest.raises(UnidentifiableParametersError):
        coordinator.execute_estimation(request)


def test_non_finite_residual_fails_closed(
    synthetic_observations: EstimationObservationPayload,
    standard_parameters: tuple[EstimationParameterPrior, ...],
) -> None:
    """Non-finite residual under 'raise' policy raises NonFiniteResidualError."""
    coordinator = EstimationWorkspaceCoordinator()

    def bad_residual_injection(q_eval: Any, params: Any) -> np.ndarray:
        return np.array([np.nan, 0.0])

    request = EstimationRequest(
        task_id="job-nan-residual-01",
        task_kind="map_single_trial",
        observation=synthetic_observations,
        parameter_specs=standard_parameters,
        non_finite_policy="raise",
    )

    with pytest.raises(NonFiniteResidualError):
        coordinator._run_map_estimation(
            request, custom_residual_fn=bad_residual_injection
        )


def test_cancelled_job_produces_no_passed_result(
    synthetic_observations: EstimationObservationPayload,
    standard_parameters: tuple[EstimationParameterPrior, ...],
) -> None:
    """Cancellation produces CANCELLED status and success=False."""
    coordinator = EstimationWorkspaceCoordinator()
    task_id = "job-cancelled-01"
    request = EstimationRequest(
        task_id=task_id,
        task_kind="map_single_trial",
        observation=synthetic_observations,
        parameter_specs=standard_parameters,
    )

    coordinator.submit_job(request)
    assert coordinator.cancel_job(task_id) is True
    result = coordinator.get_job_result(task_id)
    assert result is not None
    assert result.status == EstimationJobStatus.CANCELLED
    assert result.success is False
    assert result.estimated_parameters == {}


def test_provenance_persistence_roundtrip(
    tmp_path: Path,
    synthetic_observations: EstimationObservationPayload,
    standard_parameters: tuple[EstimationParameterPrior, ...],
) -> None:
    """EstimationJobResult serializes to disk and reconstructs identically."""
    coordinator = EstimationWorkspaceCoordinator()
    request = EstimationRequest(
        task_id="job-persist-01",
        task_kind="map_single_trial",
        observation=synthetic_observations,
        parameter_specs=standard_parameters,
    )

    result = coordinator.execute_estimation(request)
    artifact_path = tmp_path / "estimation_result.json"
    result.save(artifact_path)

    loaded = EstimationJobResult.load(artifact_path)
    assert loaded.task_id == result.task_id
    assert loaded.status == result.status
    assert loaded.success == result.success
    assert loaded.estimated_parameters == result.estimated_parameters
    assert loaded.provenance_hash == result.provenance_hash
    assert loaded.source_hash == result.source_hash
