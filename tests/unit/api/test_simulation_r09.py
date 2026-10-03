"""Regression and contract tests for Simulation API error handling (R09, issue #11149).

Verifies:
1. Real service-to-route error propagation produces appropriate HTTP status codes
   (400 for invalid inputs, 503 for unavailable engines, 500 for numerical/runtime failures,
   504 for timeouts) instead of swallowing into reasonless 200 responses.
2. Background jobs persist structured, machine-readable failure reason and stage in active tasks.
3. Completed calculation with failed saving remains recoverable in memory with
   calculation_status="completed", persistence_status="failed", export_paths=[], and retry guidance.
4. Required vs optional analysis channel outcomes are explicitly distinguished and labeled.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.dependencies import get_logger, get_simulation_service, get_task_manager
from src.api.models.requests import SimulationRequest
from src.api.routes.simulation import router
from src.api.services.simulation_service import SimulationService
from src.shared.python.core.error_utils import (
    EngineLaunchError,
    EngineNotAvailableError,
    PhysicsSimulationError,
)
from src.shared.python.engine_core.engine_manager import EngineManager

pytestmark = [pytest.mark.anyio, pytest.mark.unit]


class InMemoryTaskManager:
    """Synchronous task manager double for testing background tasks."""

    def __init__(self) -> None:
        self.tasks: dict[str, dict[str, Any]] = {}

    def exists(self, task_id: str) -> bool:
        return task_id in self.tasks

    def get(self, task_id: str) -> dict[str, Any] | None:
        return self.tasks.get(task_id)

    def set(self, task_id: str, value: dict[str, Any]) -> None:
        self.tasks[task_id] = value


@pytest.fixture
def mock_engine_manager() -> MagicMock:
    manager = MagicMock(spec=EngineManager)
    manager._load_engine = MagicMock()
    mock_engine = MagicMock()
    mock_engine.step = MagicMock()
    mock_engine.get_time = MagicMock(return_value=0.0)
    mock_engine.get_state = MagicMock(return_value=([0.0], [0.0]))
    manager.get_active_physics_engine = MagicMock(return_value=mock_engine)
    return manager


@pytest.fixture
def task_manager() -> InMemoryTaskManager:
    return InMemoryTaskManager()


def _disable_all_limiters(app: FastAPI, routes_list: list[Any]) -> None:
    from src.api.rate_limit import limiter
    import src.api.routes.simulation as sim_route

    for candidate in [
        limiter,
        getattr(sim_route, "limiter", None),
        getattr(app.state, "limiter", None),
    ]:
        if candidate is not None:
            candidate.enabled = False
            if hasattr(candidate, "reset"):
                try:
                    candidate.reset()
                except (AttributeError, RuntimeError):
                    pass
            storage = getattr(candidate, "_storage", None)
            if storage is not None:
                if hasattr(storage, "storage") and hasattr(storage.storage, "clear"):
                    storage.storage.clear()
                if hasattr(storage, "events") and hasattr(storage.events, "clear"):
                    storage.events.clear()
                if hasattr(storage, "reset"):
                    try:
                        storage.reset()
                    except (AttributeError, RuntimeError):
                        pass

    for route in routes_list:
        ep = getattr(route, "endpoint", None)
        while ep is not None:
            if hasattr(ep, "__closure__") and ep.__closure__:
                for cell in ep.__closure__:
                    obj = cell.cell_contents
                    if hasattr(obj, "enabled"):
                        obj.enabled = False
                    if hasattr(obj, "reset"):
                        try:
                            obj.reset()
                        except (AttributeError, RuntimeError):
                            pass
                    storage = getattr(obj, "_storage", None)
                    if storage is not None:
                        if hasattr(storage, "storage") and hasattr(
                            storage.storage, "clear"
                        ):
                            storage.storage.clear()
                        if hasattr(storage, "events") and hasattr(
                            storage.events, "clear"
                        ):
                            storage.events.clear()
                        if hasattr(storage, "reset"):
                            try:
                                storage.reset()
                            except (AttributeError, RuntimeError):
                                pass
            if hasattr(ep, "__globals__"):
                glob_limiter = ep.__globals__.get("limiter")
                if glob_limiter is not None:
                    glob_limiter.enabled = False
                    if hasattr(glob_limiter, "reset"):
                        try:
                            glob_limiter.reset()
                        except (AttributeError, RuntimeError):
                            pass
            ep = getattr(ep, "__wrapped__", None)


@pytest.fixture
def app_with_service(
    mock_engine_manager: MagicMock,
    task_manager: InMemoryTaskManager,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[FastAPI, SimulationService]:
    from src.api.rate_limit import limiter

    service = SimulationService(mock_engine_manager)
    test_app = FastAPI()
    test_app.state.limiter = limiter
    test_app.include_router(router)
    _disable_all_limiters(test_app, list(test_app.routes) + list(router.routes))
    monkeypatch.setattr(limiter, "enabled", False)
    test_app.dependency_overrides[get_simulation_service] = lambda: service
    test_app.dependency_overrides[get_task_manager] = lambda: task_manager
    test_app.dependency_overrides[get_logger] = lambda: None
    return test_app, service


def test_real_service_invalid_parameters_yields_400(
    app_with_service: tuple[FastAPI, SimulationService],
) -> None:
    """An invalid parameter (e.g. negative timestep or timestep > duration) yields HTTP 400."""
    app, _ = app_with_service
    client = TestClient(app, raise_server_exceptions=False)

    payload = {
        "engine_type": "mujoco",
        "duration": 0.05,
        "timestep": 0.1,  # Invalid: timestep > duration triggers service ValueError
    }
    response = client.post("/simulate", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "timestep" in data["detail"].lower() or "invalid" in data["detail"].lower()
    assert response.headers.get("X-Error-Code") == "invalid_input"
    assert response.headers.get("X-Error-Stage") == "preparation"


def test_real_service_engine_unavailable_yields_503(
    app_with_service: tuple[FastAPI, SimulationService],
    mock_engine_manager: MagicMock,
) -> None:
    """When the requested engine is not available, route yields HTTP 503."""
    app, _ = app_with_service
    mock_engine_manager._load_engine.side_effect = EngineNotAvailableError(
        "drake", "simulation"
    )
    mock_engine_manager.get_active_physics_engine.return_value = None
    client = TestClient(app, raise_server_exceptions=False)

    payload = {"engine_type": "drake", "duration": 0.01, "timestep": 0.001}
    response = client.post("/simulate", json=payload)
    assert response.status_code == 503
    assert response.headers.get("X-Error-Code") == "engine_unavailable"
    assert response.headers.get("X-Error-Stage") == "preparation"


def test_real_service_numerical_failure_yields_500(
    app_with_service: tuple[FastAPI, SimulationService],
) -> None:
    """Numerical solver diverged during simulation yields HTTP 500 with numerical_failure code."""
    app, service = app_with_service
    client = TestClient(app, raise_server_exceptions=False)

    with patch.object(
        service,
        "_execute_simulation_loop",
        side_effect=PhysicsSimulationError("Solver diverged with nonfinite state"),
    ):
        payload = {"engine_type": "mujoco", "duration": 0.01, "timestep": 0.001}
        response = client.post("/simulate", json=payload)
        assert response.status_code == 500
        assert response.headers.get("X-Error-Code") == "numerical_failure"
        assert response.headers.get("X-Error-Stage") == "execution"


def test_real_service_timeout_yields_504(
    app_with_service: tuple[FastAPI, SimulationService],
) -> None:
    """Timeout during execution yields HTTP 504."""
    app, service = app_with_service
    client = TestClient(app, raise_server_exceptions=False)

    with patch.object(
        service,
        "_execute_simulation_loop",
        side_effect=TimeoutError("Integration step timed out"),
    ):
        payload = {"engine_type": "mujoco", "duration": 0.01, "timestep": 0.001}
        response = client.post("/simulate", json=payload)
        assert response.status_code == 504
        assert response.headers.get("X-Error-Code") == "timeout"
        assert response.headers.get("X-Error-Stage") == "execution"


async def test_background_simulation_retains_machine_readable_error(
    app_with_service: tuple[FastAPI, SimulationService],
    task_manager: InMemoryTaskManager,
) -> None:
    """Background failure records machine-readable error_code, error_stage, and error_info."""
    _, service = app_with_service
    request = SimulationRequest.model_validate(
        {
            "engine_type": "mujoco",
            "duration": 0.01,
            "timestep": 0.05,  # Invalid parameter: timestep > duration triggers ValueError
        }
    )

    await service.run_simulation_background("task-r09-bg-err", request, task_manager)

    task_record = task_manager.get("task-r09-bg-err")
    assert task_record is not None
    assert task_record["status"] == "failed"
    assert task_record["error_code"] == "invalid_input"
    assert task_record["error_stage"] == "preparation"
    assert "error_info" in task_record
    assert task_record["error_info"]["code"] == "invalid_input"
    assert task_record["error_info"]["retriable"] is False


async def test_persistence_failure_preserves_memory_results_and_marks_failed(
    mock_engine_manager: MagicMock,
    tmp_path: Any,
) -> None:
    """When calculation succeeds but file persistence fails, calculation_status is completed,
    persistence_status is failed, export_paths is empty, and in-memory results are retained."""
    service = SimulationService(mock_engine_manager)

    # Configure mock recorder output
    mock_rec = MagicMock()
    mock_rec.is_recording = True
    mock_rec.buffer_exhausted = False
    mock_rec.start = MagicMock()
    mock_rec.stop = MagicMock()
    mock_rec.get_data_dict = MagicMock(
        return_value={
            "times": np.array([0.0, 0.001]),
            "joint_positions": np.array([[0.0], [0.1]]),
            "joint_velocities": np.array([[0.0], [0.0]]),
            "joint_accelerations": np.array([[0.0], [0.0]]),
        }
    )
    mock_rec.get_time_series = MagicMock(
        side_effect=lambda name: (np.array([0.0, 0.001]), np.array([[0.0], [0.1]]))
    )

    with (
        patch(
            "src.api.services.simulation_service.GenericPhysicsRecorder",
            return_value=mock_rec,
        ),
        patch.object(
            service.output_manager,
            "save_simulation_results",
            side_effect=OSError("Disk quota exceeded [Errno 122]"),
        ),
    ):
        request = SimulationRequest.model_validate(
            {"engine_type": "mujoco", "duration": 0.001, "timestep": 0.001}
        )
        response = await service.run_simulation(request, raise_on_error=False)

        assert response.success is True
        assert response.calculation_status == "completed"
        assert response.persistence_status == "failed"
        assert response.export_paths == []
        assert len(response.data["times"]) == 2
        assert response.error is not None
        assert response.error.code == "persistence_failed"
        assert response.error.stage == "persistence"
        assert response.error.retriable is True
        assert response.error.retry_guidance is not None
        assert "memory" in response.error.retry_guidance.lower()

        # In-memory session must be preserved for explicit retry
        session = service.get_session_recording()
        assert session is not None
        rec, meta = session
        assert rec is mock_rec
        assert meta["engine"] == "mujoco"


async def test_analysis_partial_status_and_optional_channel_labeling(
    mock_engine_manager: MagicMock,
) -> None:
    """When analysis is requested and some optional channels are unavailable,
    analysis_status is 'partial' and missing channels are explicitly labeled."""
    service = SimulationService(mock_engine_manager)

    mock_rec = MagicMock()
    mock_rec.is_recording = True
    mock_rec.buffer_exhausted = False
    mock_rec.start = MagicMock()
    mock_rec.stop = MagicMock()
    mock_rec.get_data_dict = MagicMock(
        return_value={
            "times": np.array([0.0, 0.001]),
            "joint_positions": np.array([[0.0], [0.1]]),
            "joint_velocities": np.array([[0.0], [0.0]]),
            "joint_accelerations": np.array([[0.0], [0.0]]),
        }
    )

    def fake_get_time_series(name: str) -> Any:
        if name == "ztcf_accel":
            return np.array([0.0, 0.001]), np.array([1.2, 1.4])
        if name == "zvcf_accel":
            raise KeyError(f"Optional channel '{name}' not recorded")
        return np.array([0.0, 0.001]), np.array([0.0, 0.0])

    mock_rec.get_time_series = MagicMock(side_effect=fake_get_time_series)

    with patch(
        "src.api.services.simulation_service.GenericPhysicsRecorder",
        return_value=mock_rec,
    ):
        request = SimulationRequest.model_validate(
            {
                "engine_type": "mujoco",
                "duration": 0.001,
                "timestep": 0.001,
                "analysis_config": {"ztcf": True, "zvcf": True},
            }
        )

        response = await service.run_simulation(request, raise_on_error=False)

        assert response.success is True
        assert response.analysis_status == "partial"
        assert response.analysis_results is not None
        assert "ztcf_acceleration" in response.analysis_results
        assert "_channel_status" in response.analysis_results
        assert (
            response.analysis_results["_channel_status"]["ztcf_acceleration"]
            == "available"
        )
        assert (
            "unavailable"
            in response.analysis_results["_channel_status"]["zvcf_acceleration"]
        )
