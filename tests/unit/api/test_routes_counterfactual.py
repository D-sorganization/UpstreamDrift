"""Unit tests for the counterfactual analysis API routes (issue #7450).

Covers:
- capability/kinds endpoint (data-driven gating),
- async task flow (started -> completed with a serialized
  ``CounterfactualResult``) reusing the /simulate/status machinery,
- 409 on missing session and unsupported kind, 422 on unknown kind.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.dependencies import get_simulation_service, get_task_manager
from src.api.routes.analysis import router as analysis_router
from src.api.routes.simulation import router as simulation_router
from src.api.services.simulation_service import SimulationService

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

N_FRAMES = 8
N_JOINTS = 2


class StubEngine:
    """Engine stub exposing the full counterfactual compute surface."""

    engine_type = "pendulum"
    model_name_str = "StubPendulum"

    def set_state(self, q: np.ndarray, v: np.ndarray) -> None:
        pass

    def set_control(self, u: np.ndarray) -> None:
        pass

    def forward(self) -> None:
        pass

    def compute_ztcf(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.full(N_JOINTS, 1.0)

    def compute_zvcf(self, q: np.ndarray) -> np.ndarray:
        return np.full(N_JOINTS, 2.0)

    def compute_drift_acceleration(self) -> np.ndarray:
        return np.full(N_JOINTS, 3.0)

    def compute_control_acceleration(self, tau: np.ndarray) -> np.ndarray:
        return np.full(N_JOINTS, 4.0)


class PartialEngine(StubEngine):
    """Engine stub missing the ZVCF method (gated out conservatively)."""

    compute_zvcf = None  # type: ignore[assignment]


class StubRecorder:
    """Recorder stub with recorded frames but no stored counterfactuals."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.times = np.linspace(0.0, 1.0, N_FRAMES)
        self.counterfactuals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.induced: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def get_time_series(self, field_name: str) -> tuple[np.ndarray, np.ndarray]:
        return np.array([]), np.array([])

    def get_counterfactual_series(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        return self.counterfactuals.get(name, (np.array([]), np.array([])))

    def get_induced_acceleration_series(
        self, name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.induced.get(name, (np.array([]), np.array([])))

    def compute_analysis_post_hoc(self) -> None:
        cf = np.tile(np.arange(N_JOINTS, dtype=float) + 1.0, (N_FRAMES, 1))
        self.counterfactuals["ztcf"] = (self.times, cf)
        self.counterfactuals["zvcf"] = (self.times, 2.0 * cf)
        self.induced["total"] = (self.times, 3.0 * cf)


class MockTaskManager:
    """Synchronous TaskManager mock (#4843 compatibility contract)."""

    def __init__(self) -> None:
        self.tasks: dict[str, dict[str, Any]] = {}

    def exists(self, task_id: str) -> bool:
        return task_id in self.tasks

    def get(self, task_id: str) -> dict[str, Any] | None:
        return self.tasks.get(task_id)

    def set(self, task_id: str, value: dict[str, Any]) -> None:
        self.tasks[task_id] = value


def _make_service(recorder: StubRecorder | None) -> SimulationService:
    service = SimulationService(engine_manager=MagicMock())
    service._last_recorder = recorder  # noqa: SLF001 - test seam
    return service


@pytest.fixture
def task_manager() -> MockTaskManager:
    return MockTaskManager()


def _make_client(
    service: SimulationService, task_manager: MockTaskManager
) -> TestClient:
    from src.api.rate_limit import limiter

    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(analysis_router)
    app.include_router(simulation_router)
    app.dependency_overrides[get_simulation_service] = lambda: service
    app.dependency_overrides[get_task_manager] = lambda: task_manager
    return TestClient(app)


# ----------------------------------------------------------------------
# Capability gating
# ----------------------------------------------------------------------


def test_kinds_endpoint_full_engine(task_manager: MockTaskManager) -> None:
    client = _make_client(_make_service(StubRecorder(StubEngine())), task_manager)
    response = client.get("/analysis/counterfactual/kinds")
    assert response.status_code == 200
    data = response.json()
    assert data["session_available"] is True
    assert data["engine"] == "pendulum"
    assert data["kinds"] == [
        "control",
        "drift",
        "gravity",
        "total",
        "ztcf",
        "zvcf",
    ]


def test_kinds_endpoint_partial_engine_gated_out(
    task_manager: MockTaskManager,
) -> None:
    """Engines missing any post-hoc method are conservatively unsupported."""
    client = _make_client(_make_service(StubRecorder(PartialEngine())), task_manager)
    data = client.get("/analysis/counterfactual/kinds").json()
    assert data["kinds"] == []
    assert data["session_available"] is True


def test_kinds_endpoint_no_session(task_manager: MockTaskManager) -> None:
    client = _make_client(_make_service(None), task_manager)
    data = client.get("/analysis/counterfactual/kinds").json()
    assert data == {"kinds": [], "engine": None, "session_available": False}


# ----------------------------------------------------------------------
# POST /analysis/counterfactual — task flow
# ----------------------------------------------------------------------


def test_counterfactual_happy_path_task_flow(
    task_manager: MockTaskManager,
) -> None:
    """started -> completed with serialized CounterfactualResult payload."""
    client = _make_client(_make_service(StubRecorder(StubEngine())), task_manager)

    response = client.post("/analysis/counterfactual", json={"kind": "ztcf"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "started"
    assert body["kind"] == "ztcf"
    task_id = body["task_id"]

    # TestClient runs BackgroundTasks before returning — poll the shared
    # /simulate/status machinery for the completed result.
    status = client.get(f"/simulate/status/{task_id}")
    assert status.status_code == 200
    data = status.json()
    assert data["status"] == "completed"
    result = data["result"]
    assert result["kind"] == "ztcf"
    assert result["units"] == "rad/s^2"
    assert len(result["times"]) == N_FRAMES
    assert len(result["values"]) == N_FRAMES
    assert result["values"][0] == [1.0, 2.0]
    assert result["metadata"]["n_frames"] == N_FRAMES


def test_counterfactual_induced_kind(task_manager: MockTaskManager) -> None:
    client = _make_client(_make_service(StubRecorder(StubEngine())), task_manager)
    task_id = client.post("/analysis/counterfactual", json={"kind": "total"}).json()[
        "task_id"
    ]
    data = client.get(f"/simulate/status/{task_id}").json()
    assert data["status"] == "completed"
    assert data["result"]["kind"] == "total"
    assert data["result"]["values"][0] == [3.0, 6.0]


def test_counterfactual_unknown_kind_is_422(
    task_manager: MockTaskManager,
) -> None:
    client = _make_client(_make_service(StubRecorder(StubEngine())), task_manager)
    response = client.post("/analysis/counterfactual", json={"kind": "warp_drive"})
    assert response.status_code == 422
    assert "warp_drive" in response.text


def test_counterfactual_unsupported_kind_is_409(
    task_manager: MockTaskManager,
) -> None:
    client = _make_client(_make_service(StubRecorder(PartialEngine())), task_manager)
    response = client.post("/analysis/counterfactual", json={"kind": "ztcf"})
    assert response.status_code == 409
    assert "does not support" in response.json()["detail"]


def test_counterfactual_no_session_is_409(task_manager: MockTaskManager) -> None:
    client = _make_client(_make_service(None), task_manager)
    response = client.post("/analysis/counterfactual", json={"kind": "ztcf"})
    assert response.status_code == 409
    assert "No completed simulation session" in response.json()["detail"]


def test_failed_compute_marks_task_failed(task_manager: MockTaskManager) -> None:
    """Engine errors during compute surface as a failed task, not a 500."""

    class ExplodingRecorder(StubRecorder):
        def compute_analysis_post_hoc(self) -> None:
            raise RuntimeError("engine exploded")

    client = _make_client(_make_service(ExplodingRecorder(StubEngine())), task_manager)
    task_id = client.post("/analysis/counterfactual", json={"kind": "ztcf"}).json()[
        "task_id"
    ]
    data = client.get(f"/simulate/status/{task_id}").json()
    assert data["status"] == "failed"
    assert "engine exploded" in data["error"]
