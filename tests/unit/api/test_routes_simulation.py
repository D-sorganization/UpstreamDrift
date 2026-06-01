"""Unit tests for the simulation API route."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.simulation import router
from src.api.dependencies import get_simulation_service, get_task_manager


class MockSimulationService:
    async def run_simulation(self, request):
        return {"success": True, "duration": 1.5, "frames": 0, "data": {"states": []}}

    def run_simulation_background(self, task_id, request, task_manager):
        pass


class MockTaskManager:
    def __init__(self):
        self.tasks = {"existing_sim": {"status": "completed"}}

    async def exists(self, task_id):
        return task_id in self.tasks

    async def get(self, task_id):
        return self.tasks.get(task_id)

    async def set(self, task_id, value):
        self.tasks[task_id] = value


@pytest.fixture
def mock_simulation_service():
    return MockSimulationService()


@pytest.fixture
def mock_task_manager():
    return MockTaskManager()


@pytest.fixture
def app(mock_simulation_service, mock_task_manager) -> FastAPI:
    """Create a FastAPI app with the simulation router."""
    from src.api.rate_limit import limiter

    test_app = FastAPI()
    test_app.state.limiter = limiter
    test_app.include_router(router)
    test_app.dependency_overrides[get_simulation_service] = lambda: (
        mock_simulation_service
    )
    test_app.dependency_overrides[get_task_manager] = lambda: mock_task_manager
    return test_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


def test_run_simulation(client: TestClient) -> None:
    payload = {"engine_type": "mujoco", "parameters": {}, "return_trajectory": True}
    response = client.post("/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


def test_run_simulation_async(client: TestClient) -> None:
    payload = {"engine_type": "mujoco", "parameters": {}, "return_trajectory": True}
    response = client.post("/simulate/async", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert data["status"] == "started"


def test_get_simulation_status(client: TestClient) -> None:
    response = client.get("/simulate/status/existing_sim")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"


def test_get_simulation_status_not_found(client: TestClient) -> None:
    response = client.get("/simulate/status/nonexistent")
    assert response.status_code == 404


# --- Error-handling: do not leak raw exception text to clients (issue #6947) ---

_SECRET = "/internal/secret/path leaked detail"


class _RaisingService:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def run_simulation(self, request):
        raise self._exc

    def run_simulation_background(self, task_id, request, task_manager):
        pass


@pytest.mark.parametrize(
    ("exc", "status"),
    [
        (ValueError(_SECRET), 400),
        (RuntimeError(_SECRET), 500),
        (ImportError(_SECRET), 500),
    ],
)
def test_error_branches_return_generic_detail(
    mock_task_manager, exc: Exception, status: int
) -> None:
    """ValueError/RuntimeError/ImportError yield generic, non-leaky detail."""
    from src.api.rate_limit import limiter

    test_app = FastAPI()
    test_app.state.limiter = limiter
    test_app.include_router(router)
    test_app.dependency_overrides[get_simulation_service] = lambda: _RaisingService(exc)
    test_app.dependency_overrides[get_task_manager] = lambda: mock_task_manager
    client = TestClient(test_app, raise_server_exceptions=False)

    payload = {"engine_type": "mujoco", "parameters": {}}
    response = client.post("/simulate", json=payload)

    assert response.status_code == status
    detail = response.json()["detail"]
    assert _SECRET not in detail
    assert "secret" not in detail.lower()
