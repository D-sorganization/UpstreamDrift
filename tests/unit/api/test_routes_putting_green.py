"""Unit tests for the putting green API route."""

import pytest
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.putting_green import router
from src.api.dependencies import get_engine_manager, get_task_manager


class MockGreen:
    def __init__(self):
        self.size_x = 10.0
        self.size_y = 10.0
        self.resolution = 0.1
        self.hole_position = np.array([5.0, 5.0])
        self.hole_radius = 0.054

    def get_height(self, x, y):
        return 0.0

    def get_slope(self, x, y):
        return np.array([0.0, 0.0])


class MockEngine:
    def __init__(self):
        self.green = MockGreen()


class MockEngineManager:
    async def get_engine(self, engine_type):
        return MockEngine()


class MockTaskManager:
    def __init__(self):
        self.tasks = {
            "existing_putt": {"status": "completed", "result": {"success": True}}
        }

    async def exists(self, task_id):
        return task_id in self.tasks

    async def get(self, task_id):
        return self.tasks.get(task_id)

    async def set(self, task_id, value):
        self.tasks[task_id] = value


@pytest.fixture
def mock_engine_manager():
    return MockEngineManager()


@pytest.fixture
def mock_task_manager():
    return MockTaskManager()


@pytest.fixture
def app(mock_engine_manager, mock_task_manager) -> FastAPI:
    test_app = FastAPI()
    from src.api.rate_limit import limiter

    test_app.state.limiter = limiter
    test_app.include_router(router)
    test_app.dependency_overrides[get_engine_manager] = lambda: mock_engine_manager
    test_app.dependency_overrides[get_task_manager] = lambda: mock_task_manager
    return test_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


def test_get_green_contours(client: TestClient) -> None:
    response = client.get("/tools/putting-green/contours")
    assert response.status_code == 200, response.text
    data = response.json()
    assert "x_grid" in data or "grid_x" in data
    assert "y_grid" in data or "grid_y" in data
    assert "z_grid" in data or "elevations" in data
    assert "hole_position" in data


def test_simulate_putt(client: TestClient) -> None:
    payload = {
        "start_position": [1.0, 1.0],
        "direction_x": 1.0,
        "direction_y": 1.0,
        "speed": 2.0,
        "green_speed_stimp": 10.0,
    }
    response = client.post("/tools/putting-green/simulate", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert "positions" in data
    assert "holed" in data
