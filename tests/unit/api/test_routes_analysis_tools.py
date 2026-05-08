"""Unit tests for the analysis tools API route."""

import pytest
from typing import Any
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.analysis_tools import router
from src.api.dependencies import get_engine_manager


class MockEngine:
    def __init__(self):
        self.time = 1.0

    def get_state(self):
        import numpy as np

        return np.array([0.1, 0.2]), np.array([0.01, 0.02])

    def compute_mass_matrix(self):
        import numpy as np

        return np.eye(2)

    def compute_jacobian(self, name: str):
        import numpy as np

        return {"linear": np.eye(2)}

    def get_body_position(self, name: str):
        if name == "A":
            return [0.0, 0.0, 0.0]
        if name == "B":
            return [1.0, 0.0, 0.0]
        return None

    def set_body_position(self, name: str, pos: list[float]):
        pass

    def set_body_rotation(self, name: str, rot: list[float]):
        pass

    def get_joint_names(self):
        return ["joint1", "joint2"]


class MockEngineManager:
    def __init__(self):
        self.engine = MockEngine()
        self._metric_history = []

    def get_active_physics_engine(self):
        return self.engine


@pytest.fixture
def mock_engine_manager():
    return MockEngineManager()


@pytest.fixture
def app(mock_engine_manager) -> FastAPI:
    """Create a FastAPI app with the analysis tools router."""
    test_app = FastAPI()
    test_app.include_router(router)
    test_app.dependency_overrides[get_engine_manager] = lambda: mock_engine_manager
    return test_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create a test client."""
    return TestClient(app)


def test_get_analysis_metrics(client: TestClient, mock_engine_manager) -> None:
    """Test getting real-time metrics."""
    response = client.get("/analysis/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "metrics" in data
    assert data["metrics"]["sim_time"] == 1.0
    assert len(mock_engine_manager._metric_history) == 1


def test_get_analysis_statistics(client: TestClient, mock_engine_manager) -> None:
    """Test getting metrics statistics."""
    # Seed history
    mock_engine_manager._metric_history = [
        {"sim_time": 0.0, "value": 1.0},
        {"sim_time": 1.0, "value": 2.0},
    ]
    response = client.get("/analysis/statistics")
    assert response.status_code == 200
    data = response.json()
    assert data["sample_count"] == 2
    assert "metrics" in data
    assert len(data["metrics"]) > 0


def test_export_analysis_data_csv(client: TestClient, mock_engine_manager) -> None:
    """Test exporting metrics as CSV."""
    mock_engine_manager._metric_history = [
        {"sim_time": 0.0, "value": 1.0},
        {"sim_time": 1.0, "value": 2.0},
    ]
    payload = {"format": "csv"}
    response = client.post("/analysis/export", json=payload)
    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    assert "sim_time" in response.text
    assert "value" in response.text


def test_export_analysis_data_json(client: TestClient, mock_engine_manager) -> None:
    """Test exporting metrics as JSON."""
    mock_engine_manager._metric_history = [
        {"sim_time": 0.0, "value": 1.0},
    ]
    payload = {"format": "json"}
    response = client.post("/analysis/export", json=payload)
    assert response.status_code == 200
    assert "application/json" in response.headers["content-type"]
    data = response.json()
    assert data["format"] == "json"
    assert data["record_count"] == 1


def test_set_body_position(client: TestClient) -> None:
    """Test setting body position."""
    payload = {"body_name": "A", "position": [1.0, 2.0, 3.0]}
    response = client.post("/simulation/position", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["body_name"] == "A"
    assert data["position"] == [1.0, 2.0, 3.0]


def test_measure_distance(client: TestClient) -> None:
    """Test measuring distance between two bodies."""
    payload = {"body_a": "A", "body_b": "B"}
    response = client.post("/simulation/measure", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["distance"] == 1.0
    assert data["delta"] == [1.0, 0.0, 0.0]


def test_get_measurement_tools(client: TestClient) -> None:
    """Test getting measurement tools."""
    response = client.get("/simulation/measurements")
    assert response.status_code == 200
    data = response.json()
    assert "joint_angles" in data
    assert len(data["joint_angles"]) == 2
    assert data["joint_angles"][0]["joint_name"] == "joint1"
