"""Unit tests for the actuator controls API route."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.actuator_controls import router
from src.api.dependencies import get_engine_manager


class MockEngine:
    def __init__(self):
        self.engine_type = "mock_engine"
        self.joint_names = ["hip", "knee"]

    def get_state(self):
        return {"torques": [0.0, 0.0]}

    def get_joint_limits(self):
        return [(-1.0, 1.0), (-2.0, 2.0)]

    def set_control(self, index, value):
        pass


class MockEngineManager:
    def __init__(self):
        self.engine = MockEngine()

    def get_active_engine(self):
        return self.engine


@pytest.fixture
def mock_engine_manager():
    return MockEngineManager()


@pytest.fixture
def app(mock_engine_manager) -> FastAPI:
    """Create a FastAPI app with the actuator controls router."""
    test_app = FastAPI()
    test_app.include_router(router)
    test_app.dependency_overrides[get_engine_manager] = lambda: mock_engine_manager
    return test_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create a test client."""
    return TestClient(app)


def test_get_actuator_panel(client: TestClient) -> None:
    """Test getting actuator panel."""
    response = client.get("/simulation/actuators")
    assert response.status_code == 200
    data = response.json()
    assert data["n_actuators"] == 2
    assert data["engine_name"] == "mock_engine"
    assert len(data["actuators"]) == 2
    assert data["actuators"][0]["name"] == "hip"
    assert data["actuators"][0]["min_value"] == -1.0


def test_send_actuator_command(client: TestClient) -> None:
    """Test sending actuator command."""
    payload = {"actuator_index": 0, "value": 0.5, "control_type": "constant"}
    response = client.post("/simulation/actuators", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["actuator_index"] == 0
    assert data["applied_value"] == 0.5
    assert data["clamped"] is False


def test_send_actuator_command_clamped(client: TestClient) -> None:
    """Test sending actuator command that requires clamping."""
    payload = {
        "actuator_index": 0,
        "value": 2.0,  # Max is 1.0
        "control_type": "constant",
    }
    response = client.post("/simulation/actuators", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["applied_value"] == 1.0
    assert data["clamped"] is True


def test_send_actuator_command_out_of_range(client: TestClient) -> None:
    """Test sending command to invalid index."""
    payload = {"actuator_index": 5, "value": 0.0, "control_type": "constant"}
    response = client.post("/simulation/actuators", json=payload)
    assert response.status_code == 400


def test_send_actuator_batch(client: TestClient) -> None:
    """Test sending batch actuator command."""
    payload = {
        "commands": [
            {"actuator_index": 0, "value": 0.5, "control_type": "constant"},
            {
                "actuator_index": 1,
                "value": -3.0,  # Min is -2.0
                "control_type": "constant",
            },
        ]
    }
    response = client.post("/simulation/actuators/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["applied_value"] == 0.5
    assert data[0]["clamped"] is False
    assert data[1]["applied_value"] == -2.0
    assert data[1]["clamped"] is True
