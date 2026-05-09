"""Unit tests for the physics API route."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.physics import router, clear_physics_caches
from src.api.dependencies import get_engine_manager, get_simulation_service


class MockEngine:
    def __init__(self):
        self.time = 1.0

    def get_state(self):
        import numpy as np

        return np.array([0.0, 0.0]), np.array([0.0, 0.0])

    def compute_gravity_forces(self):
        return [0.0, -9.81]

    def compute_contact_forces(self):
        return [0.0, 0.0]

    def compute_bias_forces(self):
        return [0.0, 0.0]

    def compute_jacobian(self, name):
        import numpy as np

        return {"linear": np.array([[1.0, 0.0], [0.0, 1.0]])}

    def compute_mass_matrix(self):
        import numpy as np

        return np.array([[1.0, 0.0], [0.0, 1.0]])


class MockEngineManager:
    def get_active_physics_engine(self):
        return MockEngine()


class MockSimStats:
    def __init__(self):
        self.start_time = 0.0
        self.frame_count = 100
        self.speed_factor = 1.0
        self.is_recording = False
        self.recorded_frames = []


class MockSimulationService:
    def __init__(self):
        self.stats = MockSimStats()

    def set_speed_factor(self, speed):
        self.stats.speed_factor = speed

    def start_recording(self):
        self.stats.is_recording = True

    def stop_recording(self):
        self.stats.is_recording = False


@pytest.fixture
def mock_engine_manager():
    return MockEngineManager()


@pytest.fixture
def mock_simulation_service():
    return MockSimulationService()


@pytest.fixture
def app(mock_engine_manager, mock_simulation_service) -> FastAPI:
    """Create a FastAPI app with the physics router."""
    test_app = FastAPI()
    test_app.include_router(router)
    test_app.dependency_overrides[get_engine_manager] = lambda: mock_engine_manager
    test_app.dependency_overrides[get_simulation_service] = lambda: (
        mock_simulation_service
    )
    clear_physics_caches()
    return test_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create a test client."""
    return TestClient(app)


def test_get_forces(client: TestClient) -> None:
    """Test getting forces."""
    response = client.get("/simulation/forces")
    assert response.status_code == 200
    data = response.json()
    assert data["sim_time"] == 1.0
    assert "gravity_forces" in data


def test_get_metrics(client: TestClient) -> None:
    """Test getting biomechanics metrics."""
    response = client.get("/simulation/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "sim_time" in data
    assert "joint_positions" in data


def test_get_simulation_stats(client: TestClient) -> None:
    """Test getting simulation stats."""
    response = client.get("/simulation/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["sim_time"] == 1.0
    assert data["speed_factor"] == 1.0


def test_set_simulation_speed(client: TestClient) -> None:
    """Test setting simulation speed."""
    response = client.post("/simulation/speed", json={"speed_factor": 2.0})
    assert response.status_code == 200
    data = response.json()
    assert data["speed_factor"] == 2.0


def test_set_camera_preset(client: TestClient) -> None:
    """Test setting camera preset."""
    response = client.post("/simulation/camera", json={"preset": "side"})
    assert response.status_code == 200
    data = response.json()
    assert data["preset"] == "side"
    assert "position" in data


def test_control_recording(client: TestClient) -> None:
    """Test controlling trajectory recording."""
    response = client.post("/simulation/recording", json={"action": "start"})
    assert response.status_code == 200
    assert response.json()["recording"] is True
