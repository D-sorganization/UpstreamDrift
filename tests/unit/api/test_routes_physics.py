"""Unit tests for the physics API route."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.physics import router, clear_physics_caches
from src.api.dependencies import get_engine_manager, get_simulation_service

pytestmark = pytest.mark.unit


class MockEngine:
    def __init__(self):
        self.time = 1.0

    def get_state(self):
        import numpy as np

        return np.array([0.0, 0.0]), np.array([0.0, 0.0])

    def compute_gravity_forces(self):
        return [0.0, -9.80665]

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
    response = client.get("/simulation/control/forces")
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


def test_list_camera_presets(client: TestClient) -> None:
    """The presets enumeration matches the canonical preset registry (#7452)."""
    from src.api.models.requests import VALID_CAMERA_PRESETS
    from src.api.routes.physics import CAMERA_PRESETS

    response = client.get("/simulation/camera/presets")
    assert response.status_code == 200
    presets = response.json()["presets"]
    names = {p["preset"] for p in presets}
    assert names == set(CAMERA_PRESETS)
    # The enumeration must stay in sync with request validation
    assert names == VALID_CAMERA_PRESETS
    for preset in presets:
        assert len(preset["position"]) == 3
        assert len(preset["target"]) == 3
        assert len(preset["up"]) == 3


def test_control_recording(client: TestClient) -> None:
    """Test controlling trajectory recording."""
    response = client.post("/simulation/recording", json={"action": "start"})
    assert response.status_code == 200
    assert response.json()["recording"] is True


def test_recording_export_json_no_frames(client: TestClient) -> None:
    """JSON export with no recorded frames reports honestly, no file."""
    response = client.post(
        "/simulation/recording",
        json={"action": "export", "export_format": "json"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "No frames to export"
    assert data["export_path"] is None


def test_recording_export_json_writes_real_json(
    client: TestClient,
    mock_simulation_service,
    tmp_path,
    monkeypatch,
) -> None:
    """JSON export writes an actual .json artifact containing the frames."""
    import json

    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path))
    frames = [{"t": 0.0, "q": [0.0, 0.0]}, {"t": 0.001, "q": [0.1, 0.0]}]
    mock_simulation_service.stats.recorded_frames = frames

    response = client.post(
        "/simulation/recording",
        json={"action": "export", "export_format": "json"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["frame_count"] == 2
    assert data["export_path"] is not None
    assert data["export_path"].endswith(".json")

    artifact = tmp_path / data["export_path"]
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["frames"] == frames
    assert payload["format"] == "json"


@pytest.mark.parametrize("fmt", ["csv", "mat", "hdf5", "c3d"])
def test_recording_export_unimplemented_format_returns_501(
    client: TestClient, fmt: str
) -> None:
    """Non-JSON formats return the honest 501 + tracking_issue contract.

    Issue #7448: the endpoint previously wrote JSON content into a file named
    with the requested extension, fabricating these formats.
    """
    response = client.post(
        "/simulation/recording",
        json={"action": "export", "export_format": fmt},
    )
    assert response.status_code == 501
    body = response.json()
    assert fmt in body["detail"]
    assert isinstance(body["tracking_issue"], int)
    assert body["tracking_issue"] == 7451


def test_recording_export_unknown_format_rejected(client: TestClient) -> None:
    """Formats outside the recognized set are rejected at validation."""
    response = client.post(
        "/simulation/recording",
        json={"action": "export", "export_format": "xlsx"},
    )
    assert response.status_code == 422


def test_stop_recording_reports_frame_count(
    client: TestClient, mock_simulation_service: MockSimulationService
) -> None:
    """Stopping a recording reports the recorded frame count (#7452)."""
    client.post("/simulation/recording", json={"action": "start"})
    mock_simulation_service.stats.recorded_frames = [{"t": 0.0}, {"t": 0.1}]

    response = client.post("/simulation/recording", json={"action": "stop"})
    assert response.status_code == 200
    data = response.json()
    assert data["recording"] is False
    assert data["frame_count"] == 2
