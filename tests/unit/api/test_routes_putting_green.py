"""Unit tests for the putting green API route."""

import pytest
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.putting_green import router
from src.api.dependencies import get_engine_manager, get_task_manager

pytestmark = pytest.mark.unit


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


def test_simulate_putt_3d_exposes_collision_and_motion_modes(
    client: TestClient,
) -> None:
    """The R3F endpoint returns one self-contained playback payload."""
    response = client.post(
        "/tools/putting-green/simulate-3d",
        json={
            "putter_speed_mps": 1.8,
            "loft_deg": 3.0,
            "stimp_rating": 10.0,
            "hole_x_m": 3.0,
            "hole_y_m": 0.0,
            "hosel_toe_m": -0.025,
            "hosel_forward_m": 0.005,
        },
    )

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["samples"][0]["mode"] in {"airborne", "slide"}
    assert data["samples"][-1]["mode"] in {"rest", "roll"}
    assert data["collision"]["ball_speed_mps"] > 0
    assert data["collision"]["putter_speed_after_mps"] < 1.8
    assert data["collision"]["contact_time_proxy_s"] > 0
    assert data["surface"]["width_m"] > 0
    assert data["surface"]["height_m"] > 0


def test_simulate_putt_3d_rejects_hosel_outside_head(
    client: TestClient,
) -> None:
    response = client.post(
        "/tools/putting-green/simulate-3d",
        json={"hosel_toe_m": 0.081},
    )

    assert response.status_code == 422


# -- ADR-0045 F1 roll-model provenance (issue #9343) --


def test_simulate_putt_names_the_ud_legacy_roll_model(client: TestClient) -> None:
    """Every UD-engine putt response names the model that produced it."""
    response = client.post(
        "/tools/putting-green/simulate",
        json={"direction_x": 0.0, "direction_y": 1.0, "speed": 2.0},
    )
    assert response.status_code == 200, response.text
    assert response.json()["roll_model"] == "ud-legacy-roll/1"


def test_scatter_names_the_ud_legacy_roll_model(client: TestClient) -> None:
    response = client.post(
        "/tools/putting-green/scatter",
        json={"direction_x": 0.0, "direction_y": 1.0, "n_simulations": 2},
    )
    assert response.status_code == 200, response.text
    assert response.json()["roll_model"] == "ud-legacy-roll/1"


def test_read_green_names_the_ud_legacy_roll_model(client: TestClient) -> None:
    response = client.post("/tools/putting-green/read-green", json={})
    assert response.status_code == 200, response.text
    assert response.json()["roll_model"] == "ud-legacy-roll/1"


def test_simulate_3d_names_the_usga_stimp_roll_model(client: TestClient) -> None:
    """The 3-D route runs the preserved counterpart law, and says so."""
    response = client.post("/tools/putting-green/simulate-3d", json={})
    assert response.status_code == 200, response.text
    assert response.json()["roll_model"] == "usga-stimp-roll/1"


def test_the_two_routes_report_different_models(client: TestClient) -> None:
    """ADR-0045: the divergence is named, never silent."""
    legacy = client.post(
        "/tools/putting-green/simulate",
        json={"direction_x": 0.0, "direction_y": 1.0},
    ).json()["roll_model"]
    usga = client.post("/tools/putting-green/simulate-3d", json={}).json()["roll_model"]
    assert legacy != usga


def test_response_models_refuse_an_unnamed_result() -> None:
    """Fail-closed at the API boundary: roll_model has no default."""
    from pydantic import ValidationError

    from src.api.routes.putting_green import PuttSimulationResponse

    with pytest.raises(ValidationError):
        PuttSimulationResponse(
            positions=[[0.0, 0.0]],
            velocities=[[0.0, 0.0]],
            times=[0.0],
            holed=False,
            final_position=[0.0, 0.0],
            total_distance=0.0,
            duration=0.0,
        )
