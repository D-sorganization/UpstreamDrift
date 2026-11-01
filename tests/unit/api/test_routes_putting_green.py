"""Unit tests for the putting green API route.

``fastapi``/``httpx`` may be unavailable locally (see AGENTS.md); the whole
module skips in that case and only runs in CI. The physics-only assertions
below (closed-form rollout, scatter determinism) are also exercised directly
against ``src.shared.python.putting_dynamics`` in
``tests/unit/putting_dynamics/test_solver.py``, which runs without fastapi.
"""

import math

import numpy as np
import pytest

pytestmark = pytest.mark.unit

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.api.dependencies import get_engine_manager, get_task_manager  # noqa: E402
from src.api.routes.putting_green import router  # noqa: E402
from src.shared.python.core.physics_constants import GOLF_BALL_MASS_KG  # noqa: E402
from src.shared.python.putting_dynamics import (  # noqa: E402
    PutterState,
    SurfaceSpec,
    simulate_strike,
)


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


def test_get_green_contours_flat_green_has_zero_elevations(
    client: TestClient,
) -> None:
    """The shared flat HeightField reproduces the legacy flat default."""
    response = client.get(
        "/tools/putting-green/contours",
        params={"width": 20.0, "height": 20.0, "resolution": 5},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert all(e == 0.0 for row in data["elevations"] for e in row)
    assert data["hole_position"] == [10.0, 10.0]


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
    assert "velocities" in data
    assert "times" in data
    assert "holed" in data
    assert "final_position" in data
    assert "total_distance" in data
    assert "duration" in data


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


# -- ADR-0045 F1 roll-model provenance (issue #9343; migrated #14141 step 7a) --


def test_simulate_putt_names_the_usga_stimp_roll_model(client: TestClient) -> None:
    """``/simulate`` now runs the shared core and names its roll model."""
    response = client.post(
        "/tools/putting-green/simulate",
        json={"direction_x": 0.0, "direction_y": 1.0, "speed": 2.0},
    )
    assert response.status_code == 200, response.text
    assert response.json()["roll_model"] == "usga-stimp-roll/1"


def test_scatter_names_the_usga_stimp_roll_model(client: TestClient) -> None:
    """``/scatter`` now runs the shared core and names its roll model."""
    response = client.post(
        "/tools/putting-green/scatter",
        json={"direction_x": 0.0, "direction_y": 1.0, "n_simulations": 2},
    )
    assert response.status_code == 200, response.text
    assert response.json()["roll_model"] == "usga-stimp-roll/1"


def test_read_green_names_the_usga_stimp_roll_model(client: TestClient) -> None:
    """``/read-green`` now runs the shared core too (epic #14141 step 7c)."""
    response = client.post("/tools/putting-green/read-green", json={})
    assert response.status_code == 200, response.text
    assert response.json()["roll_model"] == "usga-stimp-roll/1"


def test_simulate_3d_names_the_usga_stimp_roll_model(client: TestClient) -> None:
    """The 3-D route runs the preserved counterpart law, and says so."""
    response = client.post("/tools/putting-green/simulate-3d", json={})
    assert response.status_code == 200, response.text
    assert response.json()["roll_model"] == "usga-stimp-roll/1"


def test_every_roll_out_route_names_the_same_model(client: TestClient) -> None:
    """ADR-0045: after step 7c no route runs the legacy UD engine any more.

    Every roll-out response names the shared core's model, so there is no
    longer a divergence to call out — the regression to guard against is a
    route silently drifting back onto a different (unnamed) model.
    """
    read_green = client.post("/tools/putting-green/read-green", json={}).json()[
        "roll_model"
    ]
    simulate = client.post(
        "/tools/putting-green/simulate",
        json={"direction_x": 0.0, "direction_y": 1.0},
    ).json()["roll_model"]
    scatter = client.post(
        "/tools/putting-green/scatter",
        json={"direction_x": 0.0, "direction_y": 1.0, "n_simulations": 1},
    ).json()["roll_model"]
    simulate_3d = client.post("/tools/putting-green/simulate-3d", json={}).json()[
        "roll_model"
    ]
    assert {read_green, simulate, scatter, simulate_3d} == {"usga-stimp-roll/1"}


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


# -- Wind is not modelled by the shared core (epic #14141 step 7a) --


def test_simulate_rejects_positive_wind_speed(client: TestClient) -> None:
    response = client.post(
        "/tools/putting-green/simulate",
        json={
            "direction_x": 0.0,
            "direction_y": 1.0,
            "speed": 2.0,
            "wind_speed": 3.0,
        },
    )
    assert response.status_code == 400, response.text


# -- Closed-form physics on the shared core (epic #14141 step 7a) --


def test_simulate_putt_launch_obeys_the_impact_closed_form(client: TestClient) -> None:
    """``speed`` is putter-head speed, as in the legacy engine.

    The ball cannot leave faster than the one-dimensional impact limit
    ``(1 + e) M v / (M + m)``, and the putt must match the shared strike
    path used by ``/simulate-3d`` for the same head and green.
    """
    speed = 1.5
    response = client.post(
        "/tools/putting-green/simulate",
        json={
            "ball_x": 5.0,
            "ball_y": 3.0,
            "speed": speed,
            "direction_x": 0.0,
            "direction_y": 1.0,
            "stimp_rating": 10.0,
            "hole_x": 10.0,
            "hole_y": 19.0,
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()
    head_m = 0.35
    ball_m = float(GOLF_BALL_MASS_KG)
    impact_limit = (1.0 + 0.78) * head_m * speed / (head_m + ball_m)
    assert math.hypot(*data["velocities"][0]) <= impact_limit + 1e-9

    reference = simulate_strike(
        PutterState(
            head_mass_kg=head_m, moi_kg_m2=4.5e-4, loft_deg=3.0, speed_mps=speed
        ),
        SurfaceSpec.flat_uniform(stimp_ft=10.0, extent_m=20.0),
    )
    assert data["total_distance"] == pytest.approx(reference.total_distance_m, rel=1e-6)
    assert not data["holed"]

    speeds = [math.hypot(vx, vy) for vx, vy in data["velocities"]]
    assert all(a >= b - 1e-6 for a, b in zip(speeds, speeds[1:], strict=False))
    for vx, vy in data["velocities"]:
        assert vx == pytest.approx(0.0, abs=1e-9)
        assert vy >= -1e-9


# -- Scatter determinism (epic #14141 step 7a) --


def test_scatter_is_deterministic(client: TestClient) -> None:
    payload = {
        "direction_x": 0.0,
        "direction_y": 1.0,
        "speed": 2.0,
        "n_simulations": 5,
        "speed_variance": 0.1,
        "direction_variance_deg": 2.0,
    }
    first = client.post("/tools/putting-green/scatter", json=payload).json()
    second = client.post("/tools/putting-green/scatter", json=payload).json()
    assert first["final_positions"] == second["final_positions"]
    assert first["holed_count"] == second["holed_count"]
    assert first["average_distance_from_hole"] == pytest.approx(
        second["average_distance_from_hole"]
    )


def test_scatter_response_shape_is_unchanged(client: TestClient) -> None:
    response = client.post(
        "/tools/putting-green/scatter",
        json={"direction_x": 0.0, "direction_y": 1.0, "n_simulations": 3},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    for key in (
        "roll_model",
        "final_positions",
        "holed_count",
        "total_simulations",
        "average_distance_from_hole",
        "make_percentage",
    ):
        assert key in data
    assert data["total_simulations"] == 3
    assert len(data["final_positions"]) == 3


def test_read_green_on_a_flat_green_aims_at_the_target(client: TestClient) -> None:
    """A flat green has no break, so the solved aim point is the target."""
    response = client.post(
        "/tools/putting-green/read-green",
        json={"ball_x": 5.0, "ball_y": 5.0, "target_x": 8.0, "target_y": 9.0},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["distance"] == pytest.approx(5.0)
    assert data["total_break"] == pytest.approx(0.0, abs=0.01)
    assert data["aim_point"] == pytest.approx([8.0, 9.0], abs=0.01)
    assert data["recommended_speed"] > 0.0
    assert all(e == 0.0 for e in data["elevations"])
