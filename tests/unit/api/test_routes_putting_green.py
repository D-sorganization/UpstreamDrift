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


def _runs(modes: list[str], times: list[float]) -> list[dict[str, object]]:
    """Where each run of a mode starts, read off the per-sample modes."""
    return [
        {"mode": mode, "t_s": times[i], "index": i}
        for i, mode in enumerate(modes)
        if i == 0 or modes[i - 1] != mode
    ]


def test_simulate_reports_the_mode_and_phase_marks_of_every_sample(
    client: TestClient,
) -> None:
    """#14473: playback jumps come from the core's own per-sample modes."""
    data = client.post(
        "/tools/putting-green/simulate",
        json={"direction_x": 0.0, "direction_y": 1.0, "speed": 1.8},
    ).json()
    modes = data["modes"]
    assert len(modes) == len(data["positions"]) == len(data["times"])
    assert {"slide", "roll"} <= set(modes)
    assert data["phases"] == _runs(modes, data["times"])


def test_simulate_3d_reports_the_phase_marks_of_its_samples(
    client: TestClient,
) -> None:
    samples = client.post("/tools/putting-green/simulate-3d", json={}).json()
    modes = [s["mode"] for s in samples["samples"]]
    times = [s["t_s"] for s in samples["samples"]]
    assert samples["phases"] == _runs(modes, times)


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


def test_read_green_returns_the_simulator_inputs_in_one_call(
    client: TestClient,
) -> None:
    """#14465: pace, heading and outcome of the core's solve_aim, not a recipe."""
    from src.api.routes.putting_green import (
        _build_putting_surface,
        _default_putter,
        _to_surface_frame,
    )
    from src.shared.python.putting_dynamics import solve_aim

    body = {"ball_x": 10.0, "ball_y": 5.0, "target_x": 10.0, "target_y": 9.0}
    reading = client.post("/tools/putting-green/read-green", json=body).json()
    surface, half_w, half_h = _build_putting_surface(10.0, 20.0, 20.0)
    want = solve_aim(
        surface,
        _to_surface_frame(10.0, 5.0, half_w, half_h),
        _to_surface_frame(10.0, 9.0, half_w, half_h),
        _default_putter(1.0),
    )
    assert reading["recommended_speed"] == pytest.approx(want.putter_speed_mps)
    assert reading["aim_heading_deg"] == pytest.approx(math.degrees(want.heading_rad))
    assert reading["aim_heading_deg"] == pytest.approx(90.0, abs=0.1)
    assert (reading["holed"], reading["converged"]) == (want.holed, want.converged)


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


# -- Putter balance presets (#14461) --


def test_putter_presets_lists_the_core_presets(client: TestClient) -> None:
    from src.shared.python.putting_dynamics import BALANCE_PRESETS

    response = client.get("/tools/putting-green/putter-presets")
    assert response.status_code == 200, response.text
    presets = {p["key"]: p for p in response.json()["presets"]}
    assert set(presets) == set(BALANCE_PRESETS)
    blade = BALANCE_PRESETS["toe_hang"]
    assert presets["toe_hang"]["moi_kg_m2"] == blade.moi_kg_m2
    assert presets["toe_hang"]["hosel_toe_m"] == blade.hosel_toe_m


def test_simulate_putt_3d_balance_type_matches_the_core_strike(
    client: TestClient,
) -> None:
    from src.shared.python.putting_dynamics import balance_preset, strike

    response = client.post(
        "/tools/putting-green/simulate-3d",
        json={
            "balance_type": "toe_hang",
            "putter_speed_mps": 1.8,
            "loft_deg": 3.0,
            "coefficient_of_restitution": 0.78,
            "impact_toe_m": -0.006,
        },
    )
    assert response.status_code == 200, response.text
    putter = balance_preset("toe_hang").putter(1.8, loft_deg=3.0, cor=0.78)
    want = strike(putter, -0.006)
    got = response.json()["collision"]
    assert got["face_twist_rad_s"] == pytest.approx(want.face_twist_rad_s)
    assert got["ball_speed_mps"] == pytest.approx(want.ball_speed_mps)


def test_simulate_putt_3d_rejects_an_unknown_balance_type(client: TestClient) -> None:
    response = client.post(
        "/tools/putting-green/simulate-3d", json={"balance_type": "anser"}
    )
    assert response.status_code == 422


# -- Green shapes on every route (#14463) --


def _core_rest(green: dict, speed: float, stimp: float = 10.0) -> tuple[float, float]:
    """Golden: the shared core's rest point for the /simulate defaults."""
    from src.shared.python.putting_dynamics import (
        GreenSpec,
        PutterState,
        green_preset,
        simulate_ball,
        strike,
        struck_ball_state,
    )

    spec = green_preset(green["preset"]) if "preset" in green else GreenSpec(**green)
    surface = spec.surface(stimp_ft=stimp, extent_m=20.0)
    report = strike(PutterState.typical(speed))
    # Green-local (5, 10) aimed along +y, hole at (10, 15); surface is centred.
    ball = struck_ball_state(report, x_m=-5.0, y_m=0.0, heading_rad=math.pi / 2)
    result = simulate_ball(ball, surface, hole_x_m=0.0, hole_y_m=5.0, collision=report)
    return result.rest_x_m + 10.0, result.rest_y_m + 10.0


@pytest.mark.parametrize(
    "green", [{"preset": "severe_slopes"}, {"grade_percent": 3.0, "aspect_deg": 0.0}]
)
def test_simulate_rolls_on_the_requested_green(client: TestClient, green: dict) -> None:
    body = {"direction_x": 0.0, "direction_y": 1.0, "speed": 1.5, "green": green}
    response = client.post("/tools/putting-green/simulate", json=body)
    assert response.status_code == 200, response.text
    final = response.json()["final_position"]
    assert final == pytest.approx(list(_core_rest(green, 1.5)), abs=1e-9)
    assert final[0] > 5.05, "a green falling toward +x breaks the ball right"


def test_contours_show_the_requested_green(client: TestClient) -> None:
    from src.shared.python.putting_dynamics import green_preset

    response = client.get(
        "/tools/putting-green/contours",
        params={"resolution": 5, "preset": "tiered"},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    height = green_preset("tiered").height(extent_m=20.0)
    for xs, ys, got in zip(
        data["grid_x"], data["grid_y"], data["elevations"], strict=True
    ):
        want = [
            height.elevation(x - 10.0, y - 10.0) for x, y in zip(xs, ys, strict=True)
        ]
        assert got == pytest.approx(want, abs=1e-12)
    assert data["elevations"][0][0] > data["elevations"][-1][0], "upper tier at -y"


def test_read_green_reads_the_break_of_a_sloped_green(client: TestClient) -> None:
    # A short putt keeps the aim search quick on a sloped green.
    putt = {"ball_x": 9.0, "ball_y": 12.0, "target_x": 10.0, "target_y": 15.0}
    flat = client.post("/tools/putting-green/read-green", json=putt).json()
    sloped = client.post(
        "/tools/putting-green/read-green",
        json={**putt, "green": {"grade_percent": 2.0, "aspect_deg": 0.0}},
    ).json()
    assert flat["total_break"] == pytest.approx(0.0, abs=1e-3)
    assert sloped["total_break"] > 0.05


def test_scatter_rolls_on_an_undulating_green(client: TestClient) -> None:
    body = {"direction_x": 0.0, "direction_y": 1.0, "n_simulations": 3}
    flat = client.post("/tools/putting-green/scatter", json=body).json()
    bumpy = client.post(
        "/tools/putting-green/scatter",
        json={**body, "green": {"preset": "undulating_championship"}},
    ).json()
    assert bumpy["final_positions"] != flat["final_positions"]


def test_green_presets_route_lists_the_core_presets(client: TestClient) -> None:
    from src.shared.python.putting_dynamics import GREEN_PRESETS

    response = client.get("/tools/putting-green/green-presets")
    assert response.status_code == 200, response.text
    presets = response.json()
    assert set(presets) == set(GREEN_PRESETS)
    assert presets["tiered"]["tier_height_m"] == GREEN_PRESETS["tiered"].tier_height_m


@pytest.mark.parametrize(
    ("method", "path", "kwargs"),
    [
        ("post", "/simulate", {"json": {"green": {"preset": "augusta"}}}),
        ("post", "/read-green", {"json": {"green": {"grade_percent": 12.0}}}),
        ("get", "/contours", {"params": {"preset": "augusta"}}),
    ],
)
def test_unknown_or_out_of_range_greens_are_refused(
    client: TestClient, method: str, path: str, kwargs: dict
) -> None:
    response = getattr(client, method)(f"/tools/putting-green{path}", **kwargs)
    assert response.status_code == 422, response.text


@pytest.mark.parametrize("preset", ["tiered", "undulating_championship"])
def test_simulate_3d_rolls_on_the_core_green_preset(
    client: TestClient, preset: str
) -> None:
    from src.shared.python.putting_dynamics import (
        FrictionField,
        FrictionParams,
        green_preset,
        stimp_to_rolling_mu,
    )
    from src.shared.python.putting_dynamics.types import (
        TYPICAL_HEAD_MASS_KG,
        TYPICAL_HEAD_MOI_KG_M2,
        TYPICAL_LOFT_DEG,
    )

    spec = green_preset(preset)
    response = client.post(
        "/tools/putting-green/simulate-3d",
        json={"green": {"preset": preset}, "putter_speed_mps": 2.0},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    surface = SurfaceSpec(
        height=spec.height(20.0),
        friction_field=FrictionField.uniform(extent_m=20.0),
        friction=FrictionParams(mu_roll0=stimp_to_rolling_mu(10.0)),
    )
    want = simulate_strike(
        PutterState(
            head_mass_kg=TYPICAL_HEAD_MASS_KG,
            moi_kg_m2=TYPICAL_HEAD_MOI_KG_M2,
            loft_deg=TYPICAL_LOFT_DEG,
            speed_mps=2.0,
            cor=0.78,
        ),
        surface,
        hole_x_m=3.0,
        hole_y_m=0.0,
    )
    assert data["total_distance_m"] == pytest.approx(want.total_distance_m)
    end = data["samples"][-1]
    assert (end["x_m"], end["y_m"]) == pytest.approx(
        (want.samples[-1].x_m, want.samples[-1].y_m)
    )
    assert data["surface"]["grade_percent"] == pytest.approx(spec.grade_percent)
    assert data["surface"]["downhill_aspect_deg"] == pytest.approx(spec.aspect_deg)


def test_simulate_3d_strikes_along_the_aim_heading(client: TestClient) -> None:
    from src.shared.python.putting_dynamics.types import (
        TYPICAL_HEAD_MASS_KG,
        TYPICAL_HEAD_MOI_KG_M2,
        TYPICAL_LOFT_DEG,
    )

    response = client.post(
        "/tools/putting-green/simulate-3d",
        json={"aim_heading_deg": 30.0, "grade_percent": 2.0, "hole_y_m": 2.0},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    putter = PutterState(
        head_mass_kg=TYPICAL_HEAD_MASS_KG,
        moi_kg_m2=TYPICAL_HEAD_MOI_KG_M2,
        loft_deg=TYPICAL_LOFT_DEG,
        speed_mps=1.8,
        cor=0.78,
    )
    from src.api.routes.putting_green import (
        Putt3DSimulationRequest,
        _build_putting_3d_surface,
    )

    surface, _, _ = _build_putting_3d_surface(
        Putt3DSimulationRequest(grade_percent=2.0, hole_y_m=2.0)
    )
    want = simulate_strike(
        putter, surface, heading_rad=math.radians(30.0), hole_x_m=3.0, hole_y_m=2.0
    )
    end = data["samples"][-1]
    assert (end["x_m"], end["y_m"]) == pytest.approx(
        (want.samples[-1].x_m, want.samples[-1].y_m)
    )
    assert end["y_m"] > 0.5, "a 30° aim leaves the +x line"


def test_simulate_3d_rejects_an_aim_beyond_180(client: TestClient) -> None:
    response = client.post(
        "/tools/putting-green/simulate-3d", json={"aim_heading_deg": 181.0}
    )
    assert response.status_code == 422


def test_solve_aim_3d_is_the_core_read_on_the_simulator_surface(
    client: TestClient,
) -> None:
    from src.api.routes.putting_green import (
        Putt3DSimulationRequest,
        _build_putting_3d_surface,
    )
    from src.shared.python.putting_dynamics import solve_aim

    body = {
        "green": {"preset": "tiered"},
        "balance_type": "counterbalanced",
        "hole_x_m": 4.0,
        "hole_y_m": 0.5,
    }
    response = client.post("/tools/putting-green/solve-aim-3d", json=body)
    assert response.status_code == 200, response.text
    got = response.json()
    surface, _, putter = _build_putting_3d_surface(
        Putt3DSimulationRequest.model_validate(body)
    )
    want = solve_aim(surface, (0.0, 0.0), (4.0, 0.5), putter)
    assert got["putter_speed_mps"] == pytest.approx(want.putter_speed_mps)
    assert got["aim_heading_deg"] == pytest.approx(math.degrees(want.heading_rad))
    assert got["break_m"] == pytest.approx(want.break_m)
    assert (got["holed"], got["converged"]) == (want.holed, want.converged)


def test_the_solved_read_holes_on_simulate_3d(client: TestClient) -> None:
    body = {"grade_percent": 2.0, "downhill_aspect_deg": 90.0, "hole_x_m": 3.0}
    read = client.post("/tools/putting-green/solve-aim-3d", json=body).json()
    assert read["holed"]
    putt = client.post(
        "/tools/putting-green/simulate-3d",
        json={
            **body,
            "putter_speed_mps": read["putter_speed_mps"],
            "aim_heading_deg": read["aim_heading_deg"],
        },
    ).json()
    assert putt["holed"]


def test_counter_torque_serves_the_core_stroke_series(client: TestClient) -> None:
    """Golden parity (#14469): every column is the core couple model's value."""
    from dataclasses import asdict

    from src.shared.python.putting_dynamics import counter_torque as ct

    from src.shared.python.putting_dynamics import BALANCE_PRESETS

    body = {
        "preload_nm": 0.9,
        "noise_scale": 0.4,
        "putter": "counterbalanced",
        "n_points": 30,
    }
    response = client.post("/tools/putting-green/counter-torque", json=body)
    assert response.status_code == 200, response.text
    data = response.json()
    inputs = ct.CounterTorqueInputs(
        preload_nm=0.9, noise_scale=0.4, putter="counterbalanced"
    )
    core = [asdict(s) for s in ct.stroke_series(inputs, n_points=30)]
    assert data["putter_name"] == BALANCE_PRESETS["counterbalanced"].name
    for key, column in data["samples"].items():
        assert column == pytest.approx([row[key] for row in core]), key
    assert set(data["samples"]) == set(core[0])


@pytest.mark.parametrize(
    "body", [{"preload_nm": -1.0}, {"putter": "anser"}, {"n_points": 0}]
)
def test_counter_torque_rejects_bad_inputs(client: TestClient, body: dict) -> None:
    response = client.post("/tools/putting-green/counter-torque", json=body)
    assert response.status_code in (400, 422), response.text
