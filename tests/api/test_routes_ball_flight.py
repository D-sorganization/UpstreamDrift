"""Tests for the ball-flight REST route."""

from __future__ import annotations

import math

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.api.routes.ball_flight import BallFlightSimulationRequest, router
from src.shared.python.physics.flight_models import FlightModelType

pytestmark = pytest.mark.integration


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _driver_payload(model_name: str) -> dict[str, float | str]:
    return {
        "ball_speed_mps": 70.0,
        "launch_angle_deg": 12.0,
        "azimuth_angle_deg": 1.0,
        "spin_rate_rpm": 2600.0,
        "spin_axis_tilt_deg": -2.0,
        "wind_speed_mps": 0.0,
        "wind_direction_deg": 0.0,
        "model_name": model_name,
        "max_time_s": 1.0,
        "time_step_s": 0.05,
    }


@pytest.mark.parametrize("model_type", list(FlightModelType))
def test_simulate_ball_flight_happy_path_per_model(
    client: TestClient, model_type: FlightModelType
) -> None:
    response = client.post(
        "/tools/ball-flight/simulate", json=_driver_payload(model_type.value)
    )

    assert response.status_code == 200
    data = response.json()
    assert data["model_name"]
    assert data["trajectory"]
    assert data["summary"]["carry_m"] > 0.0
    assert data["summary"]["apex_m"] > 0.0
    assert data["summary"]["flight_time_s"] > 0.0
    assert "lateral_deviation_m" in data["summary"]


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("ball_speed_mps", 0.0),
        ("launch_angle_deg", 95.0),
        ("spin_rate_rpm", -1.0),
        ("wind_speed_mps", 45.0),
        ("time_step_s", 0.0),
    ],
)
def test_simulate_ball_flight_rejects_invalid_ranges(
    client: TestClient, field: str, bad_value: float
) -> None:
    payload = _driver_payload(FlightModelType.WATERLOO_PENNER.value)
    payload[field] = bad_value

    response = client.post("/tools/ball-flight/simulate", json=payload)

    assert response.status_code == 422


def test_simulate_ball_flight_rejects_time_step_larger_than_max_time() -> None:
    with pytest.raises(ValidationError):
        BallFlightSimulationRequest(max_time_s=0.25, time_step_s=0.5)


def test_simulate_ball_flight_rejects_invalid_model_name(client: TestClient) -> None:
    payload = _driver_payload("not_a_model")

    response = client.post("/tools/ball-flight/simulate", json=payload)

    assert response.status_code == 422


# =============================================================================
# GET /models — shared flight-model enumeration (issue #7456)
# =============================================================================


def test_list_models_enumerates_full_registry(client: TestClient) -> None:
    response = client.get("/tools/ball-flight/models")

    assert response.status_code == 200
    models = response.json()["models"]
    keys = [m["key"] for m in models]
    assert keys == [mt.value for mt in FlightModelType], (
        "GET /models must enumerate the same registry the desktop tracer uses"
    )
    for model in models:
        assert model["name"]
        assert model["description"]
        assert model["reference"]


# =============================================================================
# Multi-model simulate (issue #7456)
# =============================================================================


def test_simulate_multi_model_returns_per_model_results(client: TestClient) -> None:
    payload = _driver_payload(FlightModelType.WATERLOO_PENNER.value)
    payload["models"] = [
        FlightModelType.WATERLOO_PENNER.value,
        FlightModelType.NATHAN.value,
    ]

    response = client.post("/tools/ball-flight/simulate", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert [r["model_key"] for r in data["results"]] == payload["models"]
    # Back-compat: top-level fields mirror the first requested model.
    assert data["model_key"] == FlightModelType.WATERLOO_PENNER.value
    assert data["trajectory"] == data["results"][0]["trajectory"]
    assert data["summary"] == data["results"][0]["summary"]


def test_simulate_single_model_response_is_backwards_compatible(
    client: TestClient,
) -> None:
    payload = _driver_payload(FlightModelType.NATHAN.value)

    response = client.post("/tools/ball-flight/simulate", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["model_key"] == FlightModelType.NATHAN.value
    assert len(data["results"]) == 1
    assert data["results"][0]["model_key"] == FlightModelType.NATHAN.value


def test_simulate_multi_model_deduplicates_preserving_order(
    client: TestClient,
) -> None:
    payload = _driver_payload(FlightModelType.WATERLOO_PENNER.value)
    payload["models"] = [
        FlightModelType.NATHAN.value,
        FlightModelType.NATHAN.value,
        FlightModelType.BALLANTYNE.value,
    ]

    response = client.post("/tools/ball-flight/simulate", json=payload)

    assert response.status_code == 200
    assert [r["model_key"] for r in response.json()["results"]] == [
        FlightModelType.NATHAN.value,
        FlightModelType.BALLANTYNE.value,
    ]


def test_simulate_rejects_empty_models_list(client: TestClient) -> None:
    payload = _driver_payload(FlightModelType.WATERLOO_PENNER.value)
    payload["models"] = []

    response = client.post("/tools/ball-flight/simulate", json=payload)

    assert response.status_code == 422


# =============================================================================
# Structural golden tests (issue #7456)
#
# The flight models have open physics-accuracy issues (#7403-#7405), so these
# assert structural correctness (monotonic time, finite values, returns to
# ground) rather than TrackMan golden numbers.
# =============================================================================


@pytest.mark.parametrize("model_type", list(FlightModelType))
def test_trajectory_is_structurally_sound(
    client: TestClient, model_type: FlightModelType
) -> None:
    payload = _driver_payload(model_type.value)
    payload["max_time_s"] = 10.0
    payload["time_step_s"] = 0.01

    response = client.post("/tools/ball-flight/simulate", json=payload)

    assert response.status_code == 200
    data = response.json()
    trajectory = data["trajectory"]
    assert len(trajectory) >= 2

    times = [sample["time_s"] for sample in trajectory]
    assert times == sorted(times), "time must be monotonically non-decreasing"
    assert all(b > a for a, b in zip(times, times[1:], strict=False)), (
        "time must be strictly increasing"
    )

    for sample in trajectory:
        values = [sample["time_s"], *sample["position_m"], *sample["velocity_mps"]]
        assert all(math.isfinite(v) for v in values), "all samples must be finite"

    heights = [sample["position_m"][2] for sample in trajectory]
    apex = max(heights)
    assert apex > 0.0, "ball must rise above launch height"
    assert heights[-1] <= 0.5, "ball must return to (near) ground level"

    summary = data["summary"]
    assert summary["carry_m"] > 0.0
    assert summary["apex_m"] > 0.0
    assert summary["flight_time_s"] > 0.0
    assert all(math.isfinite(v) for v in summary.values())
