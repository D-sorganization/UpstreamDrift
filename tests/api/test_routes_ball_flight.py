"""Tests for the ball-flight REST route."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.api.routes.ball_flight import BallFlightSimulationRequest, router
from src.shared.python.physics.flight_models import FlightModelType


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
