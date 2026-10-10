"""Regression tests for REST simulation recorder (Repository_Management #1836 / R01).

Acceptance criteria:
- A short run with recording returns non-empty times and successful simulation response.
- An empty run (where recording is on but the recorder produced no time series/frames
  along the _extract_simulation_data path) returns an error response.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

os.environ.setdefault("DATABASE_URL", "sqlite://")
os.environ.setdefault("GOLF_AUTH_DISABLED", "true")

from fastapi.testclient import TestClient

from src.api.server import app

pytestmark = [pytest.mark.unit]


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Create test client with GOLF_AUTH_DISABLED."""
    with TestClient(app) as test_client:
        yield test_client


def test_short_run_with_recording_returns_nonempty_times(client: TestClient) -> None:
    """A short simulation run with recording produces non-empty times and success=True."""
    payload = {
        "engine_type": "pendulum",
        "duration": 0.01,
        "timestep": 0.005,
    }
    response = client.post("/simulate", json=payload)
    assert response.status_code == 200, (
        f"Expected 200, got {response.status_code}: {response.text}"
    )
    data = response.json()
    assert data["success"] is True
    assert "data" in data
    assert "times" in data["data"]
    times = data["data"]["times"]
    assert len(times) > 0, "Expected non-empty times series from short run"
    assert times[0] == pytest.approx(0.0)
    assert times[-1] == pytest.approx(0.01)
    # Recorded channels must be aligned with the time axis.
    for channel in ("joint_positions", "joint_velocities", "joint_accelerations"):
        assert len(data["data"][channel]) == len(times), channel


def test_empty_run_where_recorder_produces_no_time_series_returns_error(
    client: TestClient,
) -> None:
    """When recording is on but the recorder produced no time series/frames
    (_extract_simulation_data path), the REST simulation response must fail (HTTP 400).
    """
    payload = {
        "engine_type": "pendulum",
        "duration": 0.01,
        "timestep": 0.005,
    }
    # Simulate the R01 condition: recorder produced no time series/frames
    # (_extract_simulation_data path returns empty series)
    with patch(
        "src.api.services.simulation_service.SimulationService._extract_simulation_data",
        return_value={
            "times": [],
            "joint_positions": [],
            "joint_velocities": [],
            "joint_accelerations": [],
        },
    ):
        response = client.post("/simulate", json=payload)
        # Must return an error response (HTTP 400 Bad Request)
        assert response.status_code == 400, (
            f"Expected 400 error status, got {response.status_code}: {response.text}"
        )
        data = response.json()
        assert "detail" in data
        assert "required channel validation" in data["detail"]
