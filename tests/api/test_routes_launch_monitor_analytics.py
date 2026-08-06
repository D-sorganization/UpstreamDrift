"""Tests for the UI-neutral launch-monitor analytics API contract."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.launch_monitor_analytics import router

pytestmark = pytest.mark.integration


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _records() -> list[dict[str, object]]:
    return [
        {
            "shot_id": f"shot-{index}",
            "session_id": "api-session",
            "monitor_vendor": "FlightScope",
            "club_speed": float(index),
            "ball_speed": 1.5 * index + 2.0,
        }
        for index in range(1, 31)
    ]


def test_capabilities_are_machine_readable_and_versioned(client: TestClient) -> None:
    response = client.get("/tools/launch-monitor-analytics/capabilities")
    assert response.status_code == 200
    payload = response.json()
    assert payload["contract_version"] == "1.0.0"
    assert payload["analysis_modes"] == ["correlation", "regression", "comprehensive"]
    assert payload["aggregate_regression_allowed"] is False


def test_analyze_accepts_inline_records_and_returns_provenance(
    client: TestClient,
) -> None:
    response = client.post(
        "/tools/launch-monitor-analytics/analyze",
        json={
            "records": _records(),
            "analysis": {
                "outcome": "ball_speed",
                "predictors": ["club_speed"],
                "analysis_mode": "comprehensive",
                "min_samples": 10,
            },
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["contract_version"] == "1.0.0"
    assert payload["result"]["dataset"]["monitor_vendors"] == ["FlightScope"]
    assert len(payload["result"]["dataset"]["fingerprint_sha256"]) == 64
    assert payload["result"]["regression"]["r_squared"] == pytest.approx(1.0)


def test_analyze_rejects_aggregate_regression(client: TestClient) -> None:
    records = _records()
    for record in records:
        record["observation_kind"] = "aggregate"
    response = client.post(
        "/tools/launch-monitor-analytics/analyze",
        json={
            "records": records,
            "analysis": {
                "outcome": "ball_speed",
                "predictors": ["club_speed"],
                "analysis_mode": "regression",
                "allow_aggregate": True,
            },
        },
    )
    assert response.status_code == 400
    assert "Aggregate observations cannot enter regression" in response.json()["detail"]
