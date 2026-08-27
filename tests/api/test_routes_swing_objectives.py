"""Tests for the Swing Objective Lab REST route (issue #9128)."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.swing_objectives import (
    COMPARISON_SCHEMA_VERSION,
    SwingObjectiveCompareRequest,
    router,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/api")
    return TestClient(app)


def test_list_presets(client: TestClient) -> None:
    response = client.get("/api/tools/swing-objectives/presets")
    assert response.status_code == 200
    data = response.json()
    assert "presets" in data
    preset_names = [p["name"] for p in data["presets"]]
    assert len(preset_names) >= 1
    assert any("Tour" in name or "tour" in name.lower() for name in preset_names)


def test_compare_swing_objectives_default(client: TestClient) -> None:
    payload = {
        "duration_s": 0.28,
        "hub_torque_nm": 250.0,
        "wrist_torque_nm": 20.0,
        "node_count": 21,
    }
    response = client.post("/api/tools/swing-objectives/compare", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["schema_version"] == COMPARISON_SCHEMA_VERSION
    assert len(data["objective_keys"]) == 5
    assert "clubhead_speed" in data["objective_keys"]
    assert "matrix" in data
    assert len(data["matrix"]) == 5
    assert len(data["matrix"][0]) == 5
    assert "raw_values" in data
    assert "torque_saturation" in data
    assert "is_degenerate" in data
    assert isinstance(data["is_degenerate"], bool)


def test_compare_swing_objectives_subset_keys(client: TestClient) -> None:
    payload = {
        "duration_s": 0.28,
        "hub_torque_nm": 250.0,
        "wrist_torque_nm": 20.0,
        "node_count": 15,
        "objective_keys": ["clubhead_speed", "centrifugal"],
    }
    response = client.post("/api/tools/swing-objectives/compare", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["schema_version"] == COMPARISON_SCHEMA_VERSION
    assert data["objective_keys"] == ["clubhead_speed", "centrifugal"]
    assert len(data["matrix"]) == 2
    assert len(data["matrix"][0]) == 2


def test_compare_invalid_duration(client: TestClient) -> None:
    payload = {
        "duration_s": -0.1,
        "hub_torque_nm": 250.0,
        "wrist_torque_nm": 20.0,
        "node_count": 21,
    }
    response = client.post("/api/tools/swing-objectives/compare", json=payload)
    assert response.status_code in (400, 422)


def test_compare_invalid_node_count(client: TestClient) -> None:
    payload = {
        "duration_s": 0.28,
        "hub_torque_nm": 250.0,
        "wrist_torque_nm": 20.0,
        "node_count": 2,
    }
    response = client.post("/api/tools/swing-objectives/compare", json=payload)
    assert response.status_code in (400, 422)
