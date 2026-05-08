"""Unit tests for the terrain API route."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.terrain import router


@pytest.fixture
def app() -> FastAPI:
    """Create a FastAPI app with the terrain router."""
    test_app = FastAPI()
    test_app.include_router(router)
    return test_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create a test client."""
    return TestClient(app)


def test_list_presets(client: TestClient) -> None:
    """Test listing available environment presets."""
    response = client.get("/terrain/presets")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    # Check that putting_green is present
    names = [preset["name"] for preset in data]
    assert "putting_green" in names
    assert "full_hole" in names


def test_load_environment_success(client: TestClient) -> None:
    """Test loading an environment preset successfully."""
    payload = {
        "preset": "putting_green",
        "width": 15.0,
        "length": 20.0,
        "slope_angle_deg": 1.0,
        "slope_direction_deg": 0.0,
    }
    response = client.post("/terrain/load", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["name"] == "putting_green"
    assert data["width_m"] == 15.0
    assert data["length_m"] == 20.0


def test_load_environment_unknown_preset(client: TestClient) -> None:
    """Test loading an unknown environment preset."""
    payload = {
        "preset": "unknown_preset",
        "slope_angle_deg": 0.0,
        "slope_direction_deg": 0.0,
    }
    response = client.post("/terrain/load", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert "Unknown preset" in data["error"]


def test_query_terrain_valid(client: TestClient) -> None:
    """Test querying terrain at a specific valid point."""
    # First load a specific terrain
    client.post(
        "/terrain/load",
        json={"preset": "driving_range", "width": 100.0, "length": 200.0},
    )

    payload = {"x": 50.0, "y": 100.0}
    response = client.post("/terrain/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "elevation" in data
    assert "friction" in data
    assert "terrain_type" in data


def test_list_materials(client: TestClient) -> None:
    """Test listing surface materials."""
    response = client.get("/terrain/materials")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    names = [mat["name"] for mat in data]
    assert "green" in names
    assert "fairway" in names


def test_list_terrain_types(client: TestClient) -> None:
    """Test listing terrain types."""
    response = client.get("/terrain/types")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert "green" in data
    assert "fairway" in data
    assert "water" in data


def test_get_active_terrain(client: TestClient) -> None:
    """Test getting active terrain info."""
    client.post(
        "/terrain/load", json={"preset": "bunker", "width": 30.0, "length": 40.0}
    )

    response = client.get("/terrain/active")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "bunker"
    assert data["width_m"] == 30.0
    assert data["length_m"] == 40.0
    assert "patch_count" in data
    assert "region_count" in data
