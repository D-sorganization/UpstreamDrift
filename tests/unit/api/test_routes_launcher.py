"""Unit tests for the launcher API route."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.launcher import router


@pytest.fixture
def app() -> FastAPI:
    """Create a FastAPI app with the launcher router."""
    test_app = FastAPI()
    test_app.include_router(router)
    return test_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create a test client."""
    return TestClient(app)


def test_get_manifest(client: TestClient) -> None:
    """Test getting the launcher manifest."""
    response = client.get("/launcher/manifest")
    assert response.status_code == 200
    data = response.json()
    assert "tiles" in data
    assert len(data["tiles"]) > 0


def test_get_tiles(client: TestClient) -> None:
    """Test getting all tiles."""
    response = client.get("/launcher/tiles")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0


def test_get_tile_found(client: TestClient) -> None:
    """Test getting a specific tile."""
    # Assuming "mujoco" or something similar exists
    # we first get all tiles to find a valid id
    tiles_resp = client.get("/launcher/tiles")
    valid_id = tiles_resp.json()[0]["id"]

    response = client.get(f"/launcher/tiles/{valid_id}")
    assert response.status_code == 200
    assert response.json()["id"] == valid_id


def test_get_tile_not_found(client: TestClient) -> None:
    """Test getting a non-existent tile."""
    response = client.get("/launcher/tiles/unknown_tile_id")
    assert response.status_code == 404


def test_get_engines_and_tools(client: TestClient) -> None:
    """Test getting filtered tiles."""
    engines_resp = client.get("/launcher/engines")
    assert engines_resp.status_code == 200
    assert isinstance(engines_resp.json(), list)

    tools_resp = client.get("/launcher/tools")
    assert tools_resp.status_code == 200
    assert isinstance(tools_resp.json(), list)


def test_validate_logos(client: TestClient) -> None:
    """Test validating logos."""
    response = client.get("/launcher/logos/validate")
    assert response.status_code == 200
    data = response.json()
    assert "all_valid" in data
    assert "missing_count" in data


def test_get_logo_invalid_path(client: TestClient) -> None:
    """Test path traversal protection for logos."""
    response = client.get("/launcher/logos/..%2Fsecret.txt")
    assert response.status_code in (400, 404)


def test_get_engine_capabilities(client: TestClient) -> None:
    """Test getting all engine capabilities."""
    response = client.get("/launcher/engines/capabilities")
    assert response.status_code == 200
    data = response.json()
    assert "mujoco" in data
    assert "drake" in data


def test_get_single_engine_capabilities(client: TestClient) -> None:
    """Test getting capabilities for a specific engine."""
    response = client.get("/launcher/engines/mujoco/capabilities")
    assert response.status_code == 200
    data = response.json()
    assert data["engine_name"] == "MuJoCo"


def test_get_single_engine_capabilities_not_found(client: TestClient) -> None:
    """Test getting capabilities for an unknown engine."""
    response = client.get("/launcher/engines/unknown/capabilities")
    assert response.status_code == 404
