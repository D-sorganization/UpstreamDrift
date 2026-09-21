"""TDD tests for Engine Matrix API routes (MS-71, #10351)."""

from __future__ import annotations

import pytest

try:
    from fastapi.testclient import TestClient
    from src.api.server import app
except ImportError:
    pytest.skip("API server deps not available", allow_module_level=True)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Create test client with application lifespan."""
    with TestClient(app) as test_client:
        yield test_client


def test_get_engine_matrix_endpoint(client: TestClient) -> None:
    """GET /api/engines/matrix returns the full engine capability matrix."""
    response = client.get("/api/engines/matrix")
    assert response.status_code == 200
    data = response.json()
    assert "schema_version" in data
    assert "profiles" in data
    assert "advertised_engines" in data
    assert "mujoco" in data["profiles"]
    assert "drake" in data["profiles"]
    assert "opensim" in data["profiles"]
    assert "myosuite" in data["profiles"]
    assert "jaxsim" in data["profiles"]


def test_get_launcher_engine_capabilities(client: TestClient) -> None:
    """GET /api/launcher/engines/capabilities returns profiles derived from matrix."""
    response = client.get("/api/launcher/engines/capabilities")
    assert response.status_code == 200
    data = response.json()
    assert "mujoco" in data
    assert "drake" in data
    assert "opensim" in data
    assert "myosuite" in data
    assert "jaxsim" in data


def test_get_launcher_single_engine_capability(client: TestClient) -> None:
    """GET /api/launcher/engines/{engine_id}/capabilities returns profile for single engine."""
    response = client.get("/api/launcher/engines/mujoco/capabilities")
    assert response.status_code == 200
    data = response.json()
    assert "engine_name" in data
    assert data["engine_name"].lower() == "mujoco"

    # Unknown engine returns 404
    bad_res = client.get("/api/launcher/engines/nonexistent_engine/capabilities")
    assert bad_res.status_code == 404
