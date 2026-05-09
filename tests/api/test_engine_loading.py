"""Tests for physics engine loading and availability.

This test suite ensures all physics engines can be probed and loaded correctly.
Following TDD approach - tests written first, then implementations.
"""

from collections.abc import Generator

import pytest

try:
    from fastapi.testclient import TestClient
    from src.api.server import app
except ImportError:
    pytest.skip("API server deps not available", allow_module_level=True)


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    """Test client with proper app lifespan."""
    with TestClient(app) as test_client:
        yield test_client


class TestEngineProbing:
    """Test engine availability probing."""

    @pytest.mark.parametrize(
        "engine_name",
        [
            "mujoco",
            "drake",
            "pinocchio",
            "opensim",
            "myosuite",
            "putting_green",
        ],
    )
    def test_engine_probe(self, client, engine_name: str) -> None:
        """Test that engine probe endpoint returns correct response structure.

        Engine availability depends on the environment (Docker vs local dev),
        so we validate the response shape rather than hardcoding expected values.
        """
        response = client.get(f"/api/engines/{engine_name}/probe")
        assert response.status_code == 200, f"Failed to probe {engine_name}"

        data = response.json()
        assert "available" in data, f"Missing 'available' key for {engine_name}"
        assert isinstance(data["available"], bool), f"{engine_name} available not bool"

        if data["available"]:
            assert "capabilities" in data
            assert isinstance(data["capabilities"], list)

    def test_unknown_engine_probe(self, client) -> None:
        """Test probing unknown engine returns proper error."""
        response = client.get("/api/engines/nonexistent/probe")
        assert response.status_code == 200  # Returns 200 with error in body

        data = response.json()
        assert data["available"] is False
        assert "error" in data
        assert "Unknown engine" in data["error"]


class TestEngineLoading:
    """Test engine loading functionality."""

    def test_load_unknown_engine(self, client) -> None:
        """Test loading unknown engine returns 400."""
        response = client.post("/api/engines/nonexistent/load")
        assert response.status_code == 400

        data = response.json()
        assert "detail" in data
        assert "Unknown engine" in data["detail"]


class TestEngineList:
    """Test engine listing endpoint."""

    def test_get_engines_list(self, client) -> None:
        """Test GET /api/engines returns all configured engines."""
        response = client.get("/engines")
        assert response.status_code == 200

        data = response.json()
        assert "engines" in data
        assert isinstance(data["engines"], list)
        assert len(data["engines"]) > 0

        # Check structure of first engine
        if data["engines"]:
            engine = data["engines"][0]
            assert "name" in engine
            assert "available" in engine
            assert "capabilities" in engine


class TestSimulationStart:
    """Test simulation starting with different engines."""

    @pytest.fixture
    def loaded_mujoco(self, client) -> None:
        """Fixture to ensure MuJoCo is loaded."""
        client.post("/api/engines/mujoco/load")


class TestPuttingGreenEngine:
    """Test Putting Green specific functionality (Issue #1136)."""

    def test_putting_green_probe(self, client) -> None:
        """Test Putting Green engine is available."""
        response = client.get("/api/engines/putting_green/probe")
        assert response.status_code == 200

        data = response.json()
        assert data["available"] is True

    def test_putting_green_load(self, client) -> None:
        """Test Putting Green engine can be loaded."""
        response = client.post("/api/engines/putting_green/load")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "loaded"
        assert data["engine"] == "putting_green"

    @pytest.mark.xfail(
        reason="Proper Putting Green implementation pending (Issue #1136)",
        strict=False,
    )
    def test_putting_green_simulation(self, client) -> None:
        """Test Putting Green simulation (will be implemented in #1136)."""
        # Load engine
        client.post("/api/engines/putting_green/load")

        # Start simulation
        response = client.post(
            "/api/simulation/start",
            json={
                "engine": "putting_green",
                "config": {
                    "green_dimensions": [10.0, 10.0],
                    "slope": 0.01,
                    "ball_position": [0.0, 0.0],
                },
            },
        )
        assert response.status_code == 200
