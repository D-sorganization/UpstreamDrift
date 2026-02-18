"""Tests for physics engine loading and availability.

This test suite ensures all physics engines can be probed and loaded correctly.
Following TDD approach - tests written first, then implementations.
"""

import pytest

try:
    from fastapi.testclient import TestClient

    from src.api.server import app
except ImportError:
    pytest.skip("API server deps not available", allow_module_level=True)


@pytest.fixture(scope="module")
def client():
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
        response = client.get(f"/api/v1/engines/{engine_name}/probe")
        assert response.status_code == 200, f"Failed to probe {engine_name}"

        data = response.json()
        assert "available" in data, f"Missing 'available' key for {engine_name}"
        assert isinstance(data["available"], bool), f"{engine_name} available not bool"

        if data["available"]:
            assert "capabilities" in data
            assert isinstance(data["capabilities"], list)

    def test_unknown_engine_probe(self, client) -> None:
        """Test probing unknown engine returns proper error."""
        response = client.get("/api/v1/engines/nonexistent/probe")
        assert response.status_code == 200  # Returns 200 with error in body

        data = response.json()
        assert data["available"] is False
        assert "error" in data
        assert "Unknown engine" in data["error"]


class TestEngineLoading:
    """Test engine loading functionality."""

    @pytest.mark.skip(reason="Engine Python modules not installed in test environment")
    @pytest.mark.parametrize(
        "engine_name",
        [
            "mujoco",
            "drake",
            "pinocchio",
        ],
    )
    def test_load_available_engine(self, client, engine_name: str) -> None:
        """Test loading an available engine succeeds."""
        response = client.post(f"/api/v1/engines/{engine_name}/load")
        assert response.status_code == 200, f"Failed to load {engine_name}"

        data = response.json()
        assert data["status"] == "loaded"
        assert data["engine"] == engine_name
        assert "version" in data
        assert "capabilities" in data

    @pytest.mark.parametrize(
        "engine_name",
        [
            "myosuite",  # Not installed yet
        ],
    )
    def test_load_unavailable_engine(self, client, engine_name: str) -> None:
        """Test loading unavailable engine fails gracefully."""
        response = client.post(f"/api/v1/engines/{engine_name}/load")
        # Should return error status
        assert response.status_code in [400, 500]

    def test_load_unknown_engine(self, client) -> None:
        """Test loading unknown engine returns 400."""
        response = client.post("/api/v1/engines/nonexistent/load")
        assert response.status_code == 400

        data = response.json()
        assert "detail" in data
        assert "Unknown engine" in data["detail"]


class TestEngineList:
    """Test engine listing endpoint."""

    def test_get_engines_list(self, client) -> None:
        """Test GET /api/engines returns all configured engines."""
        response = client.get("/api/v1/engines")
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
        client.post("/api/v1/engines/mujoco/load")

    @pytest.mark.skip(reason="MuJoCo Python module not installed in test environment")
    def test_start_simulation_with_mujoco(self, client, loaded_mujoco: None) -> None:
        """Test starting a simulation with MuJoCo engine."""
        response = client.post(
            "/api/simulation/start",
            json={
                "engine": "mujoco",
                "config": {
                    "timestep": 0.001,
                    "duration": 1.0,
                },
            },
        )
        # This test will initially fail as simulation service needs implementation
        # This is intentional (TDD) - implement after seeing test fail
        assert response.status_code in [200, 201]

        data = response.json()
        assert "simulation_id" in data or "status" in data


class TestPuttingGreenEngine:
    """Test Putting Green specific functionality (Issue #1136)."""

    def test_putting_green_probe(self, client) -> None:
        """Test Putting Green engine is available."""
        response = client.get("/api/v1/engines/putting_green/probe")
        assert response.status_code == 200

        data = response.json()
        assert data["available"] is True

    def test_putting_green_load(self, client) -> None:
        """Test Putting Green engine can be loaded."""
        response = client.post("/api/v1/engines/putting_green/load")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "loaded"
        assert data["engine"] == "putting_green"

    @pytest.mark.skip(
        reason="Proper Putting Green implementation pending (Issue #1136)"
    )
    def test_putting_green_simulation(self, client) -> None:
        """Test Putting Green simulation (will be implemented in #1136)."""
        # Load engine
        client.post("/api/v1/engines/putting_green/load")

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
