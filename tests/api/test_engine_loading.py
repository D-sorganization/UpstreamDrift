"""Tests for physics engine loading and availability.

This test suite ensures all physics engines can be probed and loaded correctly.
Following TDD approach - tests written first, then implementations.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.server import app

client = TestClient(app)


class TestEngineProbing:
    """Test engine availability probing."""

    @pytest.mark.parametrize(
        "engine_name,expected_available",
        [
            ("mujoco", True),  # Should be installed in Docker
            ("drake", True),  # Should be installed in Docker
            ("pinocchio", True),  # Should be installed in Docker
            ("opensim", False),  # Not yet installed (Issue #1140)
            ("myosuite", False),  # Not yet installed (Issue #1141)
            ("putting_green", True),  # Mapped to PENDULUM temporarily
        ],
    )
    def test_engine_probe(self, engine_name: str, expected_available: bool) -> None:
        """Test that engine probe endpoint returns correct availability."""
        response = client.get(f"/api/engines/{engine_name}/probe")
        assert response.status_code == 200, f"Failed to probe {engine_name}"

        data = response.json()
        assert "available" in data, f"Missing 'available' key for {engine_name}"
        assert (
            data["available"] == expected_available
        ), f"{engine_name} availability mismatch"

        if expected_available:
            assert "version" in data or data["version"] is not None
            assert "capabilities" in data
            assert isinstance(data["capabilities"], list)

    def test_unknown_engine_probe(self) -> None:
        """Test probing unknown engine returns proper error."""
        response = client.get("/api/engines/nonexistent/probe")
        assert response.status_code == 200  # Returns 200 with error in body

        data = response.json()
        assert data["available"] is False
        assert "error" in data
        assert "Unknown engine" in data["error"]


class TestEngineLoading:
    """Test engine loading functionality."""

    @pytest.mark.parametrize(
        "engine_name",
        [
            "mujoco",
            "drake",
            "pinocchio",
        ],
    )
    def test_load_available_engine(self, engine_name: str) -> None:
        """Test loading an available engine succeeds."""
        response = client.post(f"/api/engines/{engine_name}/load")
        assert response.status_code == 200, f"Failed to load {engine_name}"

        data = response.json()
        assert data["status"] == "loaded"
        assert data["engine"] == engine_name
        assert "version" in data
        assert "capabilities" in data

    @pytest.mark.parametrize(
        "engine_name",
        [
            "opensim",  # Not installed yet
            "myosuite",  # Not installed yet
        ],
    )
    def test_load_unavailable_engine(self, engine_name: str) -> None:
        """Test loading unavailable engine fails gracefully."""
        response = client.post(f"/api/engines/{engine_name}/load")
        # Should return error status
        assert response.status_code in [400, 500]

    def test_load_unknown_engine(self) -> None:
        """Test loading unknown engine returns 400."""
        response = client.post("/api/engines/nonexistent/load")
        assert response.status_code == 400

        data = response.json()
        assert "detail" in data
        assert "Unknown engine" in data["detail"]


class TestEngineList:
    """Test engine listing endpoint."""

    def test_get_engines_list(self) -> None:
        """Test GET /api/engines returns all configured engines."""
        response = client.get("/api/engines")
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
    def loaded_mujoco(self) -> None:
        """Fixture to ensure MuJoCo is loaded."""
        client.post("/api/engines/mujoco/load")

    def test_start_simulation_with_mujoco(self, loaded_mujoco: None) -> None:
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

    def test_putting_green_probe(self) -> None:
        """Test Putting Green engine is available."""
        response = client.get("/api/engines/putting_green/probe")
        assert response.status_code == 200

        data = response.json()
        assert data["available"] is True

    def test_putting_green_load(self) -> None:
        """Test Putting Green engine can be loaded."""
        response = client.post("/api/engines/putting_green/load")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "loaded"
        assert data["engine"] == "putting_green"

    @pytest.mark.skip(reason="Proper Putting Green implementation pending (Issue #1136)")
    def test_putting_green_simulation(self) -> None:
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
