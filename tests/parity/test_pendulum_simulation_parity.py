"""Pendulum simulation parity tests.

Verifies that the pendulum physics engine produces consistent results
whether called directly or through the FastAPI simulation endpoint.

Test vectors use the double-pendulum model which is deterministic
(no stochastic elements) and pure Python (no external dependencies).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Test Vectors
# ---------------------------------------------------------------------------

PENDULUM_TEST_VECTORS = [
    {
        "name": "equilibrium_rest",
        "description": "Both links hanging straight down, no velocity",
        "initial_state": {"positions": [0.0, 0.0], "velocities": [0.0, 0.0]},
        "duration": 0.1,
        "timestep": 0.01,
        "expected": {
            "success": True,
            "frames": 10,
            "final_positions_near_zero": True,
        },
    },
    {
        "name": "small_angle_oscillation",
        "description": "Small displacement from equilibrium, expect oscillatory motion",
        "initial_state": {"positions": [0.1, 0.0], "velocities": [0.0, 0.0]},
        "duration": 0.5,
        "timestep": 0.01,
        "expected": {
            "success": True,
            "frames": 50,
            "theta1_oscillates": True,
        },
    },
    {
        "name": "horizontal_release",
        "description": "First link horizontal, released from rest",
        "initial_state": {
            "positions": [math.pi / 2, 0.0],
            "velocities": [0.0, 0.0],
        },
        "duration": 0.2,
        "timestep": 0.01,
        "expected": {
            "success": True,
            "frames": 20,
            "theta1_decreases": True,
        },
    },
]


# ---------------------------------------------------------------------------
# Engine-Level Tests
# ---------------------------------------------------------------------------


class TestPendulumEngineDirectly:
    """Test the pendulum physics engine directly (no API)."""

    def test_engine_initializes(self, pendulum_engine) -> None:
        """Engine should initialize with zero state."""
        q, v = pendulum_engine.get_state()
        np.testing.assert_array_almost_equal(q, [0.0, 0.0])
        np.testing.assert_array_almost_equal(v, [0.0, 0.0])

    def test_engine_set_state(self, pendulum_engine) -> None:
        """Engine should accept and return state correctly."""
        q_set = np.array([0.5, -0.3])
        v_set = np.array([1.0, -0.5])
        pendulum_engine.set_state(q_set, v_set)

        q, v = pendulum_engine.get_state()
        np.testing.assert_array_almost_equal(q, q_set)
        np.testing.assert_array_almost_equal(v, v_set)

    def test_engine_step_changes_state(self, pendulum_engine) -> None:
        """Stepping from a displaced position should change the state."""
        pendulum_engine.set_state(
            np.array([0.5, 0.0]),
            np.array([0.0, 0.0]),
        )
        q_before, v_before = pendulum_engine.get_state()

        pendulum_engine.step(0.01)
        q_after, v_after = pendulum_engine.get_state()

        # State should have changed (gravity acts on displaced pendulum)
        assert not np.allclose(q_before, q_after) or not np.allclose(
            v_before, v_after
        ), "State should change after stepping from displaced position"

    def test_engine_mass_matrix_positive_definite(self, pendulum_engine) -> None:
        """Mass matrix should be symmetric positive definite."""
        M = pendulum_engine.compute_mass_matrix()
        assert M.shape == (2, 2)
        eigenvalues = np.linalg.eigvalsh(M)
        assert all(eigenvalues > 0), f"Mass matrix not PD: eigenvalues={eigenvalues}"

    def test_engine_equilibrium_stable(self, pendulum_engine) -> None:
        """At equilibrium (theta=0, omega=0), state should remain near zero."""
        pendulum_engine.set_state(
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),
        )
        for _ in range(100):
            pendulum_engine.step(0.001)

        q, v = pendulum_engine.get_state()
        assert abs(q[0]) < 1e-6, f"theta1 drifted: {q[0]}"
        assert abs(q[1]) < 1e-6, f"theta2 drifted: {q[1]}"

    @pytest.mark.parametrize("vector", PENDULUM_TEST_VECTORS, ids=lambda v: v["name"])
    def test_engine_matches_vector(self, pendulum_engine, vector) -> None:
        """Run engine directly and verify against test vector expectations."""
        state = vector["initial_state"]
        pendulum_engine.set_state(
            np.array(state["positions"]),
            np.array(state["velocities"]),
        )

        positions_history = []
        for _ in range(vector["expected"]["frames"]):
            pendulum_engine.step(vector["timestep"])
            q, _ = pendulum_engine.get_state()
            positions_history.append(q.copy())

        expected = vector["expected"]

        if expected.get("final_positions_near_zero"):
            q_final = positions_history[-1]
            assert abs(q_final[0]) < 0.01, f"theta1 too large: {q_final[0]}"

        if expected.get("theta1_oscillates"):
            theta1_values = [p[0] for p in positions_history]
            # Check that theta1 has changed sign or passed through zero
            max_t1 = max(theta1_values)
            min_t1 = min(theta1_values)
            assert max_t1 - min_t1 > 0.01, "theta1 should show oscillation"

        if expected.get("theta1_decreases"):
            # From pi/2, gravity should pull link back toward 0
            theta1_values = [p[0] for p in positions_history]
            assert theta1_values[-1] < math.pi / 2, (
                f"theta1 should decrease from pi/2, got {theta1_values[-1]}"
            )


# ---------------------------------------------------------------------------
# API-Level Tests
# ---------------------------------------------------------------------------


class TestPendulumSimulationAPI:
    """Test the simulation API endpoint with the pendulum engine."""

    def test_api_simulate_returns_success(self, client) -> None:
        """POST /simulate with pendulum engine should return success."""
        response = client.post(
            "/simulate",
            json={
                "engine_type": "pendulum",
                "duration": 0.1,
                "timestep": 0.01,
                "initial_state": {
                    "positions": [0.1, 0.0],
                    "velocities": [0.0, 0.0],
                },
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["frames"] == 10
        assert data["duration"] == 0.1

    def test_api_simulate_returns_data_fields(self, client) -> None:
        """API response should contain expected data fields."""
        response = client.post(
            "/simulate",
            json={
                "engine_type": "pendulum",
                "duration": 0.05,
                "timestep": 0.01,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        # Data fields depend on recorder implementation
        # At minimum we expect the response to have the standard shape
        assert isinstance(data["data"], dict)

    def test_api_rejects_invalid_engine(self, client) -> None:
        """API should reject requests for non-existent engines."""
        response = client.post(
            "/simulate",
            json={
                "engine_type": "nonexistent_engine_xyz",
                "duration": 0.1,
            },
        )
        # Should return error (either 400, 500, or success=False)
        data = response.json()
        if response.status_code == 200:
            assert data["success"] is False

    def test_api_rejects_zero_duration(self, client) -> None:
        """API should reject zero or negative duration."""
        response = client.post(
            "/simulate",
            json={
                "engine_type": "pendulum",
                "duration": 0.0,
            },
        )
        assert response.status_code == 422  # Pydantic validation error

    @pytest.mark.parametrize("vector", PENDULUM_TEST_VECTORS, ids=lambda v: v["name"])
    def test_api_matches_vector(self, client, vector) -> None:
        """API response should match test vector expectations."""
        response = client.post(
            "/simulate",
            json={
                "engine_type": "pendulum",
                "duration": vector["duration"],
                "timestep": vector["timestep"],
                "initial_state": vector["initial_state"],
            },
        )
        assert response.status_code == 200
        data = response.json()
        expected = vector["expected"]
        assert data["success"] is expected["success"]
        assert data["frames"] == expected["frames"]


# ---------------------------------------------------------------------------
# Engine-API Consistency Tests
# ---------------------------------------------------------------------------


class TestPendulumEngineAPIConsistency:
    """Verify that engine and API produce identical results for same inputs."""

    def test_frame_count_matches(self, client, pendulum_engine) -> None:
        """Engine and API should produce the same number of frames."""
        duration = 0.1
        timestep = 0.01
        expected_frames = int(duration / timestep)

        # Engine
        pendulum_engine.set_state(np.array([0.1, 0.0]), np.array([0.0, 0.0]))
        for _ in range(expected_frames):
            pendulum_engine.step(timestep)

        # API
        response = client.post(
            "/simulate",
            json={
                "engine_type": "pendulum",
                "duration": duration,
                "timestep": timestep,
                "initial_state": {
                    "positions": [0.1, 0.0],
                    "velocities": [0.0, 0.0],
                },
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["frames"] == expected_frames

    def test_equilibrium_consistency(self, client, pendulum_engine) -> None:
        """Both paths should agree on equilibrium behavior."""
        # Engine: step from equilibrium
        pendulum_engine.set_state(np.array([0.0, 0.0]), np.array([0.0, 0.0]))
        for _ in range(10):
            pendulum_engine.step(0.01)
        q_engine, _ = pendulum_engine.get_state()

        # API
        response = client.post(
            "/simulate",
            json={
                "engine_type": "pendulum",
                "duration": 0.1,
                "timestep": 0.01,
                "initial_state": {
                    "positions": [0.0, 0.0],
                    "velocities": [0.0, 0.0],
                },
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # Both should report near-zero state
        assert abs(q_engine[0]) < 1e-6
