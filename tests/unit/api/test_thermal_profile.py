"""Tests for the Thermal Profile Predictor API router.

This router uses a self-contained RK4 ODE solver inline implementation.
Tests validate RK4 integration physics correctness and profile modes directly
without mocking (no external shared-component dependencies).
"""

from __future__ import annotations

import math

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.shared.python.calc_backend.routers.thermal_profile import router

_app = FastAPI()
_app.include_router(router)
client = TestClient(_app)


@pytest.fixture
def constant_power_payload() -> dict:
    """Return a constant-power heating scenario payload."""
    return {
        "power_w": 1000.0,
        "power_profile": "constant",
        "thermal_mass_j_per_k": 5000.0,
        "heat_loss_coeff_w_per_k": 10.0,
        "initial_temp_c": 20.0,
        "ambient_temp_c": 20.0,
        "t_start_s": 0.0,
        "t_end_s": 3600.0,
        "num_points": 100,
        "ramp_rate_w_per_s": 0.0,
        "step_time_s": 0.0,
    }


class TestThermalProfilePhysics:
    """Validate RK4 integration and thermal physics."""

    def test_constant_power_temperature_rises(
        self, constant_power_payload: dict
    ) -> None:
        """With heating power > heat loss, temperature must rise."""
        response = client.post("/api/calc/thermal-profile", json=constant_power_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["final_temp_c"] > constant_power_payload["initial_temp_c"]
        assert data["temp_change_c"] > 0

    def test_steady_state_matches_analytical(
        self, constant_power_payload: dict
    ) -> None:
        """Steady state T = P/h + T_amb; reported value should match."""
        response = client.post("/api/calc/thermal-profile", json=constant_power_payload)
        assert response.status_code == 200
        data = response.json()
        # Analytical: T_ss = 1000/10 + 20 = 120°C
        assert data["steady_state_temp_c"] == pytest.approx(120.0, abs=0.1)

    def test_time_constant_matches_analytical(
        self, constant_power_payload: dict
    ) -> None:
        """Time constant τ = C_th / h; reported value should match."""
        response = client.post("/api/calc/thermal-profile", json=constant_power_payload)
        assert response.status_code == 200
        data = response.json()
        # τ = 5000 / 10 = 500s
        assert data["time_constant_s"] == pytest.approx(500.0, abs=0.1)

    def test_correct_number_of_data_points(self, constant_power_payload: dict) -> None:
        """Response should contain exactly `num_points` data entries."""
        response = client.post("/api/calc/thermal-profile", json=constant_power_payload)
        assert response.status_code == 200
        assert len(response.json()["data"]) == 100

    def test_no_power_no_temperature_change(self, constant_power_payload: dict) -> None:
        """With zero power and no temperature delta, temperature should be constant."""
        payload = {**constant_power_payload, "power_w": 0.0}
        response = client.post("/api/calc/thermal-profile", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["temp_change_c"] == pytest.approx(0.0, abs=0.01)

    def test_linear_ramp_profile(self, constant_power_payload: dict) -> None:
        """Linear ramp should produce higher final temperature than constant for same start."""
        ramp_payload = {
            **constant_power_payload,
            "power_profile": "linear_ramp",
            "ramp_rate_w_per_s": 0.5,
        }
        r_const = client.post(
            "/api/calc/thermal-profile", json=constant_power_payload
        ).json()
        r_ramp = client.post("/api/calc/thermal-profile", json=ramp_payload).json()
        assert r_ramp["final_temp_c"] > r_const["final_temp_c"]

    def test_step_profile_drops_to_ambient_over_time(self) -> None:
        """After step-off, system should cool (final temp < initial after step at t=0)."""
        payload = {
            "power_w": 0.0,
            "power_profile": "step",
            "thermal_mass_j_per_k": 1000.0,
            "heat_loss_coeff_w_per_k": 5.0,
            "initial_temp_c": 80.0,
            "ambient_temp_c": 20.0,
            "t_start_s": 0.0,
            "t_end_s": 600.0,
            "num_points": 50,
            "ramp_rate_w_per_s": 0.0,
            "step_time_s": 0.0,  # step off immediately
        }
        response = client.post("/api/calc/thermal-profile", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["final_temp_c"] < payload["initial_temp_c"]

    def test_all_outputs_finite(self, constant_power_payload: dict) -> None:
        """All numeric outputs must be finite."""
        response = client.post("/api/calc/thermal-profile", json=constant_power_payload)
        assert response.status_code == 200
        data = response.json()
        for field in ["final_temp_c", "max_temp_c", "min_temp_c", "temp_change_c"]:
            assert math.isfinite(data[field]), f"{field} is not finite"
