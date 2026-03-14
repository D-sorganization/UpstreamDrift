"""Tests for the Pressure Drop Calculator API router.

This router uses a self-contained Darcy-Weisbach inline implementation.
Tests validate physics correctness and regime classification directly
without mocking (no external shared-component dependencies).
"""

from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

from src.shared.python.calc_backend.routers.pressure_drop import router

client = TestClient(router)


@pytest.fixture
def turbulent_payload() -> dict:
    """Return a payload designed to generate turbulent Re > 4000."""
    return {
        "flow_rate_kg_s": 2.0,
        "pipe_diameter_m": 0.1,
        "pipe_length_m": 100.0,
        "pressure_pa": 200000.0,
        "temperature_k": 350.0,
        "molecular_weight_kg_mol": 0.029,
        "roughness_m": 0.000045,
    }


@pytest.fixture
def laminar_payload() -> dict:
    """Return a payload designed to generate laminar Re < 2300."""
    return {
        "flow_rate_kg_s": 0.0001,
        "pipe_diameter_m": 0.05,
        "pipe_length_m": 10.0,
        "pressure_pa": 101325.0,
        "temperature_k": 300.0,
        "molecular_weight_kg_mol": 0.029,
        "roughness_m": 0.000045,
    }


class TestPressureDropPhysics:
    """Validate physical correctness of the inline Darcy-Weisbach implementation."""

    def test_turbulent_regime_identified(self, turbulent_payload: dict) -> None:
        """High Re flow should be classified as Turbulent."""
        response = client.post("/", json=turbulent_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["flow_regime"] == "Turbulent"
        assert data["reynolds_number"] > 4000

    def test_laminar_regime_identified(self, laminar_payload: dict) -> None:
        """Very slow flow should be classified as Laminar."""
        response = client.post("/", json=laminar_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["flow_regime"] == "Laminar"
        assert data["reynolds_number"] < 2300

    def test_pressure_drop_positive(self, turbulent_payload: dict) -> None:
        """Pressure drop should always be non-negative for valid inputs."""
        response = client.post("/", json=turbulent_payload)
        assert response.status_code == 200
        assert response.json()["pressure_drop_pa"] >= 0

    def test_pressure_drop_increases_with_length(self) -> None:
        """Doubling pipe length should approximately double pressure drop."""
        base = {
            "flow_rate_kg_s": 1.0,
            "pipe_diameter_m": 0.05,
            "pipe_length_m": 50.0,
            "pressure_pa": 200000.0,
            "temperature_k": 300.0,
            "molecular_weight_kg_mol": 0.029,
            "roughness_m": 0.000045,
        }
        double = {**base, "pipe_length_m": 100.0}
        r1 = client.post("/", json=base).json()
        r2 = client.post("/", json=double).json()
        assert r2["pressure_drop_pa"] == pytest.approx(
            r1["pressure_drop_pa"] * 2, rel=1e-3
        )

    def test_all_outputs_finite(self, turbulent_payload: dict) -> None:
        """All output fields must be finite, not NaN or inf."""
        response = client.post("/", json=turbulent_payload)
        assert response.status_code == 200
        data = response.json()
        for field in [
            "pressure_drop_pa",
            "reynolds_number",
            "friction_factor",
            "velocity_m_s",
            "density_kg_m3",
            "viscosity_pa_s",
        ]:
            assert math.isfinite(data[field]), f"{field} is not finite"

    def test_density_uses_ideal_gas_law(self, turbulent_payload: dict) -> None:
        """Verify density output matches ideal gas: rho = P*M / (R*T)."""
        response = client.post("/", json=turbulent_payload)
        expected = (200000.0 * 0.029) / (8.314462618 * 350.0)
        assert response.status_code == 200
        assert response.json()["density_kg_m3"] == pytest.approx(expected, rel=1e-4)

    def test_laminar_friction_factor(self, laminar_payload: dict) -> None:
        """In laminar regime, friction factor = 64 / Re."""
        response = client.post("/", json=laminar_payload)
        assert response.status_code == 200
        data = response.json()
        re = data["reynolds_number"]
        if re > 0:
            expected_ff = 64.0 / re
            assert data["friction_factor"] == pytest.approx(expected_ff, rel=1e-3)
