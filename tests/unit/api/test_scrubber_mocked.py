"""Tests for the Scrubber Calculator API router.

This test file adheres to the Fleet-Wide Shared Component Testing Strategy.
It mocks the physics calculation methods from `upstream_drift_tools` (Tools repo)
to verify that the API layer correctly implements the contract without testing the
internal mathematical logic directly.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.shared.python.calc_backend.routers.scrubber import router

client = TestClient(router)


@pytest.fixture
def mock_tools():
    """Mock the scrubber calculators securely from Tools."""
    with (
        patch(
            "src.shared.python.calc_backend.routers.scrubber.calculate_gas_density"
        ) as mdensity,
        patch(
            "src.shared.python.calc_backend.routers.scrubber.calculate_gas_viscosity"
        ) as mviscosity,
        patch(
            "src.shared.python.calc_backend.routers.scrubber.calculate_flooding_velocity"
        ) as mflood,
        patch(
            "src.shared.python.calc_backend.routers.scrubber.calculate_column_diameter"
        ) as mdiameter,
        patch(
            "src.shared.python.calc_backend.routers.scrubber.calculate_caustic_requirement"
        ) as mcaustic,
    ):
        yield mdensity, mviscosity, mflood, mdiameter, mcaustic


def test_calculate_scrubber_success(mock_tools) -> None:
    """Validate that the API passes through inputs and parses math results correctly."""
    mdensity, mviscosity, mflood, mdiameter, mcaustic = mock_tools

    # Setup mocks
    mdensity.return_value = 1.2
    mviscosity.return_value = 0.000018
    mflood.return_value = 2.5
    mdiameter.return_value = {
        "design_velocity_m_s": 1.75,
        "diameter_m": 1.2,
        "diameter_ft": 3.93,
        "cross_section_m2": 1.13,
    }
    mcaustic.return_value = {
        "caustic_flow_l_hr": 250.0,
        "caustic_mass_kg_hr": 300.0,
    }

    payload = {
        "packing_type": "pall_ring_1_inch_ceramic",
        "gas_flow_kg_hr": 1000.0,
        "gas_temperature_k": 300.0,
        "gas_pressure_pa": 101325.0,
        "gas_molecular_weight": 29.0,
        "liquid_flow_kg_hr": 500.0,
        "percent_of_flood": 70.0,
        "acid_gas_removed_kg_hr": 10.0,
        "caustic_concentration_pct": 20.0,
    }

    response = client.post("/", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert data["gas_density_kg_m3"] == 1.2
    assert data["flooding_velocity_m_s"] == 2.5
    assert data["design_velocity_m_s"] == 1.75
    assert data["caustic_requirement"]["caustic_flow_l_hr"] == 250.0

    # Ensure boundaries are invoked with parsed Pydantic content
    mdensity.assert_called_once_with(300.0, 101325.0, 29.0)
    mflood.assert_called_once()
    mdiameter.assert_called_once_with(
        gas_flow_kg_hr=1000.0,
        gas_density=1.2,
        flooding_velocity=2.5,
        percent_of_flood=70.0,
    )


def test_calculate_scrubber_error_handling(mock_tools) -> None:
    """Verify that arithmetic errors gracefully map to 422 errors."""
    mdensity, _, _, _, _ = mock_tools
    mdensity.side_effect = ZeroDivisionError("division by zero gas density test")

    payload = {
        "packing_type": "pall_ring_1_inch_ceramic",
        "gas_flow_kg_hr": 1000.0,
        "gas_temperature_k": 300.0,
        "gas_pressure_pa": 101325.0,
        "gas_molecular_weight": 29.0,
        "liquid_flow_kg_hr": 500.0,
        "percent_of_flood": 70.0,
        "acid_gas_removed_kg_hr": 10.0,
        "caustic_concentration_pct": 20.0,
    }

    response = client.post("/", json=payload)
    assert response.status_code == 422
    assert response.json() == {"detail": "division by zero gas density test"}


def test_calculate_scrubber_invalid_packing(mock_tools) -> None:
    """Verify missing/invalid packing types get intercepted before reaching Tools."""
    payload = {
        "packing_type": "UNKNOWN_PACKING",
        "gas_flow_kg_hr": 1000.0,
        "gas_temperature_k": 300.0,
        "gas_pressure_pa": 101325.0,
        "gas_molecular_weight": 29.0,
        "liquid_flow_kg_hr": 500.0,
        "percent_of_flood": 70.0,
        "acid_gas_removed_kg_hr": 10.0,
        "caustic_concentration_pct": 20.0,
    }

    response = client.post("/", json=payload)
    assert response.status_code == 422
    assert "Unknown packing type" in response.json()["detail"]
