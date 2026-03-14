"""Tests for the Baghouse Calculator API router.

This test file adheres to the Fleet-Wide Shared Component Testing Strategy.
It mocks the `BaghouseCalculator` from `upstream_drift_tools` (Tools repo) to verify
that the API layer correctly implements the contract without testing the
internal mathematical logic.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.shared.python.calc_backend.routers.baghouse import router

client = TestClient(router)


@pytest.fixture
def mock_calculator():
    """Mock the BaghouseCalculator securely from Tools."""
    with patch(
        "upstream_drift_tools.process_calculators.BaghouseCalculator"
    ) as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance
        yield mock_instance


def test_calculate_baghouse_success(mock_calculator) -> None:
    """Validate that the API perfectly passes through inputs and returns results matching contract."""
    mock_result = MagicMock()
    mock_result.carbon_removed_rate = 55.0
    mock_result.ash_removed_rate = 14.0
    mock_result.total_solids_removed_rate = 69.0
    mock_result.drum_fill_time_hours = 12.0
    mock_result.drum_fill_time_days = 0.5
    mock_result.carbon_only_fill_time_hours = 15.0
    mock_result.ash_only_fill_time_hours = 60.0
    mock_result.clean_gas_flow_rate = 5000.0
    mock_result.flow_acfm = 2000.0
    mock_result.flow_scfm = 1800.0
    mock_result.air_to_cloth_ratio = 2.0
    mock_result.outlet_temperature_c = 150.0
    mock_result.ash_stream_composition = {"carbon_fraction": 0.8, "ash_fraction": 0.2}
    mock_result.removal_efficiency = {"carbon": 98.0, "ash": 95.0}

    mock_calculator.calculate.return_value = mock_result

    payload = {
        "gas_flow_kg_s": 1.5,
        "inlet_temp_k": 450.0,
        "pressure_pa": 105000.0,
        "composition": {"N2": 1.0},
        "solid_carbon_in_kg_hr": 56.0,
        "ash_in_kg_hr": 14.5,
        "carbon_removal_efficiency": 0.98,
        "ash_removal_efficiency": 0.95,
        "heat_loss_w": 5000.0,
        "drum_volume_m3": 2.5,
        "solid_density_kg_m3": 850.0,
        "bag_area_ft2": 1000.0,
    }

    response = client.post("/", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert data["carbon_removed_rate_kg_hr"] == 55.0
    assert data["drum_fill_time_hours"] == 12.0
    assert data["air_to_cloth_ratio"] == 2.0
    assert data["ash_stream_composition"] == {
        "carbon_fraction": 0.8,
        "ash_fraction": 0.2,
    }

    mock_calculator.calculate.assert_called_once()
    _, kwargs = mock_calculator.calculate.call_args
    assert kwargs["gas_flow_kg_s"] == 1.5
    assert kwargs["heat_loss_w"] == 5000.0


def test_calculate_baghouse_error_handling(mock_calculator) -> None:
    """Verify that arithmetic errors gracefully map to 422 errors through the router boundary."""
    mock_calculator.calculate.side_effect = ValueError("Flow must be positive")

    payload = {
        "gas_flow_kg_s": -1.5,
        "inlet_temp_k": 450.0,
        "pressure_pa": 105000.0,
        "composition": {"N2": 1.0},
        "solid_carbon_in_kg_hr": 56.0,
        "ash_in_kg_hr": 14.5,
        "carbon_removal_efficiency": 0.98,
        "ash_removal_efficiency": 0.95,
        "heat_loss_w": 5000.0,
        "drum_volume_m3": 2.5,
        "solid_density_kg_m3": 850.0,
        "bag_area_ft2": 1000.0,
    }

    response = client.post("/", json=payload)
    assert response.status_code == 422
    assert response.json() == {"detail": "Flow must be positive"}
