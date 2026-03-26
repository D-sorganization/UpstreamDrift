"""Tests for the Flare Calculator API router.

This test file adheres to the Fleet-Wide Shared Component Testing Strategy.
It mocks the `FlareCalculator` from `upstream_drift_tools` (Tools repo) to verify
that the API layer correctly implements the contract without testing the
internal mathematical logic.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.shared.python.calc_backend.routers.flare import router

_app = FastAPI()
_app.include_router(router)
client = TestClient(_app)


@pytest.fixture
def mock_calculator():
    """Mock the FlareCalculator securely from Tools."""
    with patch("upstream_drift_tools.process_calculators.FlareCalculator") as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance
        yield mock_instance


def test_calculate_flare_success(mock_calculator) -> None:
    """Validate that the API perfectly passes through inputs and returns results matching contract."""
    mock_design = MagicMock()
    mock_design.height = 10.0
    mock_design.diameter = 1.5
    mock_design.exit_velocity = 20.0
    mock_design.heat_release = 50000.0
    mock_design.radiation_intensity = 1.6

    # Configure mock method returns
    mock_calculator.calculate_flare_size.return_value = mock_design
    mock_calculator.calculate_radiation_zones.return_value = {
        "lethal": 15.0,
        "damage": 30.0,
        "safe": 60.0,
        "comfort": 100.0,
    }
    mock_calculator.calculate_combustion_efficiency.return_value = 0.98

    payload = {
        "total_flow_kg_hr": 2000.0,
        "gas_composition": {"CH4": 1.0},
        "temperature_k": 300.0,
        "pressure_bar": 2.0,
    }

    response = client.post("/api/calc/flare", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert data["design"]["height_m"] == 10.0
    assert data["design"]["heat_release_kw"] == 50000.0
    assert data["radiation_zones"]["safe_m"] == 60.0
    assert data["combustion_efficiency"] == 0.98

    # Verify boundaries are called correctly
    mock_calculator.calculate_flare_size.assert_called_once()
    mock_calculator.calculate_radiation_zones.assert_called_once_with(mock_design)
    mock_calculator.calculate_combustion_efficiency.assert_called_once()


def test_calculate_flare_error_handling(mock_calculator) -> None:
    """Verify that arithmetic errors gracefully map to 422 errors through the router boundary."""
    mock_calculator.calculate_flare_size.side_effect = ValueError("Flow must be positive")

    payload = {
        "total_flow_kg_hr": 2000.0,
        "gas_composition": {"CH4": 1.0},
        "temperature_k": 300.0,
        "pressure_bar": 2.0,
    }

    response = client.post("/api/calc/flare", json=payload)
    assert response.status_code == 422
    assert response.json() == {"detail": "Flow must be positive"}
