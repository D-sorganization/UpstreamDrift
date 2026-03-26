"""Tests for the Acid Gas Dewpoint API router.

This test file explicitly adheres to the Fleet-Wide Shared Component Testing Strategy.
It mocks the `AcidGasDewpointCalculator` from `upstream_drift_tools` (Tools repo) to verify
that the `UpstreamDrift` API layer correctly implements the contract without testing the
underlying math or logic, which is the responsibility of the `Tools` repository.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.shared.python.calc_backend.routers.acid_gas_dewpoint import router

_app = FastAPI()
_app.include_router(router)
client = TestClient(_app)


@pytest.fixture
def mock_calculator():
    """Mock the AcidGasDewpointCalculator to adhere to Shared Component Strategy."""
    with patch(
        "upstream_drift_tools.process_calculators.acid_gas_dewpoint_calculator.AcidGasDewpointCalculator"
    ) as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance
        yield mock_instance


def test_calculate_acid_gas_dewpoint_success(mock_calculator) -> None:
    """Test successful dewpoint calculation via API.

    Validates that:
    1. The API calls the calculator with the correct parameters from the request payload.
    2. The API correctly transforms the calculator's response into the expected JSON format.
    """
    # Mock return value of calculation to assert contract out
    mock_result = MagicMock()
    mock_result.overall_dewpoint_c = 145.2
    mock_result.limiting_component = "H2SO4"
    mock_result.dewpoint_margin_c = 25.5
    mock_result.condensation_risk = "Low"
    mock_result.h2o_dewpoint_c = 100.0
    mock_result.h2o_vapor_pressure_pa = 101325.0
    mock_result.h2o_partial_pressure_pa = 50000.0
    mock_result.hf_dewpoint_c = 20.0
    mock_result.hf_vapor_pressure_pa = 1000.0
    mock_result.hf_partial_pressure_pa = 10.0
    mock_result.hcl_dewpoint_c = -85.0
    mock_result.hcl_vapor_pressure_pa = 2000.0
    mock_result.hcl_partial_pressure_pa = 20.0
    mock_result.h2s_dewpoint_c = -60.0
    mock_result.h2s_vapor_pressure_pa = 3000.0
    mock_result.h2s_partial_pressure_pa = 30.0
    mock_result.warnings = ["Test warning"]
    mock_result.calculation_method = "empirical"

    mock_calculator.calculate_dewpoint_mixture.return_value = mock_result

    payload = {
        "temperature_c": 170.0,
        "pressure_bar": 1.0,
        "h2o_fraction": 0.1,
        "hf_fraction": 0.0,
        "hcl_fraction": 0.01,
        "h2s_fraction": 0.05,
        "method": "empirical",
    }

    response = client.post("/api/calc/acid-gas-dewpoint", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert data["overall_dewpoint_c"] == 145.2
    assert data["limiting_component"] == "H2SO4"
    assert data["condensation_risk"] == "Low"
    assert data["warnings"] == ["Test warning"]

    # Assert isolation boundary: verify we passed the correctly formulated parameters to the mocked shared tool
    mock_calculator.calculate_dewpoint_mixture.assert_called_once()
    _, kwargs = mock_calculator.calculate_dewpoint_mixture.call_args
    assert kwargs["temperature_c"] == 170.0
    assert kwargs["pressure_bar"] == 1.0
    assert kwargs["method"] == "empirical"
    assert kwargs["composition"].h2o == 0.1
    assert kwargs["composition"].hcl == 0.01


def test_calculate_acid_gas_dewpoint_error_handling(mock_calculator) -> None:
    """Test that arithmetic errors from the engine translate to 422 HTTP exceptions."""
    mock_calculator.calculate_dewpoint_mixture.side_effect = ValueError("Invalid composition")

    payload = {
        "temperature_c": 170.0,
        "pressure_bar": 1.0,
        "h2o_fraction": 0.1,
        "hf_fraction": 0.0,
        "hcl_fraction": 0.01,
        "h2s_fraction": 0.05,
        "method": "empirical",
    }

    response = client.post("/api/calc/acid-gas-dewpoint", json=payload)

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid composition"}
