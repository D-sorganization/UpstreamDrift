"""Tests for the Syngas Water Calculator API router.

This test file adheres to the Fleet-Wide Shared Component Testing Strategy.
It mocks the `SyngasWaterCalculator` from `upstream_drift_tools` (Tools repo) to verify
that the API layer correctly implements the contract without testing the
internal mathematical logic.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.shared.python.calc_backend.routers.syngas_water import router

client = TestClient(router)


@pytest.fixture
def mock_calculator():
    """Mock the SyngasWaterCalculator securely from Tools."""
    with patch(
        "src.shared.python.calc_backend.routers.syngas_water.SyngasWaterCalculator"
    ) as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_estimate_risk():
    """Mock the estimate_condensation_risk function."""
    with patch(
        "src.shared.python.calc_backend.routers.syngas_water.estimate_condensation_risk"
    ) as mock_func:
        yield mock_func


def test_calculate_syngas_water_success(mock_calculator, mock_estimate_risk) -> None:
    """Validate that the API perfectly passes through inputs and returns results matching contract."""
    mock_result = MagicMock()
    mock_result.mole_fraction_water = 0.05
    mock_result.water_content_mg_per_nm3 = 50000.0
    mock_result.water_content_ppmv = 50000.0
    mock_result.water_content_g_per_m3 = 50.0
    mock_result.water_content_lb_per_mmscf = 3000.0
    mock_result.vapor_pressure_bar = 0.5
    mock_result.dew_point_c = 80.0

    mock_calculator.calculate_water_content.return_value = mock_result
    mock_estimate_risk.return_value = {
        "dew_point_c": 80.0,
        "temperature_margin_c": 20.0,
        "condensation_risk": "Low",
        "condensation_occurring": False,
        "recommended_temperature_c": 95.0,
    }

    payload = {
        "temperature_c": 100.0,
        "pressure_bar": 10.0,
        "composition_key": "typical_syngas",
        "method": "auto",
    }

    response = client.post("/", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert data["water_content"]["mole_fraction_water"] == 0.05
    assert data["water_content"]["dew_point_c"] == 80.0
    assert data["risk_assessment"]["temperature_margin_c"] == 20.0
    assert data["risk_assessment"]["condensation_risk"] == "Low"

    mock_calculator.calculate_water_content.assert_called_once_with(
        100.0, 10.0, "typical_syngas", "auto"
    )
    mock_estimate_risk.assert_called_once_with(100.0, 10.0)


def test_calculate_syngas_water_fallback() -> None:
    """Test fallback logic when tools are not imported.
    Simulate ImportError on the tools module.
    """
    with patch.dict(
        "sys.modules",
        {"upstream_drift_tools.process_calculators.syngas_water_calculator": None},
    ):
        payload = {
            "temperature_c": 100.0,
            "pressure_bar": 10.0,
            "composition_key": "typical_syngas",
            "method": "auto",
        }

        response = client.post("/", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "water_content" in data
        assert "risk_assessment" in data
