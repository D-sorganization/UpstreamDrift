"""Tests for the WGS Reactor Calculator API router.

This test file adheres to the Fleet-Wide Shared Component Testing Strategy.
It mocks the `WGSReactorEngine` from `upstream_drift_tools` (Tools repo) to verify
that the API layer correctly implements the contract without testing the
internal mathematical logic.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.shared.python.calc_backend.routers.wgs_reactor import router

client = TestClient(router)


@pytest.fixture
def mock_engine():
    """Mock the WGSReactorEngine securely from Tools."""
    with patch(
        "upstream_drift_tools.process_calculators.WGSReactorEngine"
    ) as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance
        yield mock_instance


def test_calculate_wgs_success(mock_engine) -> None:
    """Validate that the API perfectly passes through inputs and returns results matching contract."""
    mock_engine.calculate_equilibrium_composition.return_value = {
        "conversion": 85.5,
        "composition": {"CO": 2.0, "H2O": 10.0, "CO2": 30.0, "H2": 58.0},
        "h2_co_ratio": 29.0,
        "equilibrium_constant": 4.5,
        "heat_released": -41.2,
    }

    mock_engine.size_wgs_reactor.return_value = {
        "reactor_volume": 5.0,
        "catalyst_volume": 4.0,
        "diameter": 1.5,
        "length": 4.5,
        "heat_duty": 150.0,
        "ghsv": 3000.0,
    }

    payload = {
        "inlet_composition": {"CO": 20.0, "H2O": 40.0, "CO2": 10.0, "H2": 30.0},
        "temperature_k": 500.0,
        "pressure_bar": 25.0,
        "steam_ratio": 2.0,
        "feed_rate_kmol_hr": 1000.0,
        "catalyst_type": "high_temp",
    }

    response = client.post("/", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert data["equilibrium"]["conversion_pct"] == 85.5
    assert data["equilibrium"]["composition"]["CO"] == 2.0
    assert data["sizing"]["reactor_volume_m3"] == 5.0

    # Verify boundaries are called correctly
    mock_engine.calculate_equilibrium_composition.assert_called_once_with(
        inlet_composition={"CO": 20.0, "H2O": 40.0, "CO2": 10.0, "H2": 30.0},
        temperature=500.0,
        pressure=25.0,
        steam_ratio=2.0,
    )
    mock_engine.size_wgs_reactor.assert_called_once_with(
        feed_rate=1000.0, conversion=85.5, temperature=500.0, catalyst_type="high_temp"
    )


def test_calculate_wgs_error_handling(mock_engine) -> None:
    """Verify that arithmetic errors gracefully map to 422 errors through the router boundary."""
    mock_engine.calculate_equilibrium_composition.side_effect = ValueError(
        "Invalid input"
    )

    payload = {
        "inlet_composition": {"CO": 20.0},
        "temperature_k": 500.0,
        "pressure_bar": 25.0,
        "steam_ratio": 2.0,
        "feed_rate_kmol_hr": 1000.0,
        "catalyst_type": "high_temp",
    }

    response = client.post("/", json=payload)
    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid input"}
