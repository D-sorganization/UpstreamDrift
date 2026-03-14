"""Tests for the Financial Calculator API router.

This test file adheres to the Fleet-Wide Shared Component Testing Strategy.
It mocks the `FinancialModelCalculator` from `upstream_drift_tools` (Tools repo)
to verify that the API layer correctly implements the contract without testing the
internal mathematical logic directly.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.shared.python.calc_backend.routers.financial import router

client = TestClient(router)


@pytest.fixture
def mock_calculator():
    """Mock the scrubber calculators securely from Tools."""
    with patch(
        "upstream_drift_tools.process_calculators.financial_calculator.FinancialModelCalculator"
    ) as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance
        yield mock_instance


def test_calculate_financial_success(mock_calculator) -> None:
    """Validate that the API perfectly passes through inputs and parses math results correctly."""

    mock_results = MagicMock()
    mock_results.annual_feedstock_tons = 10000.0
    mock_results.annual_product_tons = 8000.0
    mock_results.total_revenue = 1000000.0
    mock_results.total_variable_costs = 400000.0
    mock_results.total_fixed_costs = 200000.0
    mock_results.ebitda = 400000.0
    mock_results.net_income = 250000.0
    mock_results.revenue_per_ton = 125.0
    mock_results.total_cost_per_ton = 75.0
    mock_results.margin_per_ton = 50.0
    mock_results.roe = 15.0
    mock_results.roa = 10.0
    mock_results.payback_period_years = 5.0

    # Only return string dict format for the projection
    mock_calculator.calculate_financial_model.return_value = mock_results
    mock_calculator.generate_yearly_projections.return_value = [
        {"year": 1, "revenue": 1000000.0},
        {"year": 2, "revenue": 1100000.0},
    ]

    payload = {
        "plant_capacity_tpd": 100.0,
        "operating_days_per_year": 330.0,
        "capacity_utilization": 0.9,
        "product_price_per_ton": 150.0,
        "byproduct_revenue_per_ton": 10.0,
        "byproduct_yield_factor": 0.05,
        "feedstock_cost_per_ton": 20.0,
        "labor_cost_per_ton": 15.0,
        "utilities_cost_per_ton": 8.0,
        "maintenance_cost_per_ton": 5.0,
        "consumables_cost_per_ton": 2.0,
        "fixed_labor_cost_annual": 500000.0,
        "insurance_annual": 50000.0,
        "property_tax_annual": 25000.0,
        "admin_overhead_annual": 100000.0,
        "total_capital_investment": 5000000.0,
        "debt_ratio": 0.6,
        "interest_rate": 0.05,
        "depreciation_years": 10.0,
        "tax_rate": 0.21,
        "baghouse_operating_cost_per_ton": 1.0,
        "scrubber_operating_cost_per_ton": 2.0,
        "glass_raw_material_cost_per_ton": 0.0,
        "projection_years": 2,
    }

    response = client.post("/", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert data["results"]["annual_feedstock_tons"] == 10000.0
    assert data["results"]["net_income"] == 250000.0
    assert data["results"]["roe"] == 15.0
    assert len(data["projections"]) == 2
    assert data["projections"][0]["revenue"] == 1000000.0

    mock_calculator.calculate_financial_model.assert_called_once()
    mock_calculator.generate_yearly_projections.assert_called_once_with(2)


def test_calculate_financial_error_handling(mock_calculator) -> None:
    """Verify that configuration errors gracefully map to 422 errors through the router boundary."""
    mock_calculator.calculate_financial_model.side_effect = ValueError(
        "Invalid tax rate"
    )

    payload = {
        "plant_capacity_tpd": 100.0,
        "operating_days_per_year": 330.0,
        "capacity_utilization": 0.9,
        "product_price_per_ton": 150.0,
        "byproduct_revenue_per_ton": 10.0,
        "byproduct_yield_factor": 0.05,
        "feedstock_cost_per_ton": 20.0,
        "labor_cost_per_ton": 15.0,
        "utilities_cost_per_ton": 8.0,
        "maintenance_cost_per_ton": 5.0,
        "consumables_cost_per_ton": 2.0,
        "fixed_labor_cost_annual": 500000.0,
        "insurance_annual": 50000.0,
        "property_tax_annual": 25000.0,
        "admin_overhead_annual": 100000.0,
        "total_capital_investment": 5000000.0,
        "debt_ratio": 0.6,
        "interest_rate": 0.05,
        "depreciation_years": 10.0,
        "tax_rate": -0.5,  # Invalid param
        "baghouse_operating_cost_per_ton": 1.0,
        "scrubber_operating_cost_per_ton": 2.0,
        "glass_raw_material_cost_per_ton": 0.0,
        "projection_years": 0,
    }

    response = client.post("/", json=payload)
    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid tax rate"}
