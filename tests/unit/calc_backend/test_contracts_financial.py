"""Tests for src.shared.python.calc_backend.contracts.financial (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from src.shared.python.calc_backend.contracts.financial import (
    FinancialRequest,
    FinancialResponse,
    FinancialResultsOut,
)


class TestFinancialRequest:
    def test_contracts_financial_default_construction(self) -> None:
        req = FinancialRequest()
        assert isinstance(req, FinancialRequest)

    def test_default_plant_capacity(self) -> None:
        req = FinancialRequest()
        assert req.plant_capacity_tpd == 100.0

    def test_default_operating_days(self) -> None:
        req = FinancialRequest()
        assert req.operating_days_per_year == 330

    def test_default_capacity_utilization(self) -> None:
        req = FinancialRequest()
        assert req.capacity_utilization == pytest.approx(0.85)

    def test_contracts_financial_custom_values(self) -> None:
        req = FinancialRequest(plant_capacity_tpd=200.0, operating_days_per_year=300)
        assert req.plant_capacity_tpd == 200.0
        assert req.operating_days_per_year == 300

    def test_negative_capacity_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FinancialRequest(plant_capacity_tpd=-1.0)

    def test_operating_days_over_366_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FinancialRequest(operating_days_per_year=400)

    def test_capacity_utilization_over_1_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FinancialRequest(capacity_utilization=1.5)

    def test_debt_ratio_over_1_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FinancialRequest(debt_ratio=2.0)

    def test_tax_rate_valid(self) -> None:
        req = FinancialRequest(tax_rate=0.25)
        assert req.tax_rate == pytest.approx(0.25)

    def test_projection_years_default(self) -> None:
        req = FinancialRequest()
        assert req.projection_years == 0

    def test_projection_years_over_50_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FinancialRequest(projection_years=51)

    def test_depreciation_years_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FinancialRequest(depreciation_years=0)


class TestFinancialResultsOut:
    def _make_results(self, **kwargs) -> FinancialResultsOut:
        defaults = {
            "annual_feedstock_tons": 33000.0,
            "annual_product_tons": 28000.0,
            "total_revenue": 5_000_000.0,
            "total_variable_costs": 2_000_000.0,
            "total_fixed_costs": 500_000.0,
            "ebitda": 2_500_000.0,
            "net_income": 1_800_000.0,
            "revenue_per_ton": 150.0,
            "total_cost_per_ton": 90.0,
            "margin_per_ton": 60.0,
            "roe": 0.15,
            "roa": 0.10,
            "payback_period_years": 6.5,
        }
        defaults.update(kwargs)
        return FinancialResultsOut(**defaults)

    def test_contracts_financial_construction(self) -> None:
        result = self._make_results()
        assert isinstance(result, FinancialResultsOut)

    def test_ebitda_stored(self) -> None:
        result = self._make_results(ebitda=3_000_000.0)
        assert result.ebitda == pytest.approx(3_000_000.0)

    def test_negative_net_income_allowed(self) -> None:
        # Losses are valid financial results
        result = self._make_results(net_income=-500_000.0)
        assert result.net_income == pytest.approx(-500_000.0)


class TestFinancialResponse:
    def _make_results_out(self) -> FinancialResultsOut:
        return FinancialResultsOut(
            annual_feedstock_tons=33000.0,
            annual_product_tons=28000.0,
            total_revenue=5_000_000.0,
            total_variable_costs=2_000_000.0,
            total_fixed_costs=500_000.0,
            ebitda=2_500_000.0,
            net_income=1_800_000.0,
            revenue_per_ton=150.0,
            total_cost_per_ton=90.0,
            margin_per_ton=60.0,
            roe=0.15,
            roa=0.10,
            payback_period_years=6.5,
        )

    def test_construction_no_projections(self) -> None:
        resp = FinancialResponse(results=self._make_results_out())
        assert isinstance(resp, FinancialResponse)
        assert resp.projections == []

    def test_with_projections(self) -> None:
        proj = [{"year": 1.0, "revenue": 5_000_000.0}]
        resp = FinancialResponse(results=self._make_results_out(), projections=proj)
        assert len(resp.projections) == 1

    def test_results_field_stored(self) -> None:
        ro = self._make_results_out()
        resp = FinancialResponse(results=ro)
        assert resp.results.ebitda == pytest.approx(2_500_000.0)
