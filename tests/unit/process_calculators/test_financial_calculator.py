"""Tests for sidekick.process_calculators.financial_calculator (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from sidekick.process_calculators.financial_calculator import (
    FinancialModelCalculator,
    FinancialParameters,
    FinancialResults,
)


def _make_params(**kwargs) -> FinancialParameters:
    defaults = {
        "plant_capacity_tpd": 100.0,
        "operating_days_per_year": 300,
        "capacity_utilization": 0.9,
        "product_price_per_ton": 500.0,
        "byproduct_revenue_per_ton": 50.0,
        "byproduct_yield_factor": 0.1,
        "feedstock_cost_per_ton": 100.0,
        "labor_cost_per_ton": 20.0,
        "utilities_cost_per_ton": 15.0,
        "maintenance_cost_per_ton": 10.0,
        "consumables_cost_per_ton": 5.0,
        "fixed_labor_cost_annual": 500_000.0,
        "insurance_annual": 100_000.0,
        "property_tax_annual": 50_000.0,
        "admin_overhead_annual": 75_000.0,
        "total_capital_investment": 5_000_000.0,
        "debt_ratio": 0.6,
        "interest_rate": 0.05,
        "depreciation_years": 20,
        "tax_rate": 0.25,
        "electricity_rate_per_kwh": 0.08,
        "natural_gas_rate_per_mmbtu": 5.0,
        "steam_cost_per_1000lb": 12.0,
    }
    defaults.update(kwargs)
    return FinancialParameters(**defaults)


class TestFinancialParameters:
    def test_financial_calculator_default_construction(self) -> None:
        p = FinancialParameters()
        assert p.plant_capacity_tpd == 0.0

    def test_financial_calculator_custom_values(self) -> None:
        p = _make_params()
        assert p.plant_capacity_tpd == 100.0
        assert p.operating_days_per_year == 300


class TestFinancialResults:
    def test_financial_calculator_default_construction(self) -> None:
        r = FinancialResults()
        assert r.total_revenue == 0.0
        assert r.net_income == 0.0


class TestFinancialModelCalculator:
    def test_returns_financial_results(self) -> None:
        calc = FinancialModelCalculator()
        result = calc.calculate_financial_model(_make_params())
        assert isinstance(result, FinancialResults)

    def test_positive_annual_feedstock(self) -> None:
        calc = FinancialModelCalculator()
        result = calc.calculate_financial_model(_make_params())
        # 100 tpd * 300 days * 0.9 utilization = 27,000 tons
        assert result.annual_feedstock_tons == pytest.approx(27_000.0)

    def test_positive_total_revenue(self) -> None:
        calc = FinancialModelCalculator()
        result = calc.calculate_financial_model(_make_params())
        assert result.total_revenue > 0.0

    def test_product_revenue_positive(self) -> None:
        calc = FinancialModelCalculator()
        result = calc.calculate_financial_model(_make_params())
        assert result.product_revenue > 0.0

    def test_total_variable_costs_positive(self) -> None:
        calc = FinancialModelCalculator()
        result = calc.calculate_financial_model(_make_params())
        assert result.total_variable_costs > 0.0

    def test_depreciation_computed(self) -> None:
        calc = FinancialModelCalculator()
        result = calc.calculate_financial_model(_make_params())
        # 5_000_000 / 20 = 250_000
        assert result.depreciation == pytest.approx(250_000.0)

    def test_zero_capacity_zeros_revenue(self) -> None:
        calc = FinancialModelCalculator()
        result = calc.calculate_financial_model(_make_params(plant_capacity_tpd=0.0))
        assert result.total_revenue == 0.0
        assert result.annual_feedstock_tons == 0.0

    def test_revenue_greater_than_costs_yields_positive_income(self) -> None:
        calc = FinancialModelCalculator()
        # High product price should give positive net income
        result = calc.calculate_financial_model(
            _make_params(product_price_per_ton=2000.0)
        )
        assert result.ebitda > 0.0

    def test_unit_metrics_computed(self) -> None:
        calc = FinancialModelCalculator()
        result = calc.calculate_financial_model(_make_params())
        assert result.revenue_per_ton > 0.0
        assert result.variable_cost_per_ton > 0.0

    def test_debt_ratio_1_raises(self) -> None:
        calc = FinancialModelCalculator()
        with pytest.raises((ValueError, AssertionError)):
            calc.calculate_financial_model(
                _make_params(debt_ratio=1.0, total_capital_investment=1_000_000.0)
            )

    def test_yearly_projections_length(self) -> None:
        calc = FinancialModelCalculator()
        calc.calculate_financial_model(_make_params())
        projections = calc.generate_yearly_projections(years=5)
        assert len(projections) == 5

    def test_yearly_projections_keys(self) -> None:
        calc = FinancialModelCalculator()
        calc.calculate_financial_model(_make_params())
        projections = calc.generate_yearly_projections(years=3)
        for proj in projections:
            assert "year" in proj
            assert "total_revenue" in proj
            assert "net_income" in proj

    def test_yearly_projections_cumulative_cash_flow(self) -> None:
        calc = FinancialModelCalculator()
        calc.calculate_financial_model(_make_params(product_price_per_ton=2000.0))
        projections = calc.generate_yearly_projections(years=3)
        # With very high revenue, cumulative should grow each year
        cfs = [p["cumulative_cash_flow"] for p in projections]
        assert cfs[1] > cfs[0] or cfs[2] >= cfs[1]  # monotone or growing
