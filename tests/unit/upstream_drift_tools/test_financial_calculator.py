"""Tests for sidekick.process_calculators.financial_calculator (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from sidekick.process_calculators.financial_calculator import (
    FinancialModelCalculator,
    FinancialParameters,
    FinancialResults,
)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


class TestFinancialParameters:
    def test_defaults_zero(self) -> None:
        p = FinancialParameters()
        assert p.plant_capacity_tpd == 0.0
        assert p.operating_days_per_year == 0
        assert p.capacity_utilization == 0.0

    def test_construct_with_values(self) -> None:
        p = FinancialParameters(
            plant_capacity_tpd=500.0,
            operating_days_per_year=330,
            capacity_utilization=0.9,
        )
        assert p.plant_capacity_tpd == 500.0
        assert p.operating_days_per_year == 330


class TestFinancialResults:
    def test_defaults_zero(self) -> None:
        r = FinancialResults()
        assert r.total_revenue == 0.0
        assert r.net_income == 0.0


# ---------------------------------------------------------------------------
# FinancialModelCalculator.calculate_financial_model
# ---------------------------------------------------------------------------


def _base_params(**overrides: float | int) -> FinancialParameters:
    p = FinancialParameters(
        plant_capacity_tpd=100.0,
        operating_days_per_year=330,
        capacity_utilization=1.0,
        product_price_per_ton=200.0,
        feedstock_cost_per_ton=50.0,
        total_capital_investment=10_000_000.0,
        debt_ratio=0.5,
        interest_rate=0.06,
        depreciation_years=10,
        tax_rate=0.25,
    )
    for k, v in overrides.items():
        setattr(p, k, v)
    return p


class TestCalculateFinancialModel:
    _CALC = FinancialModelCalculator()

    def test_returns_financial_results(self) -> None:
        result = self._CALC.calculate_financial_model(_base_params())
        assert isinstance(result, FinancialResults)

    def test_annual_feedstock_positive(self) -> None:
        result = self._CALC.calculate_financial_model(_base_params())
        assert result.annual_feedstock_tons > 0.0

    def test_revenue_positive_with_positive_price(self) -> None:
        result = self._CALC.calculate_financial_model(_base_params())
        assert result.total_revenue > 0.0

    def test_higher_capacity_higher_revenue(self) -> None:
        low = self._CALC.calculate_financial_model(
            _base_params(plant_capacity_tpd=50.0)
        )
        high = self._CALC.calculate_financial_model(
            _base_params(plant_capacity_tpd=200.0)
        )
        assert high.total_revenue > low.total_revenue

    def test_zero_capacity_zero_revenue(self) -> None:
        result = self._CALC.calculate_financial_model(
            _base_params(plant_capacity_tpd=0.0)
        )
        assert result.total_revenue == 0.0

    def test_negative_capital_raises(self) -> None:
        with pytest.raises(AssertionError):
            self._CALC.calculate_financial_model(
                _base_params(total_capital_investment=-1.0)
            )

    def test_negative_operating_days_raises(self) -> None:
        with pytest.raises(AssertionError):
            self._CALC.calculate_financial_model(
                _base_params(operating_days_per_year=-1)
            )

    def test_higher_price_higher_revenue(self) -> None:
        low = self._CALC.calculate_financial_model(
            _base_params(product_price_per_ton=100.0)
        )
        high = self._CALC.calculate_financial_model(
            _base_params(product_price_per_ton=400.0)
        )
        assert high.total_revenue > low.total_revenue
