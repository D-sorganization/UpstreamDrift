import pytest
from sidekick.process_calculators.financial_calculator import (
    FinancialModelCalculator,
    FinancialParameters,
    FinancialResults,
)


@pytest.fixture
def fin_calc() -> FinancialModelCalculator:
    return FinancialModelCalculator()


def test_financial_calculations(fin_calc: FinancialModelCalculator) -> None:
    params = FinancialParameters(
        plant_capacity_tpd=100.0,
        operating_days_per_year=300,
        capacity_utilization=0.9,
        product_price_per_ton=1000.0,
        byproduct_revenue_per_ton=50.0,
        byproduct_yield_factor=0.1,
        feedstock_cost_per_ton=100.0,
        labor_cost_per_ton=50.0,
        utilities_cost_per_ton=20.0,
        maintenance_cost_per_ton=10.0,
        consumables_cost_per_ton=5.0,
        fixed_labor_cost_annual=500000.0,
        insurance_annual=50000.0,
        property_tax_annual=20000.0,
        admin_overhead_annual=100000.0,
        total_capital_investment=10000000.0,
        debt_ratio=0.4,
        interest_rate=0.05,
        depreciation_years=10,
        tax_rate=0.25,
    )

    results = fin_calc.calculate_financial_model(params)
    assert isinstance(results, FinancialResults)

    # Check totals
    assert results.annual_feedstock_tons == 100.0 * 300 * 0.9
    assert results.product_revenue > 0.0
    assert results.total_variable_costs > 0.0
    assert results.gross_margin > 0.0
    assert results.net_income != 0.0
    assert results.roe != 0.0
    assert results.roa != 0.0
    assert results.payback_period_years > 0.0


def test_financial_debt_limit(fin_calc: FinancialModelCalculator) -> None:
    params = FinancialParameters(
        plant_capacity_tpd=100.0,
        operating_days_per_year=300,
        total_capital_investment=1000000.0,
        debt_ratio=1.0,
    )
    with pytest.raises(ValueError, match="debt_ratio must be < 1.0"):
        fin_calc.calculate_financial_model(params)


def test_zero_capital_investment(fin_calc: FinancialModelCalculator) -> None:
    params = FinancialParameters(
        plant_capacity_tpd=10.0,
        operating_days_per_year=10,
        total_capital_investment=0.0,
    )
    results = fin_calc.calculate_financial_model(params)
    assert results.roa == 0.0
    assert results.roe == 0.0
    assert results.depreciation == 0.0


def test_yearly_projections(fin_calc: FinancialModelCalculator) -> None:
    params = FinancialParameters(
        plant_capacity_tpd=100.0,
        operating_days_per_year=300,
        capacity_utilization=0.9,
        product_price_per_ton=1000.0,
        feedstock_cost_per_ton=100.0,
        total_capital_investment=10000000.0,
        debt_ratio=0.4,
    )
    fin_calc.calculate_financial_model(params)
    projections = fin_calc.generate_yearly_projections(5)

    assert len(projections) == 5
    assert projections[0]["year"] == 1
    assert projections[-1]["year"] == 5
    assert projections[-1]["total_revenue"] > projections[0]["total_revenue"]
