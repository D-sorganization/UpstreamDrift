"""Financial calculator router.  See issue #613."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..contracts.financial import (
    FinancialRequest,
    FinancialResponse,
    FinancialResultsOut,
)

router = APIRouter(prefix="/api/calc/financial", tags=["financial"])


@router.post("", response_model=FinancialResponse)
def calculate_financial(request: FinancialRequest) -> FinancialResponse:
    """Calculate financial metrics and optional yearly projections."""
    from upstream_drift_tools.process_calculators.financial_calculator import (
        FinancialModelCalculator,
        FinancialParameters,
    )

    calc = FinancialModelCalculator()
    params = FinancialParameters(
        plant_capacity_tpd=request.plant_capacity_tpd,
        operating_days_per_year=request.operating_days_per_year,
        capacity_utilization=request.capacity_utilization,
        product_price_per_ton=request.product_price_per_ton,
        byproduct_revenue_per_ton=request.byproduct_revenue_per_ton,
        byproduct_yield_factor=request.byproduct_yield_factor,
        feedstock_cost_per_ton=request.feedstock_cost_per_ton,
        labor_cost_per_ton=request.labor_cost_per_ton,
        utilities_cost_per_ton=request.utilities_cost_per_ton,
        maintenance_cost_per_ton=request.maintenance_cost_per_ton,
        consumables_cost_per_ton=request.consumables_cost_per_ton,
        fixed_labor_cost_annual=request.fixed_labor_cost_annual,
        insurance_annual=request.insurance_annual,
        property_tax_annual=request.property_tax_annual,
        admin_overhead_annual=request.admin_overhead_annual,
        total_capital_investment=request.total_capital_investment,
        debt_ratio=request.debt_ratio,
        interest_rate=request.interest_rate,
        depreciation_years=request.depreciation_years,
        tax_rate=request.tax_rate,
        baghouse_operating_cost_per_ton=request.baghouse_operating_cost_per_ton,
        scrubber_operating_cost_per_ton=request.scrubber_operating_cost_per_ton,
        glass_raw_material_cost_per_ton=request.glass_raw_material_cost_per_ton,
    )

    try:
        results = calc.calculate_financial_model(params)
    except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    projections: list[dict[str, float]] = []
    if request.projection_years > 0:
        try:
            projections = calc.generate_yearly_projections(request.projection_years)
        except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return FinancialResponse(
        results=FinancialResultsOut(
            annual_feedstock_tons=results.annual_feedstock_tons,
            annual_product_tons=results.annual_product_tons,
            total_revenue=results.total_revenue,
            total_variable_costs=results.total_variable_costs,
            total_fixed_costs=results.total_fixed_costs,
            ebitda=results.ebitda,
            net_income=results.net_income,
            revenue_per_ton=results.revenue_per_ton,
            total_cost_per_ton=results.total_cost_per_ton,
            margin_per_ton=results.margin_per_ton,
            roe=results.roe,
            roa=results.roa,
            payback_period_years=results.payback_period_years,
        ),
        projections=projections,
    )
