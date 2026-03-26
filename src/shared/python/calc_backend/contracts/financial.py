"""Pydantic contracts for financial calculator endpoints.  See issue #613."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FinancialRequest(BaseModel):
    """Request model for financial analysis calculation."""

    # Plant Capacity & Operations
    plant_capacity_tpd: float = Field(default=100.0, ge=0, description="Tons per day")
    operating_days_per_year: int = Field(
        default=330, ge=0, le=366, description="Operating days per year"
    )
    capacity_utilization: float = Field(
        default=0.85, ge=0, le=1, description="Capacity utilization (0-1)"
    )
    # Revenue
    product_price_per_ton: float = Field(default=0.0, ge=0, description="Product price [$/ton]")
    byproduct_revenue_per_ton: float = Field(
        default=0.0, ge=0, description="Byproduct revenue [$/ton]"
    )
    byproduct_yield_factor: float = Field(default=0.0, ge=0, description="Byproduct yield factor")
    # Variable costs ($/ton feedstock)
    feedstock_cost_per_ton: float = Field(default=0.0, ge=0)
    labor_cost_per_ton: float = Field(default=0.0, ge=0)
    utilities_cost_per_ton: float = Field(default=0.0, ge=0)
    maintenance_cost_per_ton: float = Field(default=0.0, ge=0)
    consumables_cost_per_ton: float = Field(default=0.0, ge=0)
    # Fixed costs ($/year)
    fixed_labor_cost_annual: float = Field(default=0.0, ge=0)
    insurance_annual: float = Field(default=0.0, ge=0)
    property_tax_annual: float = Field(default=0.0, ge=0)
    admin_overhead_annual: float = Field(default=0.0, ge=0)
    # Capital & Financial
    total_capital_investment: float = Field(default=0.0, ge=0)
    debt_ratio: float = Field(default=0.0, ge=0, le=1)
    interest_rate: float = Field(default=0.0, ge=0, le=1)
    depreciation_years: int = Field(default=20, ge=1)
    tax_rate: float = Field(default=0.0, ge=0, le=1)
    # Equipment-specific
    baghouse_operating_cost_per_ton: float = Field(default=0.0, ge=0)
    scrubber_operating_cost_per_ton: float = Field(default=0.0, ge=0)
    glass_raw_material_cost_per_ton: float = Field(default=0.0, ge=0)
    # Projections
    projection_years: int = Field(
        default=0,
        ge=0,
        le=50,
        description="Years of projections to generate (0 = skip)",
    )


class FinancialResultsOut(BaseModel):
    """Core financial metrics."""

    annual_feedstock_tons: float
    annual_product_tons: float
    total_revenue: float
    total_variable_costs: float
    total_fixed_costs: float
    ebitda: float
    net_income: float
    revenue_per_ton: float
    total_cost_per_ton: float
    margin_per_ton: float
    roe: float
    roa: float
    payback_period_years: float


class FinancialResponse(BaseModel):
    """Response model for financial calculation."""

    results: FinancialResultsOut
    projections: list[dict[str, float]] = Field(
        default_factory=list,
        description="Yearly financial projections (empty when projection_years=0)",
    )
