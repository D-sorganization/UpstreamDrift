"""Financial Model Calculator
==========================

Core calculation engine for comprehensive financial modeling of plant operations.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class FinancialParameters:
    """Core financial parameters for the plant model"""

    # Plant Capacity & Operations
    plant_capacity_tpd: float = 0.0  # Tons per day
    operating_days_per_year: int = 0  # Operating days
    capacity_utilization: float = 0.0  # Utilization

    # Revenue Parameters
    product_price_per_ton: float = 0.0  # $/ton of product
    byproduct_revenue_per_ton: float = 0.0  # $/ton byproduct revenue
    byproduct_yield_factor: float = 0.0  # Byproduct yield

    # Operating Expenses ($/ton of feedstock)
    feedstock_cost_per_ton: float = 0.0
    labor_cost_per_ton: float = 0.0
    utilities_cost_per_ton: float = 0.0
    maintenance_cost_per_ton: float = 0.0
    consumables_cost_per_ton: float = 0.0

    # Fixed Costs ($/year)
    fixed_labor_cost_annual: float = 0.0
    insurance_annual: float = 0.0
    property_tax_annual: float = 0.0
    admin_overhead_annual: float = 0.0

    # Capital & Financial
    total_capital_investment: float = 0.0  # Capital investment
    debt_ratio: float = 0.0  # Debt financing ratio
    interest_rate: float = 0.0  # Interest rate
    depreciation_years: int = 0
    tax_rate: float = 0.0  # Tax rate

    # Energy Parameters
    electricity_rate_per_kwh: float = 0.0
    natural_gas_rate_per_mmbtu: float = 0.0
    steam_cost_per_1000lb: float = 0.0

    # Equipment-specific parameters (populated from other calculators)
    electrode_power_consumption_kw: float = 0.0
    baghouse_operating_cost_per_ton: float = 0.0
    scrubber_operating_cost_per_ton: float = 0.0
    glass_raw_material_cost_per_ton: float = 0.0


@dataclass
class FinancialResults:
    """Results of financial analysis"""

    # Annual Volumes
    annual_feedstock_tons: float = 0.0
    annual_product_tons: float = 0.0
    annual_byproduct_tons: float = 0.0

    # Annual Revenue
    product_revenue: float = 0.0
    byproduct_revenue: float = 0.0
    total_revenue: float = 0.0

    # Annual Operating Expenses
    feedstock_costs: float = 0.0
    variable_labor_costs: float = 0.0
    utilities_costs: float = 0.0
    maintenance_costs: float = 0.0
    consumables_costs: float = 0.0
    total_variable_costs: float = 0.0

    # Annual Fixed Costs
    fixed_labor_costs: float = 0.0
    insurance_costs: float = 0.0
    property_tax_costs: float = 0.0
    admin_overhead_costs: float = 0.0
    total_fixed_costs: float = 0.0

    # Financial Metrics
    gross_margin: float = 0.0
    ebitda: float = 0.0
    depreciation: float = 0.0
    ebit: float = 0.0
    interest_expense: float = 0.0
    ebt: float = 0.0
    taxes: float = 0.0
    net_income: float = 0.0

    # Per-ton metrics
    revenue_per_ton: float = 0.0
    variable_cost_per_ton: float = 0.0
    total_cost_per_ton: float = 0.0
    margin_per_ton: float = 0.0

    # Return metrics
    roe: float = 0.0  # Return on Equity
    roa: float = 0.0  # Return on Assets
    payback_period_years: float = 0.0


class FinancialModelCalculator:
    """Core financial model calculation engine"""

    def __init__(self) -> None:
        """Initialize the class."""
        self.parameters = FinancialParameters()
        self.results = FinancialResults()
        self.yearly_projections: list[dict[str, Any]] = []

    def _calculate_volumes_and_revenues(
        self,
        parameters: FinancialParameters,
        results: FinancialResults,
    ) -> None:
        """Compute annual volumes and revenue line items."""
        results.annual_feedstock_tons = (
            parameters.plant_capacity_tpd
            * parameters.operating_days_per_year
            * parameters.capacity_utilization
        )
        results.annual_product_tons = results.annual_feedstock_tons * 0.85
        results.annual_byproduct_tons = (
            results.annual_feedstock_tons * parameters.byproduct_yield_factor
        )

        results.product_revenue = (
            results.annual_product_tons * parameters.product_price_per_ton
        )
        results.byproduct_revenue = (
            results.annual_byproduct_tons * parameters.byproduct_revenue_per_ton
        )
        results.total_revenue = results.product_revenue + results.byproduct_revenue

    def _calculate_operating_costs(
        self,
        parameters: FinancialParameters,
        results: FinancialResults,
    ) -> None:
        """Compute variable and fixed operating costs."""
        tons = results.annual_feedstock_tons

        results.feedstock_costs = tons * parameters.feedstock_cost_per_ton
        results.variable_labor_costs = tons * parameters.labor_cost_per_ton
        results.utilities_costs = tons * parameters.utilities_cost_per_ton
        results.maintenance_costs = tons * parameters.maintenance_cost_per_ton
        results.consumables_costs = tons * parameters.consumables_cost_per_ton

        equipment_costs = tons * (
            parameters.baghouse_operating_cost_per_ton
            + parameters.scrubber_operating_cost_per_ton
            + parameters.glass_raw_material_cost_per_ton
        )

        results.total_variable_costs = (
            results.feedstock_costs
            + results.variable_labor_costs
            + results.utilities_costs
            + results.maintenance_costs
            + results.consumables_costs
            + equipment_costs
        )

        results.fixed_labor_costs = parameters.fixed_labor_cost_annual
        results.insurance_costs = parameters.insurance_annual
        results.property_tax_costs = parameters.property_tax_annual
        results.admin_overhead_costs = parameters.admin_overhead_annual
        results.total_fixed_costs = (
            results.fixed_labor_costs
            + results.insurance_costs
            + results.property_tax_costs
            + results.admin_overhead_costs
        )

    def _calculate_income_statement(
        self,
        parameters: FinancialParameters,
        results: FinancialResults,
    ) -> None:
        """Compute financial metrics from gross margin through net income."""
        results.gross_margin = results.total_revenue - results.total_variable_costs
        results.ebitda = results.gross_margin - results.total_fixed_costs
        results.depreciation = parameters.total_capital_investment / max(
            parameters.depreciation_years,
            1,
        )
        results.ebit = results.ebitda - results.depreciation

        debt_amount = parameters.total_capital_investment * parameters.debt_ratio
        results.interest_expense = debt_amount * parameters.interest_rate
        results.ebt = results.ebit - results.interest_expense
        results.taxes = max(0, results.ebt * parameters.tax_rate)
        results.net_income = results.ebt - results.taxes

    def _calculate_unit_and_return_metrics(
        self,
        parameters: FinancialParameters,
        results: FinancialResults,
    ) -> None:
        """Compute per-ton unit economics and return metrics."""
        if results.annual_feedstock_tons > 0:
            results.revenue_per_ton = (
                results.total_revenue / results.annual_feedstock_tons
            )
            results.variable_cost_per_ton = (
                results.total_variable_costs / results.annual_feedstock_tons
            )
            results.total_cost_per_ton = (
                results.total_variable_costs + results.total_fixed_costs
            ) / results.annual_feedstock_tons
            results.margin_per_ton = (
                results.revenue_per_ton - results.total_cost_per_ton
            )
        else:
            results.revenue_per_ton = 0.0
            results.variable_cost_per_ton = 0.0
            results.total_cost_per_ton = 0.0
            results.margin_per_ton = 0.0

        if parameters.debt_ratio >= 1.0 and parameters.total_capital_investment > 0:
            raise ValueError(
                f"debt_ratio must be < 1.0 when capital is invested "
                f"(equity would be zero or negative), got {parameters.debt_ratio}"
            )
        equity = parameters.total_capital_investment * (1 - parameters.debt_ratio)
        if equity > 0:
            results.roe = results.net_income / equity
        else:
            results.roe = 0.0
        if parameters.total_capital_investment > 0:
            results.roa = results.net_income / parameters.total_capital_investment
        else:
            results.roa = 0.0
        if results.net_income > 0 and (results.net_income + results.depreciation) > 0:
            results.payback_period_years = parameters.total_capital_investment / (
                results.net_income + results.depreciation
            )
        else:
            results.payback_period_years = 0.0

    def calculate_financial_model(
        self,
        parameters: FinancialParameters,
    ) -> FinancialResults:
        """Calculate comprehensive financial model."""
        # DbC preconditions
        assert (
            parameters.total_capital_investment >= 0
        ), f"Capital investment must be non-negative, got {parameters.total_capital_investment}"
        assert (
            parameters.operating_days_per_year >= 0
        ), f"Operating days must be non-negative, got {parameters.operating_days_per_year}"

        self.parameters = parameters
        results = FinancialResults()

        self._calculate_volumes_and_revenues(parameters, results)
        self._calculate_operating_costs(parameters, results)
        self._calculate_income_statement(parameters, results)
        self._calculate_unit_and_return_metrics(parameters, results)

        self.results = results
        return results

    def generate_yearly_projections(self, years: int = 10) -> list[dict[str, Any]]:
        """Generate multi-year financial projections"""
        projections = []
        base_params = self.parameters

        for year in range(1, years + 1):
            # Apply growth/inflation assumptions
            year_params = FinancialParameters()
            year_params.__dict__.update(base_params.__dict__)

            # Price escalation (2% annually)
            year_params.product_price_per_ton *= 1.02**year
            year_params.byproduct_revenue_per_ton *= 1.02**year

            # Cost inflation (3% annually for most costs)
            year_params.feedstock_cost_per_ton *= 1.03**year
            year_params.labor_cost_per_ton *= 1.03**year
            year_params.utilities_cost_per_ton *= 1.025**year  # Lower utility inflation
            year_params.fixed_labor_cost_annual *= 1.03**year

            # Calculate year results
            year_results = self.calculate_financial_model(year_params)

            projection = {
                "year": year,
                "total_revenue": year_results.total_revenue,
                "total_costs": year_results.total_variable_costs
                + year_results.total_fixed_costs,
                "ebitda": year_results.ebitda,
                "net_income": year_results.net_income,
                "cash_flow": year_results.net_income + year_results.depreciation,
                "cumulative_cash_flow": 0,  # Will be calculated after loop
            }
            projections.append(projection)

        # Calculate cumulative cash flow
        cumulative = 0.0
        for proj in projections:
            cumulative += proj["cash_flow"]
            proj["cumulative_cash_flow"] = cumulative

        self.yearly_projections = projections
        return projections
