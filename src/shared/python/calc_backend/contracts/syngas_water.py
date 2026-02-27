"""Pydantic contracts for syngas water calculator endpoints.  See issue #608."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SyngasWaterRequest(BaseModel):
    """Request model for syngas water content calculation."""

    temperature_c: float = Field(..., ge=-50, le=400, description="Gas temperature [C]")
    pressure_bar: float = Field(..., gt=0, le=500, description="System pressure [bar]")
    composition_key: str = Field(
        default="typical_syngas",
        description="Composition preset key: typical_syngas, biomass_syngas, coal_syngas, natural_gas_reforming",
    )
    method: str = Field(
        default="auto",
        description="Vapor pressure method: auto, antoine, buck, iapws, magnus",
    )


class WaterContentOut(BaseModel):
    """Calculated water content values."""

    mole_fraction_water: float
    water_content_mg_per_nm3: float
    water_content_ppmv: float
    water_content_g_per_m3: float
    water_content_lb_per_mmscf: float
    vapor_pressure_bar: float
    dew_point_c: float


class CondensationRiskOut(BaseModel):
    """Condensation risk assessment."""

    temperature_margin_c: float
    condensation_risk: str
    recommended_temperature_c: float


class SyngasWaterResponse(BaseModel):
    """Response model for syngas water calculation."""

    water_content: WaterContentOut
    risk_assessment: CondensationRiskOut
