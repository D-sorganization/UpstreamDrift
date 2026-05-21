"""Pydantic contracts for scrubber calculator endpoints.  See issue #613."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ScrubberRequest(BaseModel):
    """Request model for packed-bed scrubber column sizing."""

    gas_flow_kg_hr: float = Field(..., gt=0, description="Gas mass flow rate [kg/hr]")
    gas_temperature_k: float = Field(..., gt=0, description="Inlet gas temperature [K]")
    gas_pressure_pa: float = Field(..., gt=0, description="Gas pressure [Pa]")
    gas_molecular_weight: float = Field(
        ..., gt=0, description="Average gas molecular weight [kg/kmol]"
    )
    liquid_flow_kg_hr: float = Field(
        ..., gt=0, description="Liquid (scrubbing solution) flow rate [kg/hr]"
    )
    packing_type: str = Field(
        default="Metal Pall Rings",
        description="Packing type from database",
    )
    percent_of_flood: float = Field(
        default=70.0,
        gt=0,
        le=100,
        description="Design velocity as percent of flooding",
    )
    acid_gas_removed_kg_hr: dict[str, float] = Field(
        default_factory=dict,
        description="Acid gas removal rates [kg/hr] (keys: HCl, SO2, H2S, HF, CO2)",
    )
    caustic_concentration_pct: float = Field(
        default=10.0, gt=0, le=50, description="NaOH solution concentration [wt%]"
    )


class ScrubberResponse(BaseModel):
    """Response model for scrubber calculation."""

    gas_density_kg_m3: float
    flooding_velocity_m_s: float
    design_velocity_m_s: float
    column_diameter_m: float
    column_diameter_ft: float
    cross_section_m2: float
    caustic_requirement: dict[str, float]
