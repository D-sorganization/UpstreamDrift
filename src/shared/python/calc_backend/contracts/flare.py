"""Pydantic contracts for flare calculator endpoints.  See issue #613."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FlareRequest(BaseModel):
    """Request model for flare size calculation."""

    total_flow_kg_hr: float = Field(
        ..., gt=0, description="Total gas flow rate [kg/hr]"
    )
    gas_composition: dict[str, float] = Field(
        ...,
        description=(
            "Gas composition as mol%.  Keys must be from: "
            "H2, CO, CH4, C2H6, C3H8, C4H10, H2S, N2, CO2, H2O"
        ),
    )
    temperature_k: float = Field(..., gt=0, description="Gas temperature [K]")
    pressure_bar: float = Field(..., gt=0, description="Gas pressure [bar]")


class FlareDesignOut(BaseModel):
    """Calculated flare design parameters."""

    height_m: float = Field(description="Flare stack height [m]")
    diameter_m: float = Field(description="Flare tip diameter [m]")
    exit_velocity_m_s: float = Field(description="Gas exit velocity [m/s]")
    heat_release_kw: float = Field(description="Total heat release [kW]")
    radiation_intensity_kw_m2: float = Field(
        description="Design ground-level radiation intensity [kW/m2]"
    )


class RadiationZonesOut(BaseModel):
    """Radiation zone distances around the flare."""

    lethal_m: float = Field(description="Lethal zone distance (37.5 kW/m2) [m]")
    damage_m: float = Field(description="Damage zone distance (12.5 kW/m2) [m]")
    safe_m: float = Field(description="Safe zone distance (1.6 kW/m2) [m]")
    comfort_m: float = Field(description="Comfort zone distance (0.5 kW/m2) [m]")


class FlareResponse(BaseModel):
    """Response model for flare calculations."""

    design: FlareDesignOut
    radiation_zones: RadiationZonesOut
    combustion_efficiency: float = Field(
        description="Estimated combustion efficiency (0-1)"
    )
