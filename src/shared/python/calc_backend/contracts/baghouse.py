"""Pydantic contracts for baghouse calculator endpoints.  See issue #613."""

from __future__ import annotations

from pydantic import BaseModel, Field


class BaghouseRequest(BaseModel):
    """Request model for baghouse performance calculation."""

    gas_flow_kg_s: float = Field(..., gt=0, description="Gas mass flow rate [kg/s]")
    inlet_temp_k: float = Field(..., gt=0, description="Inlet temperature [K]")
    pressure_pa: float = Field(..., gt=0, description="Gas pressure [Pa]")
    composition: dict[str, float] = Field(..., description="Gas composition as mole fractions")
    solid_carbon_in_kg_hr: float = Field(
        default=0.0, ge=0, description="Solid carbon input rate [kg/hr]"
    )
    ash_in_kg_hr: float = Field(default=0.0, ge=0, description="Ash input rate [kg/hr]")
    carbon_removal_efficiency: float = Field(
        default=0.99, ge=0, le=1, description="Carbon removal efficiency (0-1)"
    )
    ash_removal_efficiency: float = Field(
        default=0.999, ge=0, le=1, description="Ash removal efficiency (0-1)"
    )
    heat_loss_w: float = Field(default=0.0, ge=0, description="Heat loss rate [W]")
    drum_volume_m3: float = Field(default=2.0, gt=0, description="Collection drum volume [m3]")
    solid_density_kg_m3: float = Field(
        default=500.0, gt=0, description="Density of collected solids [kg/m3]"
    )
    bag_area_ft2: float = Field(default=5000.0, gt=0, description="Total bag filter area [ft2]")


class BaghouseResponse(BaseModel):
    """Response model for baghouse calculation."""

    carbon_removed_rate_kg_hr: float
    ash_removed_rate_kg_hr: float
    total_solids_removed_rate_kg_hr: float
    drum_fill_time_hours: float
    drum_fill_time_days: float
    carbon_only_fill_time_hours: float
    ash_only_fill_time_hours: float
    clean_gas_flow_rate_kg_hr: float
    flow_acfm: float
    flow_scfm: float
    air_to_cloth_ratio: float
    outlet_temperature_c: float
    ash_stream_composition: dict[str, float]
    removal_efficiency: dict[str, float]
