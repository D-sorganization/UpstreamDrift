"""Pydantic contracts for pressure drop calculator endpoints.  See issue #613."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PressureDropRequest(BaseModel):
    """Request model for pressure drop calculation (legacy simple API)."""

    pipe_diameter_m: float = Field(..., gt=0, description="Pipe inner diameter [m]")
    pipe_length_m: float = Field(..., gt=0, description="Pipe length [m]")
    roughness_m: float = Field(default=0.000045, ge=0, description="Pipe wall roughness [m]")
    flow_rate_kg_s: float = Field(..., gt=0, description="Mass flow rate [kg/s]")
    temperature_k: float = Field(..., gt=0, description="Gas temperature [K]")
    pressure_pa: float = Field(..., gt=0, description="Gas pressure [Pa]")
    molecular_weight_kg_mol: float = Field(..., gt=0, description="Molecular weight [kg/mol]")


class PressureDropResponse(BaseModel):
    """Response model for pressure drop calculation."""

    pressure_drop_pa: float = Field(description="Total pressure drop [Pa]")
    reynolds_number: float
    friction_factor: float
    velocity_m_s: float = Field(description="Gas velocity [m/s]")
    flow_regime: str = Field(description="Laminar / Transitional / Turbulent")
    density_kg_m3: float
    viscosity_pa_s: float
