"""Pydantic contracts for acid gas dewpoint calculator endpoints.  See issue #613."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AcidGasDewpointRequest(BaseModel):
    """Request model for acid gas dewpoint calculation."""

    temperature_c: float = Field(..., description="System temperature [degC]")
    pressure_bar: float = Field(..., gt=0, description="System pressure [bar]")
    h2o_fraction: float = Field(
        default=0.0, ge=0, le=1, description="H2O mole fraction"
    )
    hf_fraction: float = Field(default=0.0, ge=0, le=1, description="HF mole fraction")
    hcl_fraction: float = Field(
        default=0.0, ge=0, le=1, description="HCl mole fraction"
    )
    h2s_fraction: float = Field(
        default=0.0, ge=0, le=1, description="H2S mole fraction"
    )
    method: str = Field(
        default="antoine",
        description="Vapor-pressure method: 'antoine', 'extended_antoine'",
    )


class DewpointComponentOut(BaseModel):
    """Individual component dewpoint result."""

    dewpoint_c: float | None = Field(description="Dewpoint temperature [degC] or null")
    vapor_pressure_pa: float = Field(description="Vapor pressure at system T [Pa]")
    partial_pressure_pa: float = Field(description="Partial pressure [Pa]")


class AcidGasDewpointResponse(BaseModel):
    """Response model for acid gas dewpoint calculation."""

    overall_dewpoint_c: float | None = Field(
        description="Overall dewpoint [degC] (highest among components)"
    )
    limiting_component: str
    dewpoint_margin_c: float | None = Field(description="Margin above dewpoint [degC]")
    condensation_risk: str
    components: dict[str, DewpointComponentOut]
    warnings: list[str] = Field(default_factory=list)
    calculation_method: str
