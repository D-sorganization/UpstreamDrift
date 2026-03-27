"""Pydantic contracts for WGS reactor calculator endpoints.  See issue #613."""

from __future__ import annotations

from pydantic import BaseModel, Field


class WGSReactorRequest(BaseModel):
    """Request model for WGS reactor equilibrium calculation."""

    inlet_composition: dict[str, float] = Field(
        ...,
        description="Inlet gas composition (mol%).  Keys: CO, H2, CO2, H2O.",
    )
    temperature_k: float = Field(..., gt=0, description="Reactor temperature [K]")
    pressure_bar: float = Field(..., gt=0, description="Reactor pressure [bar]")
    steam_ratio: float = Field(default=2.0, gt=0, description="Steam-to-CO molar ratio")
    feed_rate_kmol_hr: float = Field(
        default=0.0,
        ge=0,
        description="Feed rate [kmol/hr] (0 = skip reactor sizing)",
    )
    catalyst_type: str = Field(
        default="HTS", description="Catalyst type label for sizing lookup"
    )


class WGSEquilibriumOut(BaseModel):
    """Equilibrium results from the WGS reaction."""

    conversion_pct: float = Field(description="CO conversion [%]")
    composition: dict[str, float] = Field(description="Outlet composition [mol%]")
    h2_co_ratio: float = Field(description="H2/CO molar ratio at equilibrium")
    equilibrium_constant: float = Field(description="K_eq at reactor temperature")
    heat_released_kj: float = Field(description="Heat released [kJ/mol CO in]")


class WGSSizingOut(BaseModel):
    """Reactor sizing results."""

    reactor_volume_m3: float = Field(description="Reactor volume [m3]")
    catalyst_volume_m3: float = Field(description="Catalyst volume [m3]")
    diameter_m: float = Field(description="Reactor diameter [m]")
    length_m: float = Field(description="Reactor length [m]")
    heat_duty_kw: float = Field(description="Heat duty [kW]")
    ghsv: float = Field(description="Gas hourly space velocity [1/h]")


class WGSReactorResponse(BaseModel):
    """Response model for WGS reactor calculation."""

    equilibrium: WGSEquilibriumOut
    sizing: WGSSizingOut | None = Field(
        default=None,
        description="Reactor sizing (None when feed_rate is 0)",
    )
