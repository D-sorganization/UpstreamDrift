"""Pydantic contracts for thermal profile predictor endpoints.  See issue #608."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ThermalProfileRequest(BaseModel):
    """Request model for thermal profile prediction."""

    initial_temp_c: float = Field(default=25.0, description="Initial temperature [C]")
    ambient_temp_c: float = Field(default=25.0, description="Ambient temperature [C]")
    thermal_mass_j_per_k: float = Field(..., gt=0, description="Thermal mass [J/K]")
    heat_loss_coeff_w_per_k: float = Field(
        ..., ge=0, description="Heat loss coefficient [W/K]"
    )
    power_w: float = Field(default=5000.0, ge=0, description="Power input [W]")
    power_profile: str = Field(
        default="constant",
        description="Power profile type: constant, linear_ramp, step",
    )
    ramp_rate_w_per_s: float = Field(
        default=0.0, ge=0, description="Ramp rate for linear profile [W/s]"
    )
    step_time_s: float = Field(
        default=1800.0, ge=0, description="Step time for step profile [s]"
    )
    t_start_s: float = Field(default=0.0, ge=0, description="Start time [s]")
    t_end_s: float = Field(default=3600.0, gt=0, description="End time [s]")
    num_points: int = Field(
        default=100, ge=10, le=10000, description="Number of output points"
    )


class ThermalProfileDataPoint(BaseModel):
    """Single data point in the temperature profile."""

    time_s: float
    temperature_c: float
    power_w: float
    heat_loss_w: float


class ThermalProfileResponse(BaseModel):
    """Response model for thermal profile prediction."""

    data: list[ThermalProfileDataPoint]
    final_temp_c: float
    max_temp_c: float
    min_temp_c: float
    temp_change_c: float
    steady_state_temp_c: float | None = None
    time_constant_s: float | None = None
