"""Pydantic contracts for ODE solver endpoints.  See issue #608."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ODESolverRequest(BaseModel):
    """Request model for ODE system solving."""

    derivatives: dict[str, str] = Field(
        ...,
        description="Variable name to derivative expression mapping, e.g. {'y': '-k*y'}",
    )
    parameters: dict[str, float] = Field(
        default_factory=dict,
        description="Parameter name to value mapping, e.g. {'k': 0.1}",
    )
    initial_conditions: dict[str, float] = Field(
        ...,
        description="Variable name to initial value mapping, e.g. {'y': 100}",
    )
    t_start: float = Field(default=0.0, ge=0, description="Start time")
    t_end: float = Field(default=20.0, gt=0, description="End time")
    num_points: int = Field(default=100, ge=10, le=10000, description="Number of output points")


class ODEVariableSummary(BaseModel):
    """Summary statistics for a single variable."""

    name: str
    initial_value: float
    final_value: float
    min_value: float
    max_value: float


class ODESolverResponse(BaseModel):
    """Response model for ODE solver."""

    times: list[float]
    solutions: dict[str, list[float]] = Field(description="Variable name to list of values mapping")
    variable_summaries: list[ODEVariableSummary]
    success: bool = True
    message: str = "Solution computed successfully"
