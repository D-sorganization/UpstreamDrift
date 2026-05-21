# ruff: noqa: E501
"""Enhanced Pydantic models for pressure drop calculator with standardized responses.

This module defines the v2 API contracts for the pressure drop calculator,
using the standardized response wrapper for consistent error handling and
request tracking.

See issue #2411 - API Standardization Foundation.
"""

from __future__ import annotations

from calc_backend.api import StandardResponse
from pydantic import BaseModel, Field


class PressureDropRequestV2(BaseModel):
    """Request model for pressure drop calculation (v2 with enhanced docs).

    All parameters use SI units and follow consistent naming conventions.
    Validation occurs at the Pydantic level before reaching the calculator.
    """

    pipe_diameter_m: float = Field(
        ...,
        gt=0,
        description="Pipe inner diameter [m]. Must be positive.",
    )
    pipe_length_m: float = Field(
        ...,
        gt=0,
        description="Pipe length [m]. Must be positive.",
    )
    roughness_m: float = Field(
        default=0.000045,
        ge=0,
        description=(
            "Pipe wall roughness [m]. Defaults to 0.000045 m for carbon steel. "
            "Must be non-negative."
        ),
    )
    flow_rate_kg_s: float = Field(
        ...,
        gt=0,
        description="Mass flow rate [kg/s]. Must be positive.",
    )
    temperature_k: float = Field(
        ...,
        gt=0,
        description="Gas temperature [K]. Must be positive.",
    )
    pressure_pa: float = Field(
        ...,
        gt=0,
        description="Gas pressure [Pa]. Must be positive.",
    )
    molecular_weight_kg_mol: float = Field(
        ...,
        gt=0,
        description="Molecular weight [kg/mol]. Must be positive.",
    )
    viscosity_pa_s: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Dynamic viscosity [Pa·s]. Optional. "
            "If provided, used directly. "
            "If omitted, Sutherland air approximation is used (air only). "
            "Must be positive if provided."
        ),
    )

    class Config:
        """Model configuration."""

        json_schema_extra = {
            "example": {
                "pipe_diameter_m": 0.1,
                "pipe_length_m": 100.0,
                "roughness_m": 0.000045,
                "flow_rate_kg_s": 1.0,
                "temperature_k": 300.0,
                "pressure_pa": 101325.0,
                "molecular_weight_kg_mol": 0.029,
                "viscosity_pa_s": None,
            }
        }


class PressureDropDataV2(BaseModel):
    """Calculation results data model for pressure drop endpoint.

    All output values use SI units. This model represents only the
    calculated data, not the full response (see PressureDropResponseV2).
    """

    pressure_drop_pa: float = Field(
        description="Total pressure drop [Pa]. Always non-negative."
    )
    reynolds_number: float = Field(
        description="Reynolds number (dimensionless). Characterizes flow regime."
    )
    friction_factor: float = Field(
        description="Darcy friction factor (dimensionless). Always positive."
    )
    velocity_m_s: float = Field(description="Gas velocity [m/s]. Always non-negative.")
    flow_regime: str = Field(
        description="Flow regime classification: 'Laminar', 'Transitional', or 'Turbulent'."  # noqa: E501
    )
    density_kg_m3: float = Field(
        description="Gas density [kg/m³] at given P and T. Always positive."
    )
    viscosity_pa_s: float = Field(
        description="Gas viscosity [Pa·s] at given temperature. Always positive."
    )

    class Config:
        """Model configuration."""

        json_schema_extra = {
            "example": {
                "pressure_drop_pa": 1023.4,
                "reynolds_number": 50000.0,
                "friction_factor": 0.015,
                "velocity_m_s": 45.2,
                "flow_regime": "Turbulent",
                "density_kg_m3": 1.225,
                "viscosity_pa_s": 1.8e-5,
            }
        }


class PressureDropResponseV2(StandardResponse[PressureDropDataV2]):
    """Standardized response for pressure drop calculation endpoint.

    Wraps calculation results or error details in a consistent envelope with
    request tracking metadata. This is the response model for all pressure
    drop v2 API endpoints.

    Attributes:
        status: "success" if calculation succeeded, "error" otherwise.
        data: PressureDropDataV2 object on success, None on error.
        error: ErrorDetail on error, None on success.
        metadata: ResponseMetadata with request_id, processing_time_ms, etc.

    Examples:
        Success response:
        {
            "status": "success",
            "data": {
                "pressure_drop_pa": 1023.4,
                "reynolds_number": 50000.0,
                "friction_factor": 0.015,
                "velocity_m_s": 45.2,
                "flow_regime": "Turbulent",
                "density_kg_m3": 1.225,
                "viscosity_pa_s": 1.8e-5
            },
            "error": null,
            "metadata": {
                "request_id": "550e8400-e29b-41d4-a716-446655440000",
                "processing_time_ms": 125.5,
                "timestamp_utc": "2026-04-30T12:34:56Z",
                "api_version": "v1"
            }
        }

        Error response (validation failure):
        {
            "status": "error",
            "data": null,
            "error": {
                "code": "CONSTRAINT_VIOLATION",
                "message": "pipe_diameter_m must be positive",
                "details": "Field: pipe_diameter_m, Value: -0.1"
            },
            "metadata": {
                "request_id": "550e8400-e29b-41d4-a716-446655440001",
                "processing_time_ms": 5.2,
                "timestamp_utc": "2026-04-30T12:34:56Z",
                "api_version": "v1"
            }
        }
    """


# Legacy aliases for backward compatibility
PressureDropRequest = PressureDropRequestV2
PressureDropResponse = PressureDropResponseV2
