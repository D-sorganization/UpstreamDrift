"""Pydantic contracts for flow rate converter endpoints.  See issue #608."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FlowRateConvertRequest(BaseModel):
    """Request model for flow rate unit conversion."""

    value: float = Field(..., description="Numeric value to convert")
    from_unit: str = Field(..., description="Source unit (e.g. 'kg/s', 'lb/h')")
    to_unit: str = Field(..., description="Target unit")
    category: str = Field(
        default="mass",
        description="Flow category: 'mass', 'molar', or 'volumetric'",
    )


class FlowRateConvertResponse(BaseModel):
    """Response model for flow rate conversion."""

    result: float = Field(description="Converted value")
    from_unit: str
    to_unit: str
    category: str
