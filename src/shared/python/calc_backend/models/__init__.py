"""Pydantic models for standardized API request/response schemas.

This package provides validated, documented models for all calculator APIs,
ensuring consistent input validation and output format across all endpoints.
"""

from __future__ import annotations

from .pressure_drop import (
    PressureDropRequest,
    PressureDropResponse,
)

__all__ = [
    "PressureDropRequest",
    "PressureDropResponse",
]
