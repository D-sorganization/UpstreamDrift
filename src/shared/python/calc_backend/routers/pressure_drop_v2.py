"""Enhanced pressure drop calculator router with standardized response format.

This router implements v2 of the pressure drop API with:
- StandardResponse wrapper for all responses
- Proper error handling with ErrorCode enums
- Request tracking via metadata
- Enhanced Pydantic models with documentation

See issue #2411 - API Standardization Foundation.
"""

from __future__ import annotations

import logging
from typing import Any

from calc_backend.api import ErrorCode, StandardResponseBuilder
from calc_backend.contracts.pressure_drop_v2 import (
    PressureDropDataV2,
    PressureDropRequestV2,
    PressureDropResponseV2,
)
from fastapi import APIRouter
from upstream_drift_tools.process_calculators.pressure_drop_calculator import (
    PressureDropCalculator,
    PressureDropResult,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/calc/pressure-drop-v2", tags=["pressure-drop-v2"])

_calculator = PressureDropCalculator()


@router.post("", response_model=PressureDropResponseV2)
def calculate_pressure_drop(
    request: PressureDropRequestV2,
) -> PressureDropResponseV2:
    """Calculate pressure drop using Darcy-Weisbach equation.

    This v2 endpoint wraps results in a StandardResponse with proper error
    handling, request tracking, and enhanced documentation.

    Args:
        request: PressureDropRequestV2 with all SI-unit pipe and fluid parameters.

    Returns:
        PressureDropResponseV2 containing:
        - status: "success" or "error"
        - data: Pressure drop calculation results (on success)
        - error: Error details with code and message (on error)
        - metadata: Request ID, processing time, and timestamp

    Raises:
        None. All errors returned as error responses with proper codes.

    Examples:
        Request:
        {
            "pipe_diameter_m": 0.1,
            "pipe_length_m": 100.0,
            "flow_rate_kg_s": 1.0,
            "temperature_k": 300.0,
            "pressure_pa": 101325.0,
            "molecular_weight_kg_mol": 0.029
        }

        Success response:
        {
            "status": "success",
            "data": {
                "pressure_drop_pa": 1023.4,
                "reynolds_number": 50000.0,
                ...
            },
            "error": null,
            "metadata": {...}
        }

        Error response (invalid input):
        {
            "status": "error",
            "data": null,
            "error": {
                "code": "CONSTRAINT_VIOLATION",
                "message": "pipe_diameter_m must be positive"
            },
            "metadata": {...}
        }
    """
    builder = StandardResponseBuilder()

    try:
        result: PressureDropResult = _calculator.calculate_pressure_drop(
            pipe_diameter_m=request.pipe_diameter_m,
            pipe_length_m=request.pipe_length_m,
            roughness_m=request.roughness_m,
            flow_rate_kg_s=request.flow_rate_kg_s,
            temperature_k=request.temperature_k,
            pressure_pa=request.pressure_pa,
            molecular_weight_kg_mol=request.molecular_weight_kg_mol,
            viscosity_pa_s=request.viscosity_pa_s,
        )
    except ValueError as exc:
        logger.warning("Calculation error in pressure drop: %s", str(exc))
        return builder.error(
            code=ErrorCode.CALCULATION_ERROR,
            message="Pressure drop calculation failed",
            details=str(exc),
        )
    except (ZeroDivisionError, OverflowError) as exc:
        logger.error("Arithmetic error in pressure drop: %s", str(exc))
        return builder.error(
            code=ErrorCode.CALCULATION_ERROR,
            message="Arithmetic error during calculation",
            details=str(exc),
        )
    except TypeError as exc:
        logger.error("Type error in pressure drop: %s", str(exc))
        return builder.error(
            code=ErrorCode.INVALID_INPUT,
            message="Invalid input types",
            details=str(exc),
        )
    except Exception as exc:
        logger.error("Unexpected error in pressure drop: %s", str(exc), exc_info=True)
        return builder.error(
            code=ErrorCode.SERVER_ERROR,
            message="Unexpected server error",
            details=str(exc),
        )

    # Build success response
    data = PressureDropDataV2(
        pressure_drop_pa=result.pressure_drop_pa,
        reynolds_number=result.reynolds_number,
        friction_factor=result.friction_factor,
        velocity_m_s=result.velocity,
        flow_regime=result.flow_regime,
        density_kg_m3=result.density,
        viscosity_pa_s=result.viscosity,
    )
    return builder.success(data=data)


@router.post("/validate", response_model=dict[str, Any])
def validate_request(request: PressureDropRequestV2) -> dict[str, Any]:
    """Validate a pressure drop calculation request.

    This endpoint checks if a request is valid without performing the
    calculation. Useful for client-side validation and debugging.

    Args:
        request: PressureDropRequestV2 to validate.

    Returns:
        Validation result with status and any constraint violations.

    Examples:
        Request:
        {
            "pipe_diameter_m": 0.1,
            ...
        }

        Response:
        {
            "valid": true,
            "errors": []
        }
    """
    builder = StandardResponseBuilder()

    # If we got here, Pydantic validation passed
    return builder.success(
        data={
            "valid": True,
            "errors": [],
            "summary": "Request is valid. Ready for calculation.",
        }
    )
