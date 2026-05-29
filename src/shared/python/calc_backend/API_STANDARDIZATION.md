# API Standardization (Issue #2411)

This document describes the standardized response format and models for all calculator backend APIs.

## Overview

All calculator backend endpoints now return a unified response format with the following structure:

- **status**: "success" or "error"
- **data**: Response payload (only present in success responses)
- **error**: Error details (only present in error responses)
- **metadata**: Request tracking information

This ensures consistent client behavior across all endpoints and improves debugging through request tracking.

## StandardResponse Class

The `StandardResponse` class (in `sidekick.api`) provides a reusable wrapper for all API responses.

### Imports

```python
from sidekick.api import (
    StandardResponse,
    ErrorDetail,
    ErrorCode,
    ResponseMetadata,
)
```

### Factory Methods

#### Success Response

```python
response = StandardResponse.success(
    data={"result": 123.4, "status": "ok"},
    processing_time_ms=125.0,
    request_id="req-12345"  # optional, auto-generated if not provided
)
```

#### Error Response

```python
error = ErrorDetail(
    code=ErrorCode.INVALID_INPUT,
    message="Validation failed",
    details={"field": "pipe_diameter_m", "reason": "must be > 0"},
)
response = StandardResponse.error(
    error=error,
    processing_time_ms=50.0
)
```

## Pressure Drop API Example

The pressure drop calculator endpoint demonstrates the standardized API pattern.

### Endpoint: POST /api/calc/pressure-drop

#### Request

```json
{
  "pipe_diameter_m": 0.1,
  "pipe_length_m": 100.0,
  "roughness_m": 0.000045,
  "flow_rate_kg_s": 5.0,
  "temperature_k": 300.0,
  "pressure_pa": 101325.0,
  "molecular_weight_kg_mol": 28.97,
  "viscosity_pa_s": null
}
```

**Request Model**: `PressureDropRequest` (defined in `calc_backend.models.pressure_drop`)

Validation:

- All dimensional parameters must be > 0
- Roughness must be >= 0 (allows smooth pipes)
- Viscosity is optional; if omitted, Sutherland approximation is used

#### Success Response (HTTP 200)

```json
{
  "status": "success",
  "data": {
    "pressure_drop_pa": 1023.4,
    "reynolds_number": 50000.0,
    "friction_factor": 0.025,
    "velocity_m_s": 45.2,
    "flow_regime": "Turbulent",
    "density_kg_m3": 1.177,
    "viscosity_pa_s": 1.86e-5
  },
  "error": null,
  "metadata": {
    "request_id": "550e8400-e29b-41d4-a716-446655440000",
    "processing_time_ms": 125.0,
    "api_version": "1.0.0"
  }
}
```

**Response Model**: `PressureDropResponse` (defined in `calc_backend.models.pressure_drop`)

#### Error Response: Invalid Input (HTTP 422)

```json
{
  "status": "error",
  "data": null,
  "error": {
    "code": "INVALID_INPUT",
    "message": "Pressure drop calculation failed: pipe_diameter_m must be > 0",
    "details": {
      "exception_type": "ValueError",
      "exception_message": "pipe_diameter_m must be > 0"
    },
    "request_id": "550e8400-e29b-41d4-a716-446655440001"
  },
  "metadata": {
    "request_id": "550e8400-e29b-41d4-a716-446655440001",
    "processing_time_ms": 15.0,
    "api_version": "1.0.0"
  }
}
```

## Error Codes

The `ErrorCode` enum defines standard error codes used across all APIs:

| Code                    | Meaning                                                   |
| ----------------------- | --------------------------------------------------------- |
| `INVALID_INPUT`         | Input validation failed (wrong type, out-of-range values) |
| `NOT_FOUND`             | Requested resource not found                              |
| `SERVER_ERROR`          | Unexpected server-side error                              |
| `UNSUPPORTED_OPERATION` | Operation not supported (e.g., invalid enum value)        |
| `TIMEOUT`               | Request processing timed out                              |
| `CONSTRAINT_VIOLATION`  | Physical or logical constraint violated                   |

## Integration with FastAPI Endpoints

### Router Pattern

```python
from fastapi import APIRouter, HTTPException
from sidekick.api import StandardResponse, ErrorDetail, ErrorCode
from calc_backend.models.pressure_drop import PressureDropRequest, PressureDropResponse

router = APIRouter(prefix="/api/calc/pressure-drop", tags=["pressure-drop"])

@router.post("")
def calculate_pressure_drop(request: PressureDropRequest) -> dict[str, Any]:
    """Calculate pressure drop."""
    start_time = time.perf_counter()

    try:
        result = _calculator.calculate_pressure_drop(
            pipe_diameter_m=request.pipe_diameter_m,
            # ... other params
        )
    except (ValueError, ZeroDivisionError) as exc:
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        error = ErrorDetail(
            code=ErrorCode.INVALID_INPUT,
            message=f"Calculation failed: {str(exc)}",
            details={"exception_type": type(exc).__name__},
        )
        response = StandardResponse.error(
            error=error,
            processing_time_ms=processing_time_ms,
        )
        raise HTTPException(status_code=422, detail=response.to_dict())

    processing_time_ms = (time.perf_counter() - start_time) * 1000
    response_data = PressureDropResponse(
        pressure_drop_pa=result.pressure_drop_pa,
        # ... other fields
    ).model_dump()

    response = StandardResponse.success(
        data=response_data,
        processing_time_ms=processing_time_ms,
    )
    return response.to_dict()
```

## Key Features

### Request Tracking

Every response includes a `request_id` in metadata, enabling end-to-end request tracing:

```json
"metadata": {
    "request_id": "550e8400-e29b-41d4-a716-446655440000",
    "processing_time_ms": 125.0,
    "api_version": "1.0.0"
}
```

Client can:

- Log request_id with request/response
- Correlate server logs with client requests
- Measure endpoint performance

### Type Safety

Pydantic models ensure type safety and validation:

```python
# Valid request
request = PressureDropRequest(
    pipe_diameter_m=0.1,
    pipe_length_m=100.0,
    ...
)

# Invalid request raises ValidationError
request = PressureDropRequest(
    pipe_diameter_m=-0.1,  # ValueError: must be > 0
    ...
)
```

### JSON Schema

Auto-generated JSON schemas are available via FastAPI's `/docs` and `/redoc`:

```python
schema = PressureDropRequest.model_json_schema()
# {
#   "properties": {
#     "pipe_diameter_m": {
#       "type": "number",
#       "gt": 0,
#       "description": "Pipe inner diameter [m]"
#     },
#     ...
#   }
# }
```

## Backward Compatibility

The old `PressureDropResponse` model from `calc_backend.contracts.pressure_drop` is deprecated.
The new models in `calc_backend.models.pressure_drop` provide the same fields with enhanced
documentation and JSON schema examples.

To migrate existing clients:

1. Update endpoints to use new StandardResponse format
2. Client code extracts data from response["data"]
3. Error handling uses response["error"]["code"] instead of HTTP 422 without details

## Testing

Comprehensive tests are provided:

- `src/shared/python/sidekick/tests/test_standard_response.py` - StandardResponse class
- `tests/calc_backend/test_pressure_drop_models.py` - PressureDropRequest/Response validation

Run tests:

```bash
python3 -m pytest src/shared/python/sidekick/tests/test_standard_response.py -v
python3 -m pytest tests/calc_backend/test_pressure_drop_models.py -v
```

## References

- Issue #2411: API Standardization Foundation
- Issue #613: Calc Backend
- StandardResponse: `src/shared/python/sidekick/api/standard_response.py`
- Models: `src/shared/python/calc_backend/models/pressure_drop.py`
- Router: `src/shared/python/calc_backend/routers/pressure_drop.py`
