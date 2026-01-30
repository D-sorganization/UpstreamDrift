"""Structured error codes and error handling utilities.

This module provides a standardized error code system for the Golf Modeling Suite API.
Each error has a unique code, category, and detailed message for diagnostics.

Error Code Format: GMS-{CATEGORY}-{NUMBER}
- GMS: Golf Modeling Suite
- CATEGORY: 3-letter category code (ENG=Engine, SIM=Simulation, VID=Video, etc.)
- NUMBER: 3-digit error number

Usage:
    from src.api.utils.error_codes import APIError, ErrorCode, raise_api_error

    # Raise a structured error
    raise_api_error(ErrorCode.ENGINE_NOT_LOADED, engine_type="mujoco")

    # Create an error response
    error = APIError.from_code(ErrorCode.SIMULATION_FAILED, details={"reason": "timeout"})
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from fastapi import HTTPException
from fastapi.responses import JSONResponse

from .tracing import get_request_id, get_trace_context


class ErrorCategory(str, Enum):
    """Error category codes."""

    GENERAL = "GEN"  # General errors
    ENGINE = "ENG"  # Physics engine errors
    SIMULATION = "SIM"  # Simulation errors
    VIDEO = "VID"  # Video processing errors
    ANALYSIS = "ANL"  # Analysis errors
    AUTH = "AUT"  # Authentication errors
    VALIDATION = "VAL"  # Input validation errors
    RESOURCE = "RES"  # Resource errors (not found, etc.)
    SYSTEM = "SYS"  # System/infrastructure errors


class ErrorCode(str, Enum):
    """Standardized error codes for the API.

    Format: GMS-{CATEGORY}-{NUMBER}
    """

    # General Errors (GMS-GEN-XXX)
    INTERNAL_ERROR = "GMS-GEN-001"
    INVALID_REQUEST = "GMS-GEN-002"
    RATE_LIMITED = "GMS-GEN-003"
    SERVICE_UNAVAILABLE = "GMS-GEN-004"

    # Engine Errors (GMS-ENG-XXX)
    ENGINE_NOT_FOUND = "GMS-ENG-001"
    ENGINE_NOT_LOADED = "GMS-ENG-002"
    ENGINE_LOAD_FAILED = "GMS-ENG-003"
    ENGINE_NOT_AVAILABLE = "GMS-ENG-004"
    ENGINE_INITIALIZATION_FAILED = "GMS-ENG-005"
    ENGINE_INVALID_STATE = "GMS-ENG-006"

    # Simulation Errors (GMS-SIM-XXX)
    SIMULATION_FAILED = "GMS-SIM-001"
    SIMULATION_TIMEOUT = "GMS-SIM-002"
    SIMULATION_INVALID_PARAMS = "GMS-SIM-003"
    SIMULATION_MODEL_NOT_FOUND = "GMS-SIM-004"
    SIMULATION_STATE_ERROR = "GMS-SIM-005"
    TASK_NOT_FOUND = "GMS-SIM-006"
    TASK_NOT_COMPLETED = "GMS-SIM-007"

    # Video Errors (GMS-VID-XXX)
    VIDEO_PIPELINE_NOT_INITIALIZED = "GMS-VID-001"
    VIDEO_INVALID_FORMAT = "GMS-VID-002"
    VIDEO_PROCESSING_FAILED = "GMS-VID-003"
    VIDEO_ESTIMATOR_INVALID = "GMS-VID-004"
    VIDEO_CONFIDENCE_INVALID = "GMS-VID-005"

    # Analysis Errors (GMS-ANL-XXX)
    ANALYSIS_SERVICE_NOT_INITIALIZED = "GMS-ANL-001"
    ANALYSIS_FAILED = "GMS-ANL-002"
    ANALYSIS_INVALID_TYPE = "GMS-ANL-003"

    # Auth Errors (GMS-AUT-XXX)
    AUTH_TOKEN_INVALID = "GMS-AUT-001"
    AUTH_TOKEN_EXPIRED = "GMS-AUT-002"
    AUTH_INSUFFICIENT_PERMISSIONS = "GMS-AUT-003"
    AUTH_QUOTA_EXCEEDED = "GMS-AUT-004"
    AUTH_USER_NOT_FOUND = "GMS-AUT-005"

    # Validation Errors (GMS-VAL-XXX)
    VALIDATION_FAILED = "GMS-VAL-001"
    VALIDATION_MISSING_FIELD = "GMS-VAL-002"
    VALIDATION_INVALID_VALUE = "GMS-VAL-003"
    VALIDATION_PATH_INVALID = "GMS-VAL-004"

    # Resource Errors (GMS-RES-XXX)
    RESOURCE_NOT_FOUND = "GMS-RES-001"
    RESOURCE_ALREADY_EXISTS = "GMS-RES-002"
    RESOURCE_ACCESS_DENIED = "GMS-RES-003"

    # System Errors (GMS-SYS-XXX)
    DATABASE_ERROR = "GMS-SYS-001"
    CONFIGURATION_ERROR = "GMS-SYS-002"
    DEPENDENCY_ERROR = "GMS-SYS-003"


# Error code metadata
ERROR_METADATA: dict[ErrorCode, dict[str, Any]] = {
    # General
    ErrorCode.INTERNAL_ERROR: {
        "status_code": 500,
        "message": "An internal server error occurred",
        "category": ErrorCategory.GENERAL,
    },
    ErrorCode.INVALID_REQUEST: {
        "status_code": 400,
        "message": "Invalid request format or parameters",
        "category": ErrorCategory.GENERAL,
    },
    ErrorCode.RATE_LIMITED: {
        "status_code": 429,
        "message": "Rate limit exceeded. Please try again later",
        "category": ErrorCategory.GENERAL,
    },
    ErrorCode.SERVICE_UNAVAILABLE: {
        "status_code": 503,
        "message": "Service temporarily unavailable",
        "category": ErrorCategory.GENERAL,
    },
    # Engine
    ErrorCode.ENGINE_NOT_FOUND: {
        "status_code": 404,
        "message": "Specified physics engine not found",
        "category": ErrorCategory.ENGINE,
    },
    ErrorCode.ENGINE_NOT_LOADED: {
        "status_code": 400,
        "message": "Physics engine not loaded. Load an engine first",
        "category": ErrorCategory.ENGINE,
    },
    ErrorCode.ENGINE_LOAD_FAILED: {
        "status_code": 500,
        "message": "Failed to load physics engine",
        "category": ErrorCategory.ENGINE,
    },
    ErrorCode.ENGINE_NOT_AVAILABLE: {
        "status_code": 400,
        "message": "Requested engine is not available on this system",
        "category": ErrorCategory.ENGINE,
    },
    ErrorCode.ENGINE_INITIALIZATION_FAILED: {
        "status_code": 500,
        "message": "Engine initialization failed",
        "category": ErrorCategory.ENGINE,
    },
    ErrorCode.ENGINE_INVALID_STATE: {
        "status_code": 400,
        "message": "Engine is in an invalid state for this operation",
        "category": ErrorCategory.ENGINE,
    },
    # Simulation
    ErrorCode.SIMULATION_FAILED: {
        "status_code": 500,
        "message": "Simulation execution failed",
        "category": ErrorCategory.SIMULATION,
    },
    ErrorCode.SIMULATION_TIMEOUT: {
        "status_code": 504,
        "message": "Simulation timed out",
        "category": ErrorCategory.SIMULATION,
    },
    ErrorCode.SIMULATION_INVALID_PARAMS: {
        "status_code": 400,
        "message": "Invalid simulation parameters",
        "category": ErrorCategory.SIMULATION,
    },
    ErrorCode.SIMULATION_MODEL_NOT_FOUND: {
        "status_code": 404,
        "message": "Simulation model not found",
        "category": ErrorCategory.SIMULATION,
    },
    ErrorCode.SIMULATION_STATE_ERROR: {
        "status_code": 500,
        "message": "Simulation state error",
        "category": ErrorCategory.SIMULATION,
    },
    ErrorCode.TASK_NOT_FOUND: {
        "status_code": 404,
        "message": "Task not found",
        "category": ErrorCategory.SIMULATION,
    },
    ErrorCode.TASK_NOT_COMPLETED: {
        "status_code": 400,
        "message": "Task has not completed yet",
        "category": ErrorCategory.SIMULATION,
    },
    # Video
    ErrorCode.VIDEO_PIPELINE_NOT_INITIALIZED: {
        "status_code": 500,
        "message": "Video processing pipeline not initialized",
        "category": ErrorCategory.VIDEO,
    },
    ErrorCode.VIDEO_INVALID_FORMAT: {
        "status_code": 400,
        "message": "Invalid video format. Must be a video file",
        "category": ErrorCategory.VIDEO,
    },
    ErrorCode.VIDEO_PROCESSING_FAILED: {
        "status_code": 500,
        "message": "Video processing failed",
        "category": ErrorCategory.VIDEO,
    },
    ErrorCode.VIDEO_ESTIMATOR_INVALID: {
        "status_code": 400,
        "message": "Invalid pose estimator type",
        "category": ErrorCategory.VIDEO,
    },
    ErrorCode.VIDEO_CONFIDENCE_INVALID: {
        "status_code": 400,
        "message": "Invalid confidence threshold",
        "category": ErrorCategory.VIDEO,
    },
    # Analysis
    ErrorCode.ANALYSIS_SERVICE_NOT_INITIALIZED: {
        "status_code": 500,
        "message": "Analysis service not initialized",
        "category": ErrorCategory.ANALYSIS,
    },
    ErrorCode.ANALYSIS_FAILED: {
        "status_code": 500,
        "message": "Analysis operation failed",
        "category": ErrorCategory.ANALYSIS,
    },
    ErrorCode.ANALYSIS_INVALID_TYPE: {
        "status_code": 400,
        "message": "Invalid analysis type",
        "category": ErrorCategory.ANALYSIS,
    },
    # Auth
    ErrorCode.AUTH_TOKEN_INVALID: {
        "status_code": 401,
        "message": "Invalid authentication token",
        "category": ErrorCategory.AUTH,
    },
    ErrorCode.AUTH_TOKEN_EXPIRED: {
        "status_code": 401,
        "message": "Authentication token has expired",
        "category": ErrorCategory.AUTH,
    },
    ErrorCode.AUTH_INSUFFICIENT_PERMISSIONS: {
        "status_code": 403,
        "message": "Insufficient permissions for this operation",
        "category": ErrorCategory.AUTH,
    },
    ErrorCode.AUTH_QUOTA_EXCEEDED: {
        "status_code": 429,
        "message": "Usage quota exceeded for this billing period",
        "category": ErrorCategory.AUTH,
    },
    ErrorCode.AUTH_USER_NOT_FOUND: {
        "status_code": 404,
        "message": "User not found",
        "category": ErrorCategory.AUTH,
    },
    # Validation
    ErrorCode.VALIDATION_FAILED: {
        "status_code": 422,
        "message": "Request validation failed",
        "category": ErrorCategory.VALIDATION,
    },
    ErrorCode.VALIDATION_MISSING_FIELD: {
        "status_code": 422,
        "message": "Required field is missing",
        "category": ErrorCategory.VALIDATION,
    },
    ErrorCode.VALIDATION_INVALID_VALUE: {
        "status_code": 422,
        "message": "Invalid value for field",
        "category": ErrorCategory.VALIDATION,
    },
    ErrorCode.VALIDATION_PATH_INVALID: {
        "status_code": 400,
        "message": "Invalid file path",
        "category": ErrorCategory.VALIDATION,
    },
    # Resource
    ErrorCode.RESOURCE_NOT_FOUND: {
        "status_code": 404,
        "message": "Requested resource not found",
        "category": ErrorCategory.RESOURCE,
    },
    ErrorCode.RESOURCE_ALREADY_EXISTS: {
        "status_code": 409,
        "message": "Resource already exists",
        "category": ErrorCategory.RESOURCE,
    },
    ErrorCode.RESOURCE_ACCESS_DENIED: {
        "status_code": 403,
        "message": "Access to resource denied",
        "category": ErrorCategory.RESOURCE,
    },
    # System
    ErrorCode.DATABASE_ERROR: {
        "status_code": 500,
        "message": "Database operation failed",
        "category": ErrorCategory.SYSTEM,
    },
    ErrorCode.CONFIGURATION_ERROR: {
        "status_code": 500,
        "message": "Server configuration error",
        "category": ErrorCategory.SYSTEM,
    },
    ErrorCode.DEPENDENCY_ERROR: {
        "status_code": 500,
        "message": "Required dependency not available",
        "category": ErrorCategory.SYSTEM,
    },
}


@dataclass
class APIError:
    """Structured API error.

    Attributes:
        code: Error code enum value
        message: Human-readable error message
        details: Additional error details
        request_id: Request ID for tracing
        correlation_id: Correlation ID for cross-service tracing
    """

    code: ErrorCode
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    request_id: str = ""
    correlation_id: str = ""

    def __post_init__(self) -> None:
        """Inject tracing context after initialization."""
        if not self.request_id:
            self.request_id = get_request_id()
        context = get_trace_context()
        if context and not self.correlation_id:
            self.correlation_id = context.correlation_id

    @classmethod
    def from_code(
        cls,
        code: ErrorCode,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> APIError:
        """Create an APIError from an error code.

        Args:
            code: Error code
            message: Optional custom message (uses default if not provided)
            details: Additional error details

        Returns:
            APIError instance
        """
        metadata = ERROR_METADATA.get(code, {})
        return cls(
            code=code,
            message=message or metadata.get("message", "An error occurred"),
            details=details or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON response."""
        result: dict[str, Any] = {
            "error": {
                "code": self.code.value,
                "message": self.message,
            }
        }

        if self.details:
            result["error"]["details"] = self.details

        if self.request_id:
            result["error"]["request_id"] = self.request_id

        if self.correlation_id:
            result["error"]["correlation_id"] = self.correlation_id

        return result

    def to_response(self) -> JSONResponse:
        """Convert to FastAPI JSONResponse."""
        metadata = ERROR_METADATA.get(self.code, {})
        status_code = metadata.get("status_code", 500)
        return JSONResponse(status_code=status_code, content=self.to_dict())


class APIException(HTTPException):
    """HTTP Exception with structured error code.

    Use this instead of HTTPException for consistent error responses.
    """

    def __init__(
        self,
        code: ErrorCode,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        """Initialize APIException.

        Args:
            code: Error code
            message: Optional custom message
            details: Additional error details
        """
        self.error = APIError.from_code(code, message, details)
        metadata = ERROR_METADATA.get(code, {})
        super().__init__(
            status_code=metadata.get("status_code", 500),
            detail=self.error.to_dict(),
        )


def raise_api_error(
    code: ErrorCode,
    message: str | None = None,
    **details: Any,
) -> None:
    """Raise an API exception with structured error code.

    Args:
        code: Error code
        message: Optional custom message
        **details: Additional error details as keyword arguments

    Raises:
        APIException: Always raises
    """
    raise APIException(code=code, message=message, details=details if details else None)


__all__ = [
    "ErrorCategory",
    "ErrorCode",
    "ERROR_METADATA",
    "APIError",
    "APIException",
    "raise_api_error",
]
