"""Standardized API response wrapper for all upstream_drift_tools endpoints.

This module provides a consistent response format across all APIs, ensuring
that clients receive uniform structure for both successful and error responses.

Example:
    >>> from standard_response import StandardResponse, ErrorDetail, ErrorCode
    >>> # Success response
    >>> response = StandardResponse.success(
    ...     data={"result": 123.4},
    ...     processing_time_ms=50
    ... )
    >>> response.to_dict()
    {
        "status": "success",
        "data": {"result": 123.4},
        "error": None,
        "metadata": {"request_id": "...", "processing_time_ms": 50}
    }

    >>> # Error response
    >>> error = ErrorDetail(
    ...     code=ErrorCode.INVALID_INPUT,
    ...     message="Invalid pipe diameter",
    ...     details={"field": "pipe_diameter_m", "reason": "must be > 0"}
    ... )
    >>> response = StandardResponse.error(error=error)
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class ErrorCode(str, Enum):
    """Standard error codes for API responses.

    These codes categorize different types of errors that can occur during
    API request processing, enabling consistent error handling across clients.
    """

    INVALID_INPUT = "INVALID_INPUT"
    """Input validation failed (e.g., negative values for positive fields)."""

    NOT_FOUND = "NOT_FOUND"
    """Requested resource or calculation target not found."""

    SERVER_ERROR = "SERVER_ERROR"
    """Unexpected server-side error."""

    UNSUPPORTED_OPERATION = "UNSUPPORTED_OPERATION"
    """Operation is not supported (e.g., invalid enum value)."""

    TIMEOUT = "TIMEOUT"
    """Request processing timed out."""

    CONSTRAINT_VIOLATION = "CONSTRAINT_VIOLATION"
    """Physical or logical constraint violated (e.g., incompatible parameters)."""


@dataclass
class ErrorDetail:
    """Details of an API error.

    Attributes:
        code: Error code from ErrorCode enum.
        message: Human-readable error message.
        details: Optional nested dict with field-level error info.
        request_id: Request ID where error occurred (auto-populated).
    """

    code: ErrorCode
    message: str
    details: dict[str, Any] | None = None
    request_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


@dataclass
class ResponseMetadata:
    """Metadata accompanying every API response.

    Attributes:
        request_id: Unique identifier for this request/response pair.
        processing_time_ms: Server-side processing time in milliseconds.
        api_version: API version that handled this request.
    """

    request_id: str
    processing_time_ms: float
    api_version: str = "1.0.0"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


class StandardResponse:
    """Standardized response wrapper for all API endpoints.

    Ensures consistent structure across all endpoints:
    - Success responses include data
    - Error responses include error details
    - All responses include metadata with request tracking

    This class is used by all upstream_drift_tools API endpoints to provide
    a uniform interface for downstream consumers.

    Attributes:
        status: "success" or "error"
        data: Response payload (for successful requests)
        error: ErrorDetail object (for error responses)
        metadata: ResponseMetadata with request tracking info
    """

    def __init__(
        self,
        status: str,
        data: dict[str, Any] | None = None,
        error: ErrorDetail | None = None,
        metadata: ResponseMetadata | None = None,
    ):
        """Initialize a StandardResponse.

        Args:
            status: Must be "success" or "error".
            data: Response payload for successful requests.
            error: ErrorDetail object for error responses.
            metadata: Response metadata (auto-generated if not provided).

        Raises:
            ValueError: If status is not "success" or "error".
        """
        if status not in ("success", "error"):
            raise ValueError(f'status must be "success" or "error", got {status!r}')

        self.status = status
        self.data = data
        self.error = error  # type: ignore[assignment]
        self.metadata = metadata or ResponseMetadata(
            request_id=str(uuid4()),
            processing_time_ms=0.0,
        )

    @classmethod
    def success(
        cls,
        data: dict[str, Any],
        processing_time_ms: float = 0.0,
        request_id: str | None = None,
    ) -> StandardResponse:
        """Create a success response.

        Args:
            data: Response payload (required).
            processing_time_ms: Server processing time in milliseconds.
            request_id: Custom request ID (auto-generated if not provided).

        Returns:
            StandardResponse with status="success".

        Example:
            >>> response = StandardResponse.success(
            ...     data={"pressure_drop": 1023.4, "velocity": 45.2},
            ...     processing_time_ms=125
            ... )
        """
        return cls(
            status="success",
            data=data,
            metadata=ResponseMetadata(
                request_id=request_id or str(uuid4()),
                processing_time_ms=processing_time_ms,
            ),
        )

    @classmethod
    def error(
        cls,
        error: ErrorDetail,
        processing_time_ms: float = 0.0,
    ) -> StandardResponse:
        """Create an error response.

        Args:
            error: ErrorDetail object describing the error.
            processing_time_ms: Server processing time in milliseconds.

        Returns:
            StandardResponse with status="error".

        Example:
            >>> error = ErrorDetail(
            ...     code=ErrorCode.INVALID_INPUT,
            ...     message="Pipe diameter must be positive",
            ...     details={"field": "pipe_diameter_m", "value": -1.5}
            ... )
            >>> response = StandardResponse.error(error)
        """
        # Ensure error has a request_id
        if error.request_id is None:
            error.request_id = str(uuid4())

        return cls(
            status="error",
            error=error,
            metadata=ResponseMetadata(
                request_id=error.request_id,
                processing_time_ms=processing_time_ms,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert response to dictionary (suitable for JSON serialization).

        Returns:
            Dictionary with keys: status, data, error, metadata.

        Example:
            >>> response = StandardResponse.success(data={"result": 123})
            >>> response.to_dict()
            {
                "status": "success",
                "data": {"result": 123},
                "error": None,
                "metadata": {
                    "request_id": "...",
                    "processing_time_ms": 0.0,
                    "api_version": "1.0.0"
                }
            }
        """
        result = {
            "status": self.status,
            "data": self.data,
            "error": self.error.to_dict() if self.error else None,  # type: ignore[attr-defined,truthy-function]
            "metadata": self.metadata.to_dict(),
        }
        return result

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"StandardResponse(status={self.status!r}, "
            f"data={self.data!r}, error={self.error!r}, "
            f"metadata={self.metadata!r})"
        )
