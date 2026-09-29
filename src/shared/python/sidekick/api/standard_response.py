# mypy: ignore-errors
"""Standardized API response wrapper for all upstream_drift_tools endpoints.

This module provides a consistent response format across all APIs, ensuring
that clients receive uniform structure for both successful and error responses.
"""

from __future__ import annotations

import logging
import time
import uuid
from importlib import import_module
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from datetime import tzinfo
    from enum import StrEnum

    UTC: tzinfo
else:
    try:
        _compatibility = import_module("src.shared.python.compatibility")
    except ImportError:  # pragma: no cover
        try:
            _compatibility = import_module("...compatibility", __package__)
        except ImportError:
            _compatibility = import_module("compatibility")

    StrEnum = _compatibility.StrEnum
    UTC = _compatibility.UTC
from pydantic import BaseModel, Field

__all__ = [
    "ErrorCode",
    "ErrorDetail",
    "ResponseMetadata",
    "StandardResponse",
    "StandardResponseBuilder",
]

_logger = logging.getLogger(__name__)

T = TypeVar("T")


class ErrorCode(StrEnum):
    """Standard error codes for API responses.

    These codes categorize different types of errors that can occur during
    API request processing, enabling consistent error handling across clients.
    """

    INVALID_INPUT = "INVALID_INPUT"
    NOT_FOUND = "NOT_FOUND"
    SERVER_ERROR = "SERVER_ERROR"
    CALCULATION_ERROR = "CALCULATION_ERROR"
    CONSTRAINT_VIOLATION = "CONSTRAINT_VIOLATION"
    UNSUPPORTED_OPERATION = "UNSUPPORTED_OPERATION"
    TIMEOUT = "TIMEOUT"


class ErrorDetail(BaseModel):
    """Details of an API error.

    Attributes:
        code: Error code from ErrorCode enum.
        message: Human-readable error message.
        details: Optional detailed information.
        request_id: Request ID where error occurred.
    """

    code: ErrorCode = Field(description="Machine-readable error code")
    message: str = Field(description="Human-readable error message")
    details: Any | None = Field(default=None, description="Additional error context")
    request_id: str | None = Field(
        default=None, description="Request ID where error occurred"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return self.model_dump()


class ResponseMetadata(BaseModel):
    """Metadata accompanying every API response.

    Attributes:
        request_id: Unique identifier for this request/response pair.
        processing_time_ms: Server-side processing time in milliseconds.
        timestamp_utc: ISO 8601 timestamp when response was generated.
        api_version: API version that handled this request.
    """

    request_id: str = Field(description="Unique request identifier")
    processing_time_ms: float = Field(description="Processing time in milliseconds")
    timestamp_utc: str | None = Field(
        default=None, description="ISO 8601 response timestamp"
    )
    api_version: str = Field(default="1.0.0", description="API version")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return self.model_dump()


class StandardResponse(BaseModel, Generic[T]):
    """Standardized API response wrapper."""

    status: str = Field(
        description='Response status: "success" or "error"',
        pattern="^(success|error)$",
    )
    data: T | None = Field(default=None, description="Calculation result or None")
    error: ErrorDetail | None = Field(default=None, description="Error details")
    metadata: ResponseMetadata = Field(description="Response metadata")

    def to_dict(self) -> dict[str, Any]:
        """Convert response to dictionary."""
        return self.model_dump()

    @classmethod
    def success(
        cls,
        data: Any,
        processing_time_ms: float | None = None,
        request_id: str | None = None,
        api_version: str = "1.0.0",
    ) -> StandardResponse[Any]:
        """Create a success response with explicit metadata controls."""
        return cls(
            status="success",
            data=data,
            error=None,
            metadata=_build_metadata(
                processing_time_ms=processing_time_ms,
                request_id=request_id,
                api_version=api_version,
            ),
        )


class _StandardResponseErrorFactory:
    def __get__(
        self,
        instance: StandardResponse[Any] | None,
        owner: type[StandardResponse[Any]],
    ) -> Any:
        if instance is not None:
            return instance.__dict__.get("error")

        def create_error_response(
            *,
            error: ErrorDetail,
            processing_time_ms: float | None = None,
            request_id: str | None = None,
            api_version: str = "1.0.0",
        ) -> StandardResponse[Any]:
            metadata_request_id = request_id or error.request_id
            metadata = _build_metadata(
                processing_time_ms=processing_time_ms,
                request_id=metadata_request_id,
                api_version=api_version,
            )
            response_error = error
            if response_error.request_id is None:
                response_error = response_error.model_copy(
                    update={"request_id": metadata.request_id}
                )
            return owner(
                status="error",
                data=None,
                error=response_error,
                metadata=metadata,
            )

        return create_error_response


class StandardResponseBuilder:
    """Builder for creating StandardResponse instances with tracking metadata."""

    def __init__(self) -> None:
        self._request_id = str(uuid.uuid4())
        self._start_time = time.time()

    def success(
        self,
        data: Any,
        api_version: str = "1.0.0",
    ) -> StandardResponse[Any]:
        processing_time_ms = (time.time() - self._start_time) * 1000
        metadata = ResponseMetadata(
            request_id=self._request_id,
            processing_time_ms=round(processing_time_ms, 2),
            timestamp_utc=_get_utc_timestamp(),
            api_version=api_version,
        )
        return StandardResponse(
            status="success",
            data=data,
            error=None,
            metadata=metadata,
        )

    def error(
        self,
        code: ErrorCode,
        message: str,
        details: Any | None = None,
        api_version: str = "1.0.0",
    ) -> StandardResponse[Any]:
        processing_time_ms = (time.time() - self._start_time) * 1000
        metadata = ResponseMetadata(
            request_id=self._request_id,
            processing_time_ms=round(processing_time_ms, 2),
            timestamp_utc=_get_utc_timestamp(),
            api_version=api_version,
        )
        error_detail = ErrorDetail(
            code=code,
            message=message,
            details=details,
            request_id=self._request_id,
        )
        return StandardResponse(
            status="error",
            data=None,
            error=error_detail,
            metadata=metadata,
        )


def _build_metadata(
    *,
    processing_time_ms: float | None,
    request_id: str | None,
    api_version: str,
) -> ResponseMetadata:
    if processing_time_ms is not None and processing_time_ms < 0:
        raise ValueError("processing_time_ms must be non-negative")
    if request_id is not None and not request_id:
        raise ValueError("request_id must be a non-empty string")

    return ResponseMetadata(
        request_id=request_id or str(uuid.uuid4()),
        processing_time_ms=0.0 if processing_time_ms is None else processing_time_ms,
        timestamp_utc=_get_utc_timestamp(),
        api_version=api_version,
    )


def _get_utc_timestamp() -> str:
    from datetime import datetime

    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


StandardResponse.error = _StandardResponseErrorFactory()  # type: ignore[method-assign]
