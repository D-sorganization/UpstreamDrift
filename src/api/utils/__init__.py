"""API utility helpers.

This module provides shared utilities for the Golf Modeling Suite API:
- datetime_compat: Timezone-aware datetime utilities (DRY)
- error_codes: Structured error codes for diagnostics
- tracing: Request tracing and correlation IDs
- path_validation: Secure file path validation
"""

from .datetime_compat import UTC, add_days, add_minutes, iso_format, utc_now
from .error_codes import (
    APIError,
    APIException,
    ErrorCategory,
    ErrorCode,
    raise_api_error,
)
from .tracing import (
    CORRELATION_ID_HEADER,
    REQUEST_ID_HEADER,
    RequestTracer,
    TraceContext,
    generate_correlation_id,
    generate_request_id,
    get_request_id,
    get_trace_context,
    traced_log,
)

__all__ = [
    # datetime_compat
    "UTC",
    "utc_now",
    "iso_format",
    "add_days",
    "add_minutes",
    # error_codes
    "APIError",
    "APIException",
    "ErrorCode",
    "ErrorCategory",
    "raise_api_error",
    # tracing
    "CORRELATION_ID_HEADER",
    "REQUEST_ID_HEADER",
    "RequestTracer",
    "TraceContext",
    "generate_correlation_id",
    "generate_request_id",
    "get_request_id",
    "get_trace_context",
    "traced_log",
]
