"""Request tracing and correlation ID utilities.

This module provides request tracing capabilities for better diagnostics
and troubleshooting. Every request gets a unique correlation ID that can
be traced through logs and error responses.

Usage:
    from src.api.utils.tracing import get_request_id, trace_context

    # In middleware or route
    request_id = get_request_id(request)
    logger.info("Processing request", extra={"request_id": request_id})

Design by Contract:
    - Every request MUST have a correlation ID
    - Correlation IDs MUST be unique per request
    - Correlation IDs MUST be propagated to all log entries and responses
"""

from __future__ import annotations

import contextvars
import uuid
from dataclasses import dataclass, field
from typing import Any

from src.shared.python.logging_config import get_logger

logger = get_logger(__name__)

# Context variable for request-scoped tracing
_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)
_trace_context_var: contextvars.ContextVar[TraceContext | None] = (
    contextvars.ContextVar("trace_context", default=None)
)

# Header name for correlation ID
CORRELATION_ID_HEADER = "X-Correlation-ID"
REQUEST_ID_HEADER = "X-Request-ID"


@dataclass
class TraceContext:
    """Context for request tracing.

    Attributes:
        request_id: Unique identifier for this request
        correlation_id: ID for tracing across services (may be inherited)
        operation: Current operation name
        start_time: Request start timestamp
        metadata: Additional tracing metadata
    """

    request_id: str
    correlation_id: str
    operation: str = ""
    start_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/responses."""
        return {
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "operation": self.operation,
            "metadata": self.metadata,
        }


def generate_request_id() -> str:
    """Generate a unique request ID.

    Returns:
        Unique UUID string prefixed with 'req_'
    """
    return f"req_{uuid.uuid4().hex[:16]}"


def generate_correlation_id() -> str:
    """Generate a correlation ID for cross-service tracing.

    Returns:
        Unique UUID string prefixed with 'cor_'
    """
    return f"cor_{uuid.uuid4().hex[:16]}"


def get_request_id() -> str:
    """Get the current request ID from context.

    Returns:
        Current request ID or empty string if not set.
    """
    return _request_id_var.get()


def set_request_id(request_id: str) -> contextvars.Token[str]:
    """Set the request ID in context.

    Args:
        request_id: Request ID to set.

    Returns:
        Token that can be used to reset the context.
    """
    return _request_id_var.set(request_id)


def get_trace_context() -> TraceContext | None:
    """Get the current trace context.

    Returns:
        Current TraceContext or None if not set.
    """
    return _trace_context_var.get()


def set_trace_context(context: TraceContext) -> contextvars.Token[TraceContext | None]:
    """Set the trace context.

    Args:
        context: TraceContext to set.

    Returns:
        Token that can be used to reset the context.
    """
    return _trace_context_var.set(context)


class RequestTracer:
    """Middleware-compatible request tracer.

    This class provides tracing functionality that can be used as
    FastAPI middleware to automatically trace all requests.

    Example:
        app.middleware("http")(RequestTracer().trace_request)
    """

    async def trace_request(self, request: Any, call_next: Any) -> Any:
        """Middleware function for request tracing.

        Args:
            request: FastAPI Request object
            call_next: Next middleware/route handler

        Returns:
            Response with tracing headers added
        """
        import time

        # Extract or generate correlation ID
        correlation_id = request.headers.get(CORRELATION_ID_HEADER)
        if not correlation_id:
            correlation_id = generate_correlation_id()

        # Generate request ID
        request_id = generate_request_id()

        # Create trace context
        context = TraceContext(
            request_id=request_id,
            correlation_id=correlation_id,
            operation=f"{request.method} {request.url.path}",
            start_time=time.time(),
            metadata={
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host if request.client else "unknown",
            },
        )

        # Set context variables
        request_id_token = set_request_id(request_id)
        trace_token = set_trace_context(context)

        try:
            # Log request start
            logger.info(
                "request_started",
                extra={
                    "request_id": request_id,
                    "correlation_id": correlation_id,
                    "method": request.method,
                    "path": request.url.path,
                },
            )

            # Process request
            response = await call_next(request)

            # Calculate duration
            duration_ms = (time.time() - context.start_time) * 1000

            # Log request completion
            logger.info(
                "request_completed",
                extra={
                    "request_id": request_id,
                    "correlation_id": correlation_id,
                    "status_code": response.status_code,
                    "duration_ms": round(duration_ms, 2),
                },
            )

            # Add tracing headers to response
            response.headers[REQUEST_ID_HEADER] = request_id
            response.headers[CORRELATION_ID_HEADER] = correlation_id
            response.headers["X-Response-Time-Ms"] = str(round(duration_ms, 2))

            return response

        except (RuntimeError, ValueError, OSError) as e:
            # Log error with trace context
            duration_ms = (time.time() - context.start_time) * 1000
            logger.error(
                "request_failed",
                extra={
                    "request_id": request_id,
                    "correlation_id": correlation_id,
                    "error": str(e),
                    "duration_ms": round(duration_ms, 2),
                },
                exc_info=True,
            )
            raise

        finally:
            # Reset context variables
            _request_id_var.reset(request_id_token)
            _trace_context_var.reset(trace_token)


def traced_log(
    level: str,
    message: str,
    **kwargs: Any,
) -> None:
    """Log a message with automatic trace context injection.

    Args:
        level: Log level (debug, info, warning, error, critical)
        message: Log message
        **kwargs: Additional fields to include in log
    """
    extra = dict(kwargs)

    # Inject trace context
    request_id = get_request_id()
    if request_id:
        extra["request_id"] = request_id

    context = get_trace_context()
    if context:
        extra["correlation_id"] = context.correlation_id

    log_func = getattr(logger, level, logger.info)
    log_func(message, extra=extra)


__all__ = [
    "CORRELATION_ID_HEADER",
    "REQUEST_ID_HEADER",
    "TraceContext",
    "RequestTracer",
    "generate_request_id",
    "generate_correlation_id",
    "get_request_id",
    "set_request_id",
    "get_trace_context",
    "set_trace_context",
    "traced_log",
]
