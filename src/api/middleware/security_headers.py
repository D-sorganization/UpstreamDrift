"""Security headers middleware for the API."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import Request
from fastapi.responses import Response

from src.api.config import HSTS_MAX_AGE_SECONDS


async def add_security_headers(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Add security headers to all responses."""
    response = await call_next(request)
    return add_security_headers_to_response(response, request)


def add_security_headers_to_response(response: Response, request: Request) -> Response:
    """Add security headers to a response (used for early-return responses)."""
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    if request.url.scheme == "https":
        response.headers["Strict-Transport-Security"] = (
            f"max-age={HSTS_MAX_AGE_SECONDS}; includeSubDomains"
        )
    return response
