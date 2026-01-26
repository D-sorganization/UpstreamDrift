"""Upload size validation middleware."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import cast

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from src.api.config import MAX_UPLOAD_SIZE_BYTES, MAX_UPLOAD_SIZE_MB
from src.api.middleware.security_headers import add_security_headers_to_response


async def validate_upload_size(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Reject requests exceeding upload size limits."""
    content_length = request.headers.get("content-length")

    if content_length:
        try:
            content_length_int = int(content_length)
        except ValueError:
            response = JSONResponse(
                status_code=400,
                content={"detail": "Invalid Content-Length header"},
            )
            return add_security_headers_to_response(response, request)
        if content_length_int > MAX_UPLOAD_SIZE_BYTES:
            response = JSONResponse(
                status_code=413,
                content={
                    "detail": f"Request too large. Maximum size is {MAX_UPLOAD_SIZE_MB}MB"
                },
            )
            return add_security_headers_to_response(response, request)

    return cast(Response, await call_next(request))
