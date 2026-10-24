"""Shared fixtures for the API route tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest


def _error_message(response: Any) -> str:
    """Return the human-readable message of an API error response.

    ``handle_api_errors`` (``src/api/middleware/error_handler.py``) maps client
    input errors to ``APIException``, whose ``HTTPException.detail`` is the
    envelope ``{"error": {"code": ..., "message": ...}}``; the response body is
    therefore ``{"detail": {"error": {...}}}``. Asserting on the extracted
    string (not ``"x" in response.json()["detail"]``) checks the message
    rather than dict keys.
    """
    body = response.json()
    error = body.get("detail", {}).get("error")
    assert isinstance(error, dict), f"expected an error envelope, got {body!r}"
    message = error.get("message")
    assert isinstance(message, str), f"envelope has no message: {body!r}"
    return message


@pytest.fixture
def error_message() -> Callable[[Any], str]:
    """Extract the message from an API error-envelope response."""
    return _error_message
