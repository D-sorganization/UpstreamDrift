"""Centralized API error handling decorator.

Eliminates duplicated try/except HTTPException patterns across route modules.
See issue #1489.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Callable
from typing import Any

from fastapi import HTTPException

logger = logging.getLogger(__name__)


def _handle_common_exceptions(e: Exception, func_name: str) -> None:
    """Handle common exceptions and raise appropriate HTTPException.

    Args:
        e: The exception to handle
        func_name: Name of the function where the exception occurred

    Raises:
        HTTPException: Always — no code path returns None.
    """
    if isinstance(e, HTTPException):
        raise e
    if isinstance(e, ValueError):
        raise HTTPException(status_code=400, detail=str(e)) from e
    if isinstance(e, FileNotFoundError):
        raise HTTPException(status_code=404, detail=str(e)) from e
    if isinstance(e, PermissionError):
        raise HTTPException(status_code=403, detail=str(e)) from e
    if isinstance(e, NotImplementedError):
        raise HTTPException(status_code=501, detail=str(e)) from e
    if isinstance(e, (RuntimeError, TypeError, KeyError, AttributeError, OSError)):
        logger.exception("Unhandled error in %s: %s", func_name, e)
        raise HTTPException(status_code=500, detail="Internal server error") from e
    # Exhaustive fallback: any exception type not listed above still produces a
    # well-formed 500 response rather than silently returning None.
    logger.exception("Unexpected error type in %s: %s", func_name, e)
    raise HTTPException(status_code=500, detail="Internal server error") from e


def handle_api_errors(func: Callable[..., Any]) -> Callable[..., Any]:  # noqa: C901
    """Decorator that provides consistent error handling for API route handlers.

    Catches common exceptions and maps them to appropriate HTTP responses:
    - ValueError -> 400 Bad Request
    - FileNotFoundError -> 404 Not Found
    - PermissionError -> 403 Forbidden
    - NotImplementedError -> 501 Not Implemented
    - HTTPException -> re-raised as-is
    - Exception -> 500 Internal Server Error (logged)

    Supports both async and sync route handlers.
    """
    if asyncio.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except (
                HTTPException,
                ValueError,
                FileNotFoundError,
                PermissionError,
                NotImplementedError,
                RuntimeError,
                TypeError,
                KeyError,
                AttributeError,
                OSError,
            ) as e:
                _handle_common_exceptions(e, func.__name__)
                return None  # pragma: no cover

        return async_wrapper

    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except (
            HTTPException,
            ValueError,
            FileNotFoundError,
            PermissionError,
            NotImplementedError,
            RuntimeError,
            TypeError,
            KeyError,
            AttributeError,
            OSError,
        ) as e:
            _handle_common_exceptions(e, func.__name__)
            return None  # pragma: no cover

    return sync_wrapper
