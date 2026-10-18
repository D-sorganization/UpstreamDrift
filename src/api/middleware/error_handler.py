"""Centralized API error handling decorator.

Eliminates duplicated try/except HTTPException patterns across route modules.
See issue #1489.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import re
from collections.abc import Callable
from typing import Any

from fastapi import HTTPException

from src.api.utils.error_codes import APIException, ClientInputError, ErrorCode

logger = logging.getLogger(__name__)

_WIN_PATH_RE = re.compile(r"(?i)[a-z]:[/\\][^\s'\"]*")
_UNC_PATH_RE = re.compile(r"\\\\[^\s'\"]+")
_POSIX_PATH_RE = re.compile(
    r"/(?:Users|home|etc|var|tmp|opt|root|bin|lib|srv|private|usr|abs)[/\w.-]*"
)
_INTERNAL_MARKERS = (
    "codec can't decode",
    "[Errno ",
    "Traceback (most recent call last)",
    "<class '",
    "<function ",
    'File "',
)


def _contains_path_or_internal_leak(msg: str) -> bool:
    """Return True if the message contains an absolute path or internal exception marker."""
    if any(marker in msg for marker in _INTERNAL_MARKERS):
        return True
    return bool(
        _WIN_PATH_RE.search(msg)
        or _UNC_PATH_RE.search(msg)
        or _POSIX_PATH_RE.search(msg)
    )


def _sanitize_paths(msg: str) -> str:
    """Redact absolute filesystem paths from an error string."""
    msg = _WIN_PATH_RE.sub("[REDACTED_PATH]", msg)
    msg = _UNC_PATH_RE.sub("[REDACTED_PATH]", msg)
    msg = _POSIX_PATH_RE.sub("[REDACTED_PATH]", msg)
    return msg


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
    if isinstance(e, FileNotFoundError):
        logger.warning("Resource not found (%s): %s", func_name, e, exc_info=True)  # noqa: LOG014
        raise APIException(
            code=ErrorCode.RESOURCE_NOT_FOUND,
            message="Resource not found",
        ) from e
    if isinstance(e, PermissionError):
        logger.warning("Permission denied (%s): %s", func_name, e, exc_info=True)  # noqa: LOG014
        raise APIException(
            code=ErrorCode.RESOURCE_ACCESS_DENIED,
            message="Permission denied",
        ) from e
    if isinstance(e, TimeoutError):
        logger.warning("Operation timed out (%s): %s", func_name, e, exc_info=True)  # noqa: LOG014
        raise APIException(
            code=ErrorCode.SIMULATION_TIMEOUT,
            message="Operation timed out",
        ) from e
    if isinstance(e, ValueError):
        logger.warning("Client input error (%s): %s", func_name, e, exc_info=True)  # noqa: LOG014
        if isinstance(e, UnicodeError):
            raise APIException(
                code=ErrorCode.INVALID_REQUEST,
                message="Invalid request encoding",
            ) from e
        if isinstance(e, ClientInputError):
            raise APIException(
                code=ErrorCode.INVALID_REQUEST,
                message=_sanitize_paths(str(e)),
            ) from e
        raw_msg = str(e)
        if _contains_path_or_internal_leak(raw_msg):
            raise APIException(
                code=ErrorCode.INVALID_REQUEST,
                message="Invalid request format or parameters",
            ) from e
        raise APIException(
            code=ErrorCode.INVALID_REQUEST,
            message=_sanitize_paths(raw_msg),
        ) from e
    if isinstance(e, NotImplementedError):
        raise HTTPException(status_code=501, detail=str(e)) from e
    if isinstance(e, (RuntimeError, TypeError, KeyError, AttributeError, OSError)):
        logger.exception("Unhandled error in %s: %s", func_name, e)  # noqa: LOG004
        raise APIException(
            code=ErrorCode.INTERNAL_ERROR,
            message="Internal server error",
        ) from e
    # Exhaustive fallback: any exception type not listed above still produces a
    # well-formed 500 response rather than silently returning None.
    logger.exception("Unexpected error type in %s: %s", func_name, e)  # noqa: LOG004
    raise APIException(
        code=ErrorCode.INTERNAL_ERROR,
        message="Internal server error",
    ) from e


def handle_api_errors(func: Callable[..., Any]) -> Callable[..., Any]:  # noqa: C901
    """Decorator that provides consistent error handling for API route handlers.

    Catches common exceptions and maps them to appropriate HTTP responses:
    - ValueError -> 400 Bad Request
    - FileNotFoundError -> 404 Not Found
    - PermissionError -> 403 Forbidden
    - TimeoutError -> 504 Gateway Timeout
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


__all__ = [
    "ClientInputError",
    "handle_api_errors",
]
