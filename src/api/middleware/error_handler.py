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


def handle_api_errors(func: Callable[..., Any]) -> Callable[..., Any]:
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
            except HTTPException:
                raise
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            except FileNotFoundError as e:
                raise HTTPException(status_code=404, detail=str(e)) from e
            except PermissionError as e:
                raise HTTPException(status_code=403, detail=str(e)) from e
            except NotImplementedError as e:
                raise HTTPException(status_code=501, detail=str(e)) from e
            except Exception as e:
                logger.exception("Unhandled error in %s: %s", func.__name__, e)
                raise HTTPException(
                    status_code=500, detail="Internal server error"
                ) from e

        return async_wrapper

    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except HTTPException:
            raise
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except PermissionError as e:
            raise HTTPException(status_code=403, detail=str(e)) from e
        except NotImplementedError as e:
            raise HTTPException(status_code=501, detail=str(e)) from e
        except Exception as e:
            logger.exception("Unhandled error in %s: %s", func.__name__, e)
            raise HTTPException(
                status_code=500, detail="Internal server error"
            ) from e

    return sync_wrapper
