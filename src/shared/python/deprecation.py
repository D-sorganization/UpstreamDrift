"""Deprecation utilities for the Tools shared library.

Provides a ``@deprecated`` decorator for marking functions and methods that
are scheduled for removal.  Emits a :class:`DeprecationWarning` on first
call so that downstream consumers see actionable guidance.

Usage::

    from deprecation import deprecated

    @deprecated(
        reason="Use new_function() instead.",
        removal_version="2.0.0",
    )
    def old_function(x: float) -> float:
        return new_function(x)

The decorator preserves the original function's ``__name__``, ``__doc__``,
and all other metadata via :func:`functools.wraps`.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def deprecated(
    reason: str = "",
    removal_version: str | None = None,
) -> Callable[[F], F]:
    """Mark a callable as deprecated.

    Emits a :class:`DeprecationWarning` when the decorated function is called.

    Args:
        reason: Human-readable explanation and migration guidance.
        removal_version: The version in which the callable will be removed.

    Returns:
        A decorator that wraps the target callable.

    Raises:
        TypeError: If ``reason`` is not a string.
        ValueError: If ``removal_version`` is provided but is an empty string.

    Example::

        @deprecated(reason="Use calculate() instead.", removal_version="2.0.0")
        def legacy_calculate(x: float) -> float:
            return calculate(x)
    """
    if not isinstance(reason, str):
        raise TypeError(f"reason must be a string, got {type(reason).__name__}")
    if removal_version is not None and not removal_version.strip():
        raise ValueError("removal_version must not be an empty string")

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            message = f"{func.__qualname__} is deprecated"
            if reason:
                message += f": {reason}"
            if removal_version:
                message += f" (will be removed in {removal_version})"
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
