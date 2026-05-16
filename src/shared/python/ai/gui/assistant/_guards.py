"""Shared Design-by-Contract precondition helpers for the assistant package."""

from __future__ import annotations

from typing import Any


def require_not_none(value: Any, name: str) -> None:
    """Raise ValueError if *value* is None.

    Args:
        value: The value to check.
        name: Argument name used in the error message.

    Raises:
        ValueError: When *value* is ``None``.
    """
    if value is None:
        raise ValueError(f"{name} must be provided")
