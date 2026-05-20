from __future__ import annotations

from typing import Any

from src.shared.python._contracts_exceptions import _handle_violation
from src.shared.python._contracts_level import ContractLevel, _ContractState


def require(condition: bool, message: str, value: Any = None) -> None:
    """Assert a pre-condition at function entry."""
    if condition is None:
        raise ValueError("condition must be provided")
    if _ContractState.level == ContractLevel.OFF:
        return
    if not condition:
        _handle_violation("pre-condition", message, value)


def ensure(condition: bool, message: str, value: Any = None) -> None:
    """Assert a post-condition before function return."""
    if condition is None:
        raise ValueError("condition must be provided")
    if _ContractState.level == ContractLevel.OFF:
        return
    if not condition:
        _handle_violation("post-condition", message, value)


def invariant(condition: bool, message: str, value: Any = None) -> None:
    """Assert a class or loop invariant."""
    if condition is None:
        raise ValueError("condition must be provided")
    if _ContractState.level == ContractLevel.OFF:
        return
    if not condition:
        _handle_violation("invariant", message, value)
