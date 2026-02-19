"""Core contract primitives (function-call style).

Provides ``require()`` and ``ensure()`` for inline precondition and
postcondition checks within function bodies.
"""

from __future__ import annotations

from typing import Any

from src.shared.python.logging_pkg.logging_config import get_logger

from .exceptions import (
    ContractViolationError,
    InvariantError,
    PostconditionError,
    PreconditionError,
)
from .level import ContractLevel

logger = get_logger(__name__)


def _handle_violation(
    contract_type: str,
    message: str,
    function_name: str | None = None,
    value: Any = None,
) -> None:
    """Handle a contract violation according to the current DBC_LEVEL."""
    from .level import DBC_LEVEL  # re-import to capture runtime changes

    if DBC_LEVEL == ContractLevel.ENFORCE:
        if contract_type == "Precondition":
            raise PreconditionError(message, function_name=function_name, value=value)
        if contract_type == "Postcondition":
            raise PostconditionError(message, function_name=function_name, result=value)
        if contract_type == "Invariant":
            raise InvariantError(message)
        raise ContractViolationError(contract_type, message)
    if DBC_LEVEL == ContractLevel.WARN:
        detail = f"[DbC {contract_type}] {message}"
        if value is not None:
            detail += f" (got: {value!r})"
        logger.warning(detail)
    # OFF: do nothing


def require(condition: bool, message: str, value: Any = None) -> None:
    """Assert a precondition at function entry.

    Args:
        condition: Boolean expression that must hold.
        message: Descriptive message for the violated contract.
        value: The offending value, for diagnostics.
    """
    from .level import DBC_LEVEL  # re-import to capture runtime changes

    if DBC_LEVEL == ContractLevel.OFF:
        return
    if not condition:
        _handle_violation("Precondition", message, value=value)


def ensure(condition: bool, message: str, value: Any = None) -> None:
    """Assert a postcondition before function return.

    Args:
        condition: Boolean expression that must hold.
        message: Descriptive message for the violated contract.
        value: The offending value, for diagnostics.
    """
    from .level import DBC_LEVEL  # re-import to capture runtime changes

    if DBC_LEVEL == ContractLevel.OFF:
        return
    if not condition:
        _handle_violation("Postcondition", message, value=value)
