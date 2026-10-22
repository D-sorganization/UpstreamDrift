"""Design by Contract (DbC) framework for Repository_Management.

Provides ``require()`` and ``ensure()`` helpers that raise descriptive
``ContractViolationError`` exceptions when preconditions or postconditions
are not met.  Intended for use at the top of public functions to make
expectations explicit and fail fast.

Usage::

    from src.contracts import require, ensure

    def divide(a: float, b: float) -> float:
        require(b != 0, "divisor must be non-zero")
        result = a / b
        ensure(isinstance(result, float), "result must be a float")
        return result
"""

from __future__ import annotations


class ContractViolationError(Exception):
    """Raised when a DbC precondition or postcondition is violated."""


class PreconditionError(ContractViolationError):
    """Raised when a precondition (``require``) is violated."""


class PostconditionError(ContractViolationError):
    """Raised when a postcondition (``ensure``) is violated."""


def require(condition: bool, message: str = "Precondition violated") -> None:
    """Assert a precondition before executing a function body.

    Args:
        condition: Boolean expression that must be ``True``.
        message: Human-readable description of the expectation.

    Raises:
        PreconditionError: If *condition* is falsy.
    """
    if not condition:
        raise PreconditionError(message)


def ensure(condition: bool, message: str = "Postcondition violated") -> None:
    """Assert a postcondition after computing a result.

    Args:
        condition: Boolean expression that must be ``True``.
        message: Human-readable description of the guarantee.

    Raises:
        PostconditionError: If *condition* is falsy.
    """
    if not condition:
        raise PostconditionError(message)
