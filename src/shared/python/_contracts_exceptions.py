from __future__ import annotations

import logging

from src.shared.python._contracts_level import (
    ContractLevel,
    _ContractState,
)

logger = logging.getLogger(__name__)


class ContractViolationError(AssertionError, ValueError):
    """Base exception for contract violations."""

    def __init__(
        self,
        condition_type: str,
        message: str,
        value=None,
    ) -> None:
        self.condition_type = condition_type
        self.message = message
        self.value = value
        detail = f"[DbC {condition_type}] {message}"
        if value is not None:
            detail += f" (got: {value!r})"
        super().__init__(detail)


class PreconditionError(ContractViolationError):
    """Raised when a pre-condition is violated."""

    def __init__(self, message: str, value=None) -> None:
        super().__init__("pre-condition", message, value)


class PostconditionError(ContractViolationError):
    """Raised when a post-condition is violated."""

    def __init__(self, message: str, value=None) -> None:
        super().__init__("post-condition", message, value)


class InvariantError(ContractViolationError):
    """Raised when a class or loop invariant is violated."""

    def __init__(self, message: str, value=None) -> None:
        super().__init__("invariant", message, value)


class ContractEvaluationError(ContractViolationError):
    """Raised when a contract condition cannot be evaluated.

    This error is raised when a precondition or postcondition lambda/function
    cannot be evaluated due to signature mismatches, type errors, or other
    evaluation failures. This ensures contracts fail closed rather than silently
    passing when the condition cannot be checked.
    """

    def __init__(self, message: str, value=None) -> None:
        super().__init__("evaluation-error", message, value)


_VIOLATION_CLASSES: dict[str, type[ContractViolationError]] = {
    "pre-condition": PreconditionError,
    "post-condition": PostconditionError,
    "invariant": InvariantError,
    "evaluation-error": ContractEvaluationError,
}


def _handle_violation(
    condition_type: str,
    message: str,
    value=None,
) -> None:
    level = _ContractState.level
    if level == ContractLevel.ENFORCE:
        exc_cls = _VIOLATION_CLASSES.get(condition_type, ContractViolationError)
        raise exc_cls(message, value)
    if level == ContractLevel.WARN:
        detail = f"[DbC {condition_type}] {message}"
        if value is not None:
            detail += f" (got: {value!r})"
        logger.warning(detail)
