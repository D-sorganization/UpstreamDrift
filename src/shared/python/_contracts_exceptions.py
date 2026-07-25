from __future__ import annotations

import logging

from src.shared.python._contracts_level import (
    ContractLevel,
    _ContractState,
)

logger = logging.getLogger(__name__)


class ContractViolationError(AssertionError, ValueError):
    """Base exception for contract violations.

    The constructor validates its own arguments: a contract violation reported
    without a condition type or message is not diagnosable, and silently
    accepting ``None`` produces misleading detail strings such as
    ``"[DbC None] None"``. Subclasses supply ``condition_type`` themselves and
    forward ``message``, so both checks live here.

    Raises:
        ValueError: If ``condition_type`` or ``message`` is missing or blank.
    """

    def __init__(
        self,
        condition_type: str,
        message: str,
        value=None,
    ) -> None:
        if not isinstance(condition_type, str) or not condition_type.strip():
            raise ValueError(
                "condition_type must be provided as a non-empty string "
                f"(got: {condition_type!r})"
            )
        if not isinstance(message, str) or not message.strip():
            raise ValueError(
                f"message must be provided as a non-empty string (got: {message!r})"
            )
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
        exc_cls = _VIOLATION_CLASSES.get(condition_type)
        if exc_cls is None:
            # Unknown condition type: fall back to the base class. It takes
            # condition_type as its first argument, unlike the subclasses, so
            # it must be constructed explicitly — passing (message, value)
            # here would bind message to condition_type and silently drop the
            # message entirely.
            raise ContractViolationError(condition_type, message, value)
        raise exc_cls(message, value)
    if level == ContractLevel.WARN:
        detail = f"[DbC {condition_type}] {message}"
        if value is not None:
            detail += f" (got: {value!r})"
        logger.warning(detail)
