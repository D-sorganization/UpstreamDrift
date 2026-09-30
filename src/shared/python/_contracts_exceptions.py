from __future__ import annotations

import logging

from src.shared.python._contracts_level import (
    ContractLevel,
    _ContractState,
)
from src.shared.python.contracts import (
    ContractViolationError,
    InvariantError,
    PostconditionError,
    PreconditionError,
)

logger = logging.getLogger(__name__)

# Exception-identity seam: the classes above are re-exported from the public
# ``src.shared.python.contracts`` module instead of being redefined here, so
# decorators that raise (e.g. ``precondition`` / postcondition``) produce the
# SAME class object that callers importing
# ``from src.shared.python._contracts_exceptions import PreconditionError``
# receive. Both import paths must remain usable and referentially identical.

__all__ = [
    "ContractViolationError",
    "PostconditionError",
    "PreconditionError",
    "InvariantError",
    "ContractEvaluationError",
]


class ContractEvaluationError(ContractViolationError):
    """Raised when a contract condition cannot be evaluated.

    This error is raised when a precondition or postcondition lambda/function
    cannot be evaluated due to signature mismatches, type errors, or other
    evaluation failures. This ensures contracts fail closed rather than silently
    passing when the condition cannot be checked.
    """

    def __init__(self, message: str, value=None) -> None:
        if not isinstance(message, str) or not message.strip():
            raise ValueError(
                f"message must be provided as a non-empty string (got: {message!r})"
            )
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
