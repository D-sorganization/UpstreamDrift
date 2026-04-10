from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import Any, TypeVar, cast

from shared.python._contracts_exceptions import InvariantError
from shared.python._contracts_level import ContractLevel, _ContractState

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class ContractChecker:
    """Mixin providing class invariant checking.

    Subclasses override ``_get_invariants()`` to define their invariants.
    """

    def _get_invariants(self) -> list[tuple[Callable[[], bool], str]]:
        """Return list of (condition, message) tuples for invariants."""
        return []

    def verify_invariants(self) -> bool:
        """Verify all class invariants hold."""
        if _ContractState.level == ContractLevel.OFF:
            return True

        for condition_fn, message in self._get_invariants():
            try:
                if not condition_fn():
                    if _ContractState.level == ContractLevel.ENFORCE:
                        raise InvariantError(f"{self.__class__.__name__}: {message}")
                    logger.warning(
                        "[DbC invariant] %s: %s",
                        self.__class__.__name__,
                        message,
                    )
            except InvariantError:
                raise
            except (RuntimeError, TypeError, ValueError) as exc:
                if _ContractState.level == ContractLevel.ENFORCE:
                    raise InvariantError(
                        f"{self.__class__.__name__}: "
                        f"Failed to evaluate invariant: {exc}"
                    ) from exc

        return True


def invariant_checked(func: F) -> F:
    """Decorator to check class invariants after method execution."""
    if _ContractState.level == ContractLevel.OFF:
        return func

    @functools.wraps(func)
    def wrapper(self: ContractChecker, *args: Any, **kwargs: Any) -> Any:
        result = func(self, *args, **kwargs)
        self.verify_invariants()
        return result

    return cast(F, wrapper)
