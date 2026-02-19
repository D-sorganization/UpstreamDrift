"""Class invariant mixin and class-level invariant decorator.

Provides ``ContractChecker`` mixin for runtime invariant verification
and the ``@invariant`` class decorator for declarative invariants.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar, cast

from src.shared.python.logging_pkg.logging_config import get_logger

from .exceptions import InvariantError
from .level import ContractLevel

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")


class ContractChecker:
    """Mixin class providing invariant checking for physics engines.

    Subclasses should override _get_invariants() to define their invariants.

    Example:
        class MyEngine(ContractChecker, BasePhysicsEngine):
            def _get_invariants(self) -> list[tuple[Callable[[], bool], str]]:
                return [
                    (lambda: self.mass > 0, "mass must be positive"),
                    (lambda: self._is_initialized implies self.model is not None,
                     "initialized engine must have model"),
                ]
    """

    def _get_invariants(self) -> list[tuple[Callable[[], bool], str]]:
        """Return list of (condition, message) tuples for invariants.

        Override this method to define class invariants.

        Returns:
            List of (condition_callable, error_message) tuples.
        """
        return []

    def verify_invariants(self) -> bool:
        """Verify all class invariants hold.

        Returns:
            True if all invariants are satisfied.

        Raises:
            InvariantError: If any invariant is violated.
        """
        from .level import DBC_LEVEL  # re-import to capture runtime changes

        if DBC_LEVEL == ContractLevel.OFF:
            return True

        for condition, message in self._get_invariants():
            try:
                if not condition():
                    if DBC_LEVEL == ContractLevel.ENFORCE:
                        raise InvariantError(
                            message,
                            class_name=self.__class__.__name__,
                        )
                    logger.warning(
                        "[DbC invariant] %s: %s",
                        self.__class__.__name__,
                        message,
                    )
            except InvariantError:
                raise
            except (RuntimeError, TypeError, ValueError) as e:
                if DBC_LEVEL == ContractLevel.ENFORCE:
                    raise InvariantError(
                        f"Failed to evaluate invariant: {e}",
                        class_name=self.__class__.__name__,
                    ) from e

        return True

    def _check_invariants_after(self, method_name: str) -> None:
        """Check invariants after a method call.

        Args:
            method_name: Name of the method that was just called.

        Raises:
            InvariantError: If any invariant is violated.
        """
        from .level import DBC_LEVEL  # re-import to capture runtime changes

        if DBC_LEVEL == ContractLevel.OFF:
            return

        for condition, message in self._get_invariants():
            try:
                if not condition():
                    if DBC_LEVEL == ContractLevel.ENFORCE:
                        raise InvariantError(
                            message,
                            class_name=self.__class__.__name__,
                            method_name=method_name,
                        )
                    logger.warning(
                        "[DbC invariant] %s.%s: %s",
                        self.__class__.__name__,
                        method_name,
                        message,
                    )
            except InvariantError:
                raise
            except (RuntimeError, TypeError, ValueError) as e:
                if DBC_LEVEL == ContractLevel.ENFORCE:
                    raise InvariantError(
                        f"Failed to evaluate invariant after {method_name}: {e}",
                        class_name=self.__class__.__name__,
                        method_name=method_name,
                    ) from e


def invariant_checked(func: F) -> F:
    """Decorator to check class invariants after method execution.

    Use this on methods that modify state to ensure invariants are maintained.

    Args:
        func: Method to wrap.

    Returns:
        Wrapped method that checks invariants after execution.

    Example:
        class MyEngine(ContractChecker):
            @invariant_checked
            def set_mass(self, mass: float) -> None:
                self._mass = mass
    """
    from .level import DBC_LEVEL  # defer to capture runtime changes

    if DBC_LEVEL == ContractLevel.OFF:
        return func

    @functools.wraps(func)
    def wrapper(self: ContractChecker, *args: Any, **kwargs: Any) -> Any:
        """Execute the method and verify class invariants afterward."""
        result = func(self, *args, **kwargs)
        self._check_invariants_after(func.__name__)
        return result

    return cast(F, wrapper)


def invariant(
    condition: Callable[[Any], bool],
    message: str = "Invariant violated",
) -> Callable[[type[T]], type[T]]:
    """Class decorator to declare a class-level invariant.

    The *condition* callable receives ``self`` and must return ``True``
    when the invariant holds.  The check is executed after ``__init__``
    completes; if the invariant is violated, an ``InvariantError`` is
    raised (or a warning is logged, depending on ``DBC_LEVEL``).

    Multiple ``@invariant`` decorators can be stacked on a single class.
    They are evaluated in top-to-bottom (outermost-first) order.

    Args:
        condition: A callable ``(self) -> bool`` expressing the invariant.
        message: Human-readable description of the invariant.

    Returns:
        A class decorator that wraps ``__init__`` with the check.

    Example::

        @invariant(lambda self: self.timestep > 0, "Timestep must be positive")
        @invariant(lambda self: self.model is not None, "Model must be loaded")
        class MyEngine:
            def __init__(self, timestep: float, model: Any) -> None:
                self.timestep = timestep
                self.model = model
    """
    from .level import DBC_LEVEL  # defer to capture runtime changes

    def decorator(cls: type[T]) -> type[T]:
        """Wrap the class __init__ to enforce the invariant after construction."""
        if DBC_LEVEL == ContractLevel.OFF:
            return cls

        original_init = cls.__init__  # type: ignore[misc]

        @functools.wraps(original_init)
        def new_init(self: Any, *args: Any, **kwargs: Any) -> None:
            """Run the original __init__ and check the class invariant."""
            original_init(self, *args, **kwargs)
            try:
                result = condition(self)
            except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
                if DBC_LEVEL == ContractLevel.ENFORCE:
                    raise InvariantError(
                        f"Failed to evaluate invariant: {exc}",
                        class_name=cls.__name__,
                    ) from exc
                logger.warning(
                    "[DbC invariant] %s: failed to evaluate - %s",
                    cls.__name__,
                    exc,
                )
                return

            if not result:
                if DBC_LEVEL == ContractLevel.ENFORCE:
                    raise InvariantError(message, class_name=cls.__name__)
                logger.warning(
                    "[DbC invariant] %s: %s",
                    cls.__name__,
                    message,
                )

        cls.__init__ = new_init  # type: ignore[method-assign]
        return cls

    return decorator
