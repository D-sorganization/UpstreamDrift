"""Decorator-based contract enforcement.

Provides ``@precondition``, ``@postcondition``, ``@require_state``,
``@finite_result``, ``@non_empty_result``, and convenience aliases.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any, TypeVar, cast

from src.shared.python.logging_pkg.logging_config import get_logger

from .level import ContractLevel
from .primitives import _handle_violation
from .validators import check_finite

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def precondition(
    condition: Callable[..., bool],
    message: str = "Precondition failed",
    enabled: bool = True,
) -> Callable[[F], F]:
    """Decorator to enforce a precondition on a function or method.

    The condition function receives the same arguments as the decorated function.
    For methods, 'self' is available as the first argument.

    Args:
        condition: A callable that returns True if precondition is satisfied.
                   Receives the same arguments as the decorated function.
        message: Error message if precondition fails.
        enabled: Whether this precondition is active (can be disabled for performance).

    Returns:
        Decorated function that checks precondition before execution.

    Raises:
        PreconditionError: If condition returns False.

    Example:
        @precondition(lambda self: self._is_initialized, "Engine must be initialized")
        def step(self, dt: float) -> None:
            ...

        @precondition(lambda x: x > 0, "x must be positive")
        def sqrt(x: float) -> float:
            return math.sqrt(x)
    """
    from .level import DBC_LEVEL  # defer to capture runtime changes

    def decorator(func: F) -> F:
        """Wrap the function with precondition checking logic."""
        if not enabled or DBC_LEVEL == ContractLevel.OFF:
            return func

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Execute precondition check before calling the wrapped function."""
            # Bind arguments to get named parameters
            sig = inspect.signature(func)
            try:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
            except TypeError:
                bound = None

            # Evaluate the condition
            try:
                result = condition(*args, **kwargs)
            except (RuntimeError, TypeError, ValueError) as e:
                _handle_violation(
                    "Precondition",
                    f"Failed to evaluate precondition: {e}",
                    function_name=func.__qualname__,
                )
                return func(*args, **kwargs)

            if not result:
                _handle_violation(
                    "Precondition",
                    message,
                    function_name=func.__qualname__,
                )

            return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


def postcondition(
    condition: Callable[[Any], bool],
    message: str = "Postcondition failed",
    enabled: bool = True,
) -> Callable[[F], F]:
    """Decorator to enforce a postcondition on a function's return value.

    The condition function receives the return value of the decorated function.

    Args:
        condition: A callable that receives the return value and returns True if satisfied.
        message: Error message if postcondition fails.
        enabled: Whether this postcondition is active.

    Returns:
        Decorated function that checks postcondition after execution.

    Raises:
        PostconditionError: If condition returns False.

    Example:
        @postcondition(lambda result: result is not None, "Result must not be None")
        def get_state(self) -> State:
            ...

        @postcondition(lambda arr: np.all(np.isfinite(arr)), "Result must be finite")
        def compute_acceleration(self) -> np.ndarray:
            ...
    """
    from .level import DBC_LEVEL  # defer to capture runtime changes

    def decorator(func: F) -> F:
        """Wrap the function with postcondition checking logic."""
        if not enabled or DBC_LEVEL == ContractLevel.OFF:
            return func

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Execute the wrapped function and verify its postcondition."""
            result = func(*args, **kwargs)

            # Evaluate the postcondition
            try:
                check_result = condition(result)
            except (RuntimeError, TypeError, ValueError) as e:
                _handle_violation(
                    "Postcondition",
                    f"Failed to evaluate postcondition: {e}",
                    function_name=func.__qualname__,
                    value=result,
                )
                return result

            if not check_result:
                _handle_violation(
                    "Postcondition",
                    message,
                    function_name=func.__qualname__,
                    value=result,
                )

            return result

        return cast(F, wrapper)

    return decorator


def require_state(
    state_check: Callable[[Any], bool],
    state_name: str,
    operation_desc: str | None = None,
) -> Callable[[F], F]:
    """Decorator to require a specific object state before method execution.

    This is a specialized precondition for state-dependent operations.

    Args:
        state_check: Callable that takes 'self' and returns True if state is valid.
        state_name: Human-readable name of the required state.
        operation_desc: Description of the operation being attempted.

    Returns:
        Decorated method that checks state before execution.

    Raises:
        StateError: If the required state is not met.

    Example:
        @require_state(lambda self: self._is_initialized, "initialized")
        def step(self, dt: float) -> None:
            ...
    """
    from .exceptions import StateError
    from .level import DBC_LEVEL  # defer to capture runtime changes

    def decorator(func: F) -> F:
        """Wrap the method with state validation logic."""
        if DBC_LEVEL == ContractLevel.OFF:
            return func

        @functools.wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            """Verify required object state before executing the wrapped method."""
            if not state_check(self):
                operation = operation_desc or func.__name__
                if DBC_LEVEL == ContractLevel.ENFORCE:
                    raise StateError(
                        f"Cannot perform '{operation}' - engine not {state_name}",
                        current_state="not " + state_name,
                        required_state=state_name,
                        operation=func.__qualname__,
                    )
                if DBC_LEVEL == ContractLevel.WARN:
                    logger.warning(
                        "[DbC State] Cannot perform '%s' - not %s",
                        operation,
                        state_name,
                    )
            return func(self, *args, **kwargs)

        return cast(F, wrapper)

    return decorator


# Convenience aliases for common patterns
requires_initialized = require_state(
    lambda self: getattr(self, "_is_initialized", False),
    "initialized",
)

requires_model_loaded = require_state(
    lambda self: getattr(self, "model", None) is not None,
    "model loaded",
)


def finite_result(func: F) -> F:
    """Decorator ensuring function returns finite values (no NaN/Inf).

    Args:
        func: Function returning np.ndarray.

    Returns:
        Wrapped function with postcondition check.
    """
    return postcondition(
        lambda result: result is None or check_finite(result),
        "Result contains NaN or Inf values",
    )(func)


def non_empty_result(func: F) -> F:
    """Decorator ensuring function returns non-empty array.

    Args:
        func: Function returning np.ndarray.

    Returns:
        Wrapped function with postcondition check.
    """
    return postcondition(
        lambda result: result is not None and result.size > 0,
        "Result is empty",
    )(func)
