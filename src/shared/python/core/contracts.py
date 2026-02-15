"""Design by Contract (DbC) decorators and utilities.

This module provides formal contract enforcement for the UpstreamDrift platform,
implementing preconditions, postconditions, and class invariants.

Enforcement Levels (controlled via ``DBC_LEVEL`` environment variable):
  - ``enforce`` (default): Raise contract violation errors on failure.
  - ``warn``: Log violations at WARNING level but do not raise.
  - ``off``: Skip all contract checks (maximum performance).

The legacy ``CONTRACTS_ENABLED`` boolean is still available for backward
compatibility; it is derived from ``DBC_LEVEL``.

Design by Contract Principles:
- Preconditions: What must be true before a method executes
- Postconditions: What must be true after a method completes
- Invariants: What must always be true for an object's state

Usage (decorator style):

    from src.shared.python.core.contracts import precondition, postcondition

    @precondition(lambda self: self._is_initialized, "Engine must be initialized")
    @postcondition(lambda result: result.shape[0] > 0, "Result must be non-empty")
    def compute_acceleration(self) -> np.ndarray:
        ...

Usage (function-call style):

    from src.shared.python.core.contracts import require, ensure

    def step(self, dt: float) -> State:
        require(dt > 0, "dt must be positive", dt)
        result = self._integrate(dt)
        ensure(result.is_valid(), "result must be valid", result)
        return result

Class invariant mixin:

    from src.shared.python.core.contracts import ContractChecker

    class PhysicsBody(ContractChecker):
        def _get_invariants(self) -> list[tuple[Callable[[], bool], str]]:
            return [
                (lambda: self.mass > 0, "mass must be positive"),
            ]
"""

from __future__ import annotations

import enum
import functools
import inspect
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np

from src.shared.python.logging_pkg.logging_config import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")


# ─── Contract Enforcement Level ────────────────────────────────


class ContractLevel(enum.Enum):
    """Tri-state enforcement level for Design by Contract checks.

    Attributes:
        OFF: No checks (production hot path, maximum performance).
        WARN: Log violations at WARNING level without raising.
        ENFORCE: Raise contract violation errors on any failure.
    """

    OFF = "off"
    WARN = "warn"
    ENFORCE = "enforce"


def _resolve_contract_level() -> ContractLevel:
    """Determine the contract level from environment.

    Reads ``DBC_LEVEL`` environment variable.  Falls back to ``enforce``
    when ``__debug__`` is True (normal Python) or ``off`` when running
    with ``python -O``.
    """
    env_val = os.environ.get("DBC_LEVEL", "").lower().strip()
    if env_val in ("off", "warn", "enforce"):
        return ContractLevel(env_val)
    return ContractLevel.ENFORCE if __debug__ else ContractLevel.OFF


DBC_LEVEL: ContractLevel = _resolve_contract_level()

# Legacy compatibility flag (derived from DBC_LEVEL)
CONTRACTS_ENABLED = DBC_LEVEL != ContractLevel.OFF


def set_contract_level(level: ContractLevel) -> None:
    """Set the global contract enforcement level at runtime.

    Args:
        level: The desired enforcement level.
    """
    global DBC_LEVEL, CONTRACTS_ENABLED  # noqa: PLW0603
    DBC_LEVEL = level
    CONTRACTS_ENABLED = level != ContractLevel.OFF
    logger.info("Contract enforcement level set to %s", level.value)


def get_contract_level() -> ContractLevel:
    """Return the current global contract enforcement level."""
    return DBC_LEVEL


# ─── Exception Hierarchy ───────────────────────────────────────


class ContractViolationError(Exception):
    """Base exception for contract violations."""

    def __init__(
        self,
        contract_type: str,
        message: str,
        function_name: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.contract_type = contract_type
        self.function_name = function_name
        self.details = details or {}

        error_msg = f"{contract_type} violation"
        if function_name:
            error_msg += f" in {function_name}"
        error_msg += f": {message}"

        if details:
            error_msg += f"\n  Details: {details}"

        super().__init__(error_msg)


class PreconditionError(ContractViolationError):
    """Raised when a precondition is violated."""

    def __init__(
        self,
        message: str,
        function_name: str | None = None,
        parameter: str | None = None,
        value: Any = None,
    ) -> None:
        details = {}
        if parameter:
            details["parameter"] = parameter
        if value is not None:
            details["value"] = repr(value)

        super().__init__(
            contract_type="Precondition",
            message=message,
            function_name=function_name,
            details=details if details else None,
        )
        self.parameter = parameter
        self.value = value


class PostconditionError(ContractViolationError):
    """Raised when a postcondition is violated."""

    def __init__(
        self,
        message: str,
        function_name: str | None = None,
        result: Any = None,
    ) -> None:
        details: dict[str, Any] = {}
        if result is not None:
            # Avoid storing large arrays
            if isinstance(result, np.ndarray):
                details["result_shape"] = result.shape
                details["result_dtype"] = str(result.dtype)
            else:
                details["result"] = repr(result)[:100]

        super().__init__(
            contract_type="Postcondition",
            message=message,
            function_name=function_name,
            details=details if details else None,
        )
        self.result = result


class InvariantError(ContractViolationError):
    """Raised when a class invariant is violated."""

    def __init__(
        self,
        message: str,
        class_name: str | None = None,
        method_name: str | None = None,
    ) -> None:
        details = {}
        if class_name:
            details["class"] = class_name
        if method_name:
            details["after_method"] = method_name

        super().__init__(
            contract_type="Invariant",
            message=message,
            function_name=method_name,
            details=details if details else None,
        )
        self.class_name = class_name
        self.method_name = method_name


class StateError(ContractViolationError):
    """Raised when an operation is attempted in an invalid state."""

    def __init__(
        self,
        message: str,
        current_state: str | None = None,
        required_state: str | None = None,
        operation: str | None = None,
    ) -> None:
        details = {}
        if current_state:
            details["current_state"] = current_state
        if required_state:
            details["required_state"] = required_state

        super().__init__(
            contract_type="State",
            message=message,
            function_name=operation,
            details=details if details else None,
        )
        self.current_state = current_state
        self.required_state = required_state
        self.operation = operation


# ─── Core Contract Primitives (function-call style) ───────────


def _handle_violation(
    contract_type: str,
    message: str,
    function_name: str | None = None,
    value: Any = None,
) -> None:
    """Handle a contract violation according to the current DBC_LEVEL."""
    if DBC_LEVEL == ContractLevel.ENFORCE:
        if contract_type == "Precondition":
            raise PreconditionError(message, function_name=function_name, value=value)
        elif contract_type == "Postcondition":
            raise PostconditionError(message, function_name=function_name, result=value)
        elif contract_type == "Invariant":
            raise InvariantError(message)
        else:
            raise ContractViolationError(contract_type, message)
    elif DBC_LEVEL == ContractLevel.WARN:
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
    if DBC_LEVEL == ContractLevel.OFF:
        return
    if not condition:
        _handle_violation("Postcondition", message, value=value)


# ─── Decorator-Based Contracts ─────────────────────────────────


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
                elif DBC_LEVEL == ContractLevel.WARN:
                    logger.warning(
                        "[DbC State] Cannot perform '%s' - not %s",
                        operation,
                        state_name,
                    )
            return func(self, *args, **kwargs)

        return cast(F, wrapper)

    return decorator


# ─── Array/Numeric Helpers ─────────────────────────────────────


def check_finite(arr: np.ndarray | None) -> bool:
    """Check if array contains only finite values (no NaN or Inf).

    Args:
        arr: NumPy array to check.

    Returns:
        True if array is not None and all values are finite.
    """
    if arr is None:
        return False
    return bool(np.all(np.isfinite(arr)))


def check_shape(arr: np.ndarray | None, expected_shape: tuple[int, ...]) -> bool:
    """Check if array has expected shape.

    Args:
        arr: NumPy array to check.
        expected_shape: Expected shape tuple.

    Returns:
        True if array has the expected shape.
    """
    if arr is None:
        return False
    return arr.shape == expected_shape


def check_positive(value: float | int | np.ndarray) -> bool:
    """Check if value(s) are positive.

    Args:
        value: Scalar or array to check.

    Returns:
        True if all values are > 0.
    """
    if isinstance(value, np.ndarray):
        return bool(np.all(value > 0))
    return value > 0


def check_non_negative(value: float | int | np.ndarray) -> bool:
    """Check if value(s) are non-negative.

    Args:
        value: Scalar or array to check.

    Returns:
        True if all values are >= 0.
    """
    if isinstance(value, np.ndarray):
        return bool(np.all(value >= 0))
    return value >= 0


def check_symmetric(matrix: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if matrix is symmetric within tolerance.

    Args:
        matrix: 2D NumPy array to check.
        tol: Tolerance for symmetry check.

    Returns:
        True if matrix is symmetric.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False
    return bool(np.allclose(matrix, matrix.T, atol=tol))


def check_positive_definite(matrix: np.ndarray) -> bool:
    """Check if matrix is positive definite.

    Args:
        matrix: 2D symmetric NumPy array to check.

    Returns:
        True if all eigenvalues are positive.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False
    try:
        eigenvalues = np.linalg.eigvalsh(matrix)
        return bool(np.all(eigenvalues > 0))
    except np.linalg.LinAlgError:
        return False


# ─── Class Invariant Mixin ─────────────────────────────────────


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
                    else:
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
                    else:
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
    if DBC_LEVEL == ContractLevel.OFF:
        return func

    @functools.wraps(func)
    def wrapper(self: ContractChecker, *args: Any, **kwargs: Any) -> Any:
        """Execute the method and verify class invariants afterward."""
        result = func(self, *args, **kwargs)
        self._check_invariants_after(func.__name__)
        return result

    return cast(F, wrapper)


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


# Module-level functions for enabling/disabling contracts


def enable_contracts() -> None:
    """Enable contract checking globally (sets level to ENFORCE)."""
    set_contract_level(ContractLevel.ENFORCE)


def disable_contracts() -> None:
    """Disable contract checking globally (sets level to OFF)."""
    set_contract_level(ContractLevel.OFF)


def contracts_enabled() -> bool:
    """Check if contracts are currently enabled."""
    return CONTRACTS_ENABLED


# ─── Class-Level Invariant Decorator ──────────────────────────


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
                else:
                    logger.warning(
                        "[DbC invariant] %s: failed to evaluate – %s",
                        cls.__name__,
                        exc,
                    )
                return

            if not result:
                if DBC_LEVEL == ContractLevel.ENFORCE:
                    raise InvariantError(message, class_name=cls.__name__)
                else:
                    logger.warning(
                        "[DbC invariant] %s: %s",
                        cls.__name__,
                        message,
                    )

        cls.__init__ = new_init  # type: ignore[method-assign]
        return cls

    return decorator
