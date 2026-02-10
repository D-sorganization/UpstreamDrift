"""Design by Contract (DbC) decorators and utilities.

This module provides formal contract enforcement for the Golf Modeling Suite,
implementing preconditions, postconditions, and class invariants.

Design by Contract Principles:
- Preconditions: What must be true before a method executes
- Postconditions: What must be true after a method completes
- Invariants: What must always be true for an object's state

Usage:
    from src.shared.python.contracts import precondition, postcondition, invariant

    @precondition(lambda self: self._is_initialized, "Engine must be initialized")
    @postcondition(lambda result: result.shape[0] > 0, "Result must be non-empty")
    def compute_acceleration(self) -> np.ndarray:
        ...

    @invariant(lambda self: self.mass > 0, "Mass must be positive")
    class PhysicsBody:
        ...
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np

from src.shared.python.logging_config import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")

# Global flag to enable/disable contract checking (for performance in production)
CONTRACTS_ENABLED = True


class ContractViolationError(Exception):
    """Base exception for contract violations."""

    def __init__(
        self,
        contract_type: str,
        message: str,
        function_name: str | None = None,
        details: dict[str, Any] | None = None,
    ):
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
    ):
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
    ):
        details = {}
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
    ):
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
    ):
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
        if not enabled or not CONTRACTS_ENABLED:
            return func

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
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
                # If condition evaluation fails, report it
                raise PreconditionError(
                    f"Failed to evaluate precondition: {e}",
                    function_name=func.__qualname__,
                ) from e

            if not result:
                raise PreconditionError(
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
        if not enabled or not CONTRACTS_ENABLED:
            return func

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)

            # Evaluate the postcondition
            try:
                check_result = condition(result)
            except (RuntimeError, TypeError, ValueError) as e:
                raise PostconditionError(
                    f"Failed to evaluate postcondition: {e}",
                    function_name=func.__qualname__,
                    result=result,
                ) from e

            if not check_result:
                raise PostconditionError(
                    message,
                    function_name=func.__qualname__,
                    result=result,
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
        if not CONTRACTS_ENABLED:
            return func

        @functools.wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            if not state_check(self):
                operation = operation_desc or func.__name__
                raise StateError(
                    f"Cannot perform '{operation}' - engine not {state_name}",
                    current_state="not " + state_name,
                    required_state=state_name,
                    operation=func.__qualname__,
                )
            return func(self, *args, **kwargs)

        return cast(F, wrapper)

    return decorator


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
        if not CONTRACTS_ENABLED:
            return True

        for condition, message in self._get_invariants():
            try:
                if not condition():
                    raise InvariantError(
                        message,
                        class_name=self.__class__.__name__,
                    )
            except InvariantError:
                raise
            except (RuntimeError, TypeError, ValueError) as e:
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
        if not CONTRACTS_ENABLED:
            return

        for condition, message in self._get_invariants():
            try:
                if not condition():
                    raise InvariantError(
                        message,
                        class_name=self.__class__.__name__,
                        method_name=method_name,
                    )
            except InvariantError:
                raise
            except (RuntimeError, TypeError, ValueError) as e:
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
    if not CONTRACTS_ENABLED:
        return func

    @functools.wraps(func)
    def wrapper(self: ContractChecker, *args: Any, **kwargs: Any) -> Any:
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
    """Enable contract checking globally."""
    global CONTRACTS_ENABLED
    CONTRACTS_ENABLED = True
    logger.info("Contract checking enabled")


def disable_contracts() -> None:
    """Disable contract checking globally (for performance)."""
    global CONTRACTS_ENABLED
    CONTRACTS_ENABLED = False
    logger.info("Contract checking disabled")


def contracts_enabled() -> bool:
    """Check if contracts are currently enabled."""
    return CONTRACTS_ENABLED
