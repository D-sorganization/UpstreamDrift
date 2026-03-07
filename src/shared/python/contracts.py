"""Design by Contract (DbC) enforcement for the Tools platform.

This module provides lightweight helpers and decorators for enforcing
pre-conditions, post-conditions, and invariants at runtime.

Enforcement Levels (controlled via ``DBC_LEVEL`` environment variable):
  - ``enforce`` (default): Raise ``ContractViolationError`` on failure.
  - ``warn``: Log violations at WARNING level but do not raise.
  - ``off``: Skip all contract checks (maximum performance).

Usage (function-call style)::

    from src.shared.python.contracts import require, ensure

    def calculate_pressure_drop(flow_rate: float, diameter: float) -> float:
        require(flow_rate > 0, "flow_rate must be positive", flow_rate)
        require(diameter > 0, "diameter must be positive", diameter)
        result = _compute(flow_rate, diameter)
        ensure(result >= 0, "pressure drop must be non-negative", result)
        return result

Usage (decorator style)::

    from src.shared.python.contracts import precondition, postcondition

    @precondition(lambda self, t: t > 0, "temperature must be positive")
    @postcondition(lambda r: r >= 0, "result must be non-negative")
    def compute_enthalpy(self, t: float) -> float:
        ...
"""

from __future__ import annotations

import enum
import functools
import inspect
import logging
import os
from collections.abc import Callable
from typing import Any, TypeVar, cast

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# ─── Contract Enforcement Level ────────────────────────────────


class ContractLevel(enum.Enum):
    """Tri-state enforcement level for Design by Contract checks."""

    OFF = "off"
    WARN = "warn"
    ENFORCE = "enforce"


def _resolve_contract_level() -> ContractLevel:
    """Determine the contract level from environment."""
    env_val = os.environ.get("DBC_LEVEL", "").lower().strip()
    if env_val in ("off", "warn", "enforce"):
        return ContractLevel(env_val)
    return ContractLevel.ENFORCE if __debug__ else ContractLevel.OFF


# Contract state holder (avoids mutable globals + global keyword)
class _ContractState:
    level: ContractLevel = _resolve_contract_level()

    @classmethod  # type: ignore[misc]  # @classmethod+@property deprecated in 3.12
    @property
    def enabled(cls) -> bool:
        return cls.level != ContractLevel.OFF


# Module-level aliases for backward compatibility (read via property delegation)
DBC_LEVEL: ContractLevel = _ContractState.level
CONTRACTS_ENABLED: bool = _ContractState.level != ContractLevel.OFF


def set_contract_level(level: ContractLevel) -> None:
    """Set the global contract enforcement level at runtime."""
    import sys

    _ContractState.level = level
    # Update module-level aliases so existing references see the new values
    current_module = sys.modules[__name__]
    current_module.DBC_LEVEL = level  # type: ignore[attr-defined]
    current_module.CONTRACTS_ENABLED = level != ContractLevel.OFF  # type: ignore[attr-defined]
    logger.info("Contract enforcement level set to %s", level.value)


def get_contract_level() -> ContractLevel:
    """Return the current global contract enforcement level."""
    return _ContractState.level


# ─── Exception Hierarchy ───────────────────────────────────────


class ContractViolationError(AssertionError, ValueError):
    """Base exception for contract violations."""

    def __init__(
        self,
        condition_type: str,
        message: str,
        value: Any = None,
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

    def __init__(self, message: str, value: Any = None) -> None:
        super().__init__("pre-condition", message, value)


class PostconditionError(ContractViolationError):
    """Raised when a post-condition is violated."""

    def __init__(self, message: str, value: Any = None) -> None:
        super().__init__("post-condition", message, value)


class InvariantError(ContractViolationError):
    """Raised when a class or loop invariant is violated."""

    def __init__(self, message: str, value: Any = None) -> None:
        super().__init__("invariant", message, value)


# ─── Core Contract Primitives ─────────────────────────────────


_VIOLATION_CLASSES: dict[str, type[ContractViolationError]] = {
    "pre-condition": PreconditionError,
    "post-condition": PostconditionError,
    "invariant": InvariantError,
}


def _handle_violation(
    condition_type: str,
    message: str,
    value: Any = None,
) -> None:
    """Handle a contract violation according to the current DBC_LEVEL."""
    if DBC_LEVEL == ContractLevel.ENFORCE:
        exc_cls = _VIOLATION_CLASSES.get(condition_type, ContractViolationError)
        raise exc_cls(message, value)
    if DBC_LEVEL == ContractLevel.WARN:
        detail = f"[DbC {condition_type}] {message}"
        if value is not None:
            detail += f" (got: {value!r})"
        logger.warning(detail)


def require(condition: bool, message: str, value: Any = None) -> None:
    """Assert a pre-condition at function entry."""
    if DBC_LEVEL == ContractLevel.OFF:
        return
    if not condition:
        _handle_violation("pre-condition", message, value)


def ensure(condition: bool, message: str, value: Any = None) -> None:
    """Assert a post-condition before function return."""
    if DBC_LEVEL == ContractLevel.OFF:
        return
    if not condition:
        _handle_violation("post-condition", message, value)


def invariant(condition: bool, message: str, value: Any = None) -> None:
    """Assert a class or loop invariant."""
    if DBC_LEVEL == ContractLevel.OFF:
        return
    if not condition:
        _handle_violation("invariant", message, value)


# ─── Decorator-Based Contracts ─────────────────────────────────


def _evaluate_precondition(
    condition: Callable[..., bool],
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> bool:
    """Try to evaluate a precondition, using argument-name binding.

    Prefers name-based binding when the condition only accepts a subset of
    the decorated function's parameters (e.g. ``lambda gender_factor: ...``
    should receive ``gender_factor`` by name, not the first positional arg).
    Falls back to positional call only when the condition accepts all args.
    """
    # Always try name-based binding first using the decorated function's sig
    try:
        func_sig = inspect.signature(func)
        bound = func_sig.bind(*args, **kwargs)
        bound.apply_defaults()
        all_arguments: dict[str, Any] = dict(bound.arguments)

        cond_sig = inspect.signature(condition)
        cond_params = set(cond_sig.parameters)

        # If all condition params are known function params, bind by name
        if cond_params and cond_params <= set(all_arguments):
            call_args = {name: all_arguments[name] for name in cond_params}
            return bool(condition(**call_args))
    except (TypeError, ValueError):
        pass

    # Fallback: call with same positional args (works when condition mirrors func)
    try:
        return bool(condition(*args, **kwargs))
    except TypeError:
        pass

    return True  # Cannot evaluate — let the function proceed


def precondition(
    condition: Callable[..., bool],
    message: str = "Precondition failed",
) -> Callable[[F], F]:
    """Decorator to enforce a precondition on a function or method.

    The *condition* callable may accept either the same arguments as the
    decorated function, or a subset matched by parameter name.
    """

    def decorator(func: F) -> F:
        if DBC_LEVEL == ContractLevel.OFF:
            return func

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                result = _evaluate_precondition(condition, func, args, kwargs)
            except (TypeError, ValueError) as exc:
                _handle_violation(
                    "pre-condition",
                    f"Failed to evaluate precondition for {func.__qualname__}: {exc}",
                )
                return func(*args, **kwargs)

            if not result:
                _handle_violation("pre-condition", message)

            return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


def postcondition(
    condition: Callable[[Any], bool],
    message: str = "Postcondition failed",
) -> Callable[[F], F]:
    """Decorator to enforce a postcondition on a function's return value."""

    def decorator(func: F) -> F:
        if DBC_LEVEL == ContractLevel.OFF:
            return func

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)

            try:
                check = condition(result)
            except (TypeError, ValueError) as exc:
                _handle_violation(
                    "post-condition",
                    f"Failed to evaluate postcondition for {func.__qualname__}: {exc}",
                )
                return result

            if not check:
                _handle_violation("post-condition", message, result)

            return result

        return cast(F, wrapper)

    return decorator


def contract(
    pre: Callable[..., bool] | None = None,
    post: Callable[[Any], bool] | None = None,
    pre_msg: str = "Precondition violated",
    post_msg: str = "Postcondition violated",
) -> Callable[[F], F]:
    """Combined precondition and postcondition decorator.

    Args:
        pre: Precondition function (receives same args as decorated function).
        post: Postcondition function (receives return value).
        pre_msg: Precondition error message.
        post_msg: Postcondition error message.

    Example::

        @contract(
            pre=lambda x: x >= 0,
            post=lambda result: result >= 0,
            pre_msg="Input must be non-negative",
            post_msg="Output must be non-negative",
        )
        def sqrt(x: float) -> float:
            return x ** 0.5
    """

    def decorator(func: F) -> F:
        result_func = func
        if post is not None:
            result_func = postcondition(post, post_msg)(result_func)
        if pre is not None:
            result_func = precondition(pre, pre_msg)(result_func)
        return result_func

    return decorator


# ─── Class Invariant Decorator ─────────────────────────────────


def _check_class_invariant(
    instance: Any,
    condition: Callable[[Any], bool],
    message: str,
    context: str,
) -> None:
    """Evaluate a class invariant and raise on failure.

    Args:
        instance: The object whose invariant is being checked.
        condition: Callable that takes ``self`` and returns ``bool``.
        message: Human-readable invariant description.
        context: Where the check happened (e.g. ``"after __init__"``).

    Raises:
        InvariantError: If the condition fails or raises.
    """
    try:
        if not condition(instance):
            raise InvariantError(f"{message} ({context})")
    except InvariantError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError, ArithmeticError) as exc:
        raise InvariantError(
            f"Error checking invariant '{message}' {context}: {exc}"
        ) from exc


def _wrap_method_with_invariant(
    orig_method: Callable[..., Any],
    method_name: str,
    condition: Callable[[Any], bool],
    message: str,
) -> Callable[..., Any]:
    """Wrap a single method to check the class invariant after execution."""

    @functools.wraps(orig_method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = orig_method(self, *args, **kwargs)
        _check_class_invariant(self, condition, message, f"after {method_name}")
        return result

    return wrapper


def class_invariant(
    condition: Callable[[Any], bool],
    message: str = "Invariant violated",
) -> Callable[[type], type]:
    """Class decorator to check invariants after ``__init__`` and public methods.

    The *condition* callable receives ``self`` and must return ``True`` when
    the invariant holds.

    Args:
        condition: Callable that takes ``self`` and returns ``bool``.
        message: Error message when the invariant is violated.

    Example::

        @class_invariant(lambda self: self.count >= 0, "count must be non-negative")
        class Counter:
            def __init__(self) -> None:
                self.count = 0
            def decrement(self) -> None:
                self.count -= 1
    """

    def class_decorator(cls: type) -> type:
        if DBC_LEVEL == ContractLevel.OFF:
            return cls

        # Wrap __init__
        original_init = cls.__init__  # type: ignore[misc]

        @functools.wraps(original_init)
        def new_init(self: Any, *args: Any, **kwargs: Any) -> None:
            original_init(self, *args, **kwargs)
            _check_class_invariant(self, condition, message, "after __init__")

        cls.__init__ = new_init  # type: ignore[misc]

        # Wrap all public methods
        for name, method in inspect.getmembers(cls, inspect.isfunction):
            if not name.startswith("_"):
                setattr(
                    cls,
                    name,
                    _wrap_method_with_invariant(method, name, condition, message),
                )

        return cls

    return class_decorator


# ─── Class Invariant Mixin ─────────────────────────────────────


class ContractChecker:
    """Mixin providing class invariant checking.

    Subclasses override ``_get_invariants()`` to define their invariants.
    """

    def _get_invariants(self) -> list[tuple[Callable[[], bool], str]]:
        """Return list of (condition, message) tuples for invariants."""
        return []

    def verify_invariants(self) -> bool:
        """Verify all class invariants hold."""
        if DBC_LEVEL == ContractLevel.OFF:
            return True

        for condition_fn, message in self._get_invariants():
            try:
                if not condition_fn():
                    if DBC_LEVEL == ContractLevel.ENFORCE:
                        raise InvariantError(f"{self.__class__.__name__}: {message}")
                    logger.warning(
                        "[DbC invariant] %s: %s",
                        self.__class__.__name__,
                        message,
                    )
            except InvariantError:
                raise
            except (RuntimeError, TypeError, ValueError) as exc:
                if DBC_LEVEL == ContractLevel.ENFORCE:
                    raise InvariantError(
                        f"{self.__class__.__name__}: "
                        f"Failed to evaluate invariant: {exc}"
                    ) from exc

        return True


def invariant_checked(func: F) -> F:
    """Decorator to check class invariants after method execution."""
    if DBC_LEVEL == ContractLevel.OFF:
        return func

    @functools.wraps(func)
    def wrapper(self: ContractChecker, *args: Any, **kwargs: Any) -> Any:
        result = func(self, *args, **kwargs)
        self.verify_invariants()
        return result

    return cast(F, wrapper)


# ─── Domain Helpers ────────────────────────────────────────────


def check_positive(value: float, name: str = "value") -> None:
    """Assert that a numeric value is strictly positive."""
    require(value > 0, f"{name} must be positive", value)


def check_non_negative(value: float, name: str = "value") -> None:
    """Assert that a numeric value is non-negative."""
    require(value >= 0, f"{name} must be non-negative", value)


def check_range(
    value: float,
    low: float,
    high: float,
    name: str = "value",
) -> None:
    """Assert that a numeric value falls within [low, high]."""
    require(low <= value <= high, f"{name} must be in [{low}, {high}]", value)


def check_temperature(value: float, name: str = "temperature") -> None:
    """Assert that a temperature is physically reasonable (> 0 K)."""
    require(value > 0, f"{name} must be > 0 K", value)


def check_pressure(value: float, name: str = "pressure") -> None:
    """Assert that a pressure is physically reasonable (> 0)."""
    require(value > 0, f"{name} must be > 0", value)


# ─── Backward-Compatibility Helpers ───────────────────────────


def set_contracts_enabled(enabled: bool) -> None:
    """Enable or disable contract checking globally.

    This is a convenience wrapper around :func:`set_contract_level` that
    maps ``True`` to ``ENFORCE`` and ``False`` to ``OFF``, preserving
    backward compatibility with satellite modules.
    """
    set_contract_level(ContractLevel.ENFORCE if enabled else ContractLevel.OFF)


# ─── Convenience Validation Functions ─────────────────────────


def require_positive(value: float, name: str = "value") -> None:
    """Require that *value* is strictly positive.

    Raises:
        PreconditionError: If *value* ``<= 0``.
    """
    if not CONTRACTS_ENABLED:
        return
    if value <= 0:
        raise PreconditionError(f"{name} must be positive (got {value})")


def require_finite(array: Any, name: str = "array") -> None:
    """Require all elements of *array* to be finite (no NaN / Inf).

    Raises:
        PreconditionError: If any element is NaN or Inf.
    """
    import numpy as np

    if not CONTRACTS_ENABLED:
        return
    if not np.all(np.isfinite(array)):
        raise PreconditionError(f"{name} contains NaN or Inf values")


def require_unit_vector(vector: Any, name: str = "vector", tol: float = 1e-6) -> None:
    """Require *vector* to have unit length.

    Raises:
        PreconditionError: If the norm deviates from 1.0 by more than *tol*.
    """
    import numpy as np

    if not CONTRACTS_ENABLED:
        return
    norm = np.linalg.norm(vector)
    if abs(norm - 1.0) > tol:
        raise PreconditionError(f"{name} must be a unit vector (norm = {norm})")


def ensure_valid_result(result: Any) -> None:
    """Ensure a ``ValidationResult``-like object is valid.

    Raises:
        PostconditionError: If ``result.is_valid`` is falsy.
    """
    if not CONTRACTS_ENABLED:
        return
    if not result.is_valid:
        errors = "; ".join(result.get_error_messages())
        raise PostconditionError(f"Validation failed: {errors}")


# ─── Reusable Condition Predicates ────────────────────────────


def is_positive(value: float) -> bool:
    """Return ``True`` if *value* is strictly positive."""
    return value > 0


def is_non_negative(value: float) -> bool:
    """Return ``True`` if *value* is non-negative."""
    return value >= 0


def is_valid_result(result: Any) -> bool:
    """Return ``True`` if ``result.is_valid`` is truthy."""
    return bool(result.is_valid)


def has_finite_elements(array: Any) -> bool:
    """Return ``True`` if all elements of *array* are finite."""
    import numpy as np

    return bool(np.all(np.isfinite(array)))
