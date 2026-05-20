from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any, TypeVar, cast

from src.shared.python._contracts_exceptions import (
    ContractEvaluationError,
    InvariantError,
    _handle_violation,
)
from src.shared.python._contracts_level import ContractLevel, _ContractState

F = TypeVar("F", bound=Callable[..., Any])


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

    Raises:
        ContractEvaluationError: If the condition cannot be evaluated due to
            signature mismatches, type errors, or other evaluation failures.
    """
    if condition is None:
        raise ValueError("condition must be provided")

    # Try name-based binding first
    try:
        func_sig = inspect.signature(func)
        bound = func_sig.bind(*args, **kwargs)
        bound.apply_defaults()
        all_arguments: dict[str, Any] = dict(bound.arguments)

        cond_sig = inspect.signature(condition)
        cond_params = set(cond_sig.parameters)

        if cond_params and cond_params <= set(all_arguments):
            call_args = {name: all_arguments[name] for name in cond_params}
            return bool(condition(**call_args))
    except (TypeError, ValueError) as exc:
        raise ContractEvaluationError(
            f"Failed to bind arguments for precondition of {func.__qualname__}: {exc}"
        ) from exc

    # Fall back to positional call
    try:
        return bool(condition(*args, **kwargs))
    except TypeError as exc:
        raise ContractEvaluationError(
            f"Failed to evaluate precondition for {func.__qualname__}: {exc}"
        ) from exc


def precondition(
    condition: Callable[..., bool],
    message: str = "Precondition failed",
) -> Callable[[F], F]:
    """Decorator to enforce a precondition on a function or method.

    The *condition* callable may accept either the same arguments as the
    decorated function, or a subset matched by parameter name.
    """

    if condition is None:
        raise ValueError("condition must be provided")

    def decorator(func: F) -> F:
        if _ContractState.level == ContractLevel.OFF:
            return func

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = _evaluate_precondition(condition, func, args, kwargs)

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

    if condition is None:
        raise ValueError("condition must be provided")

    def decorator(func: F) -> F:
        if _ContractState.level == ContractLevel.OFF:
            return func

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)

            try:
                check = condition(result)
            except (
                TypeError,
                ValueError,
                ZeroDivisionError,
                AttributeError,
                KeyError,
                ArithmeticError,
            ) as exc:
                raise ContractEvaluationError(
                    f"Failed to evaluate postcondition for {func.__qualname__}: {exc}"
                ) from exc

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

    if pre_msg is None:
        raise ValueError("pre_msg must be provided")

    def decorator(func: F) -> F:
        result_func = func
        if post is not None:
            result_func = postcondition(post, post_msg)(result_func)
        if pre is not None:
            result_func = precondition(pre, pre_msg)(result_func)
        return result_func

    return decorator


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

    if orig_method is None:
        raise ValueError("orig_method must be provided")

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

    if condition is None:
        raise ValueError("condition must be provided")

    def class_decorator(cls: type) -> type:
        if _ContractState.level == ContractLevel.OFF:
            return cls

        original_init = cls.__init__  # type: ignore[misc]

        @functools.wraps(original_init)
        def new_init(self: Any, *args: Any, **kwargs: Any) -> None:
            original_init(self, *args, **kwargs)
            _check_class_invariant(self, condition, message, "after __init__")

        cls.__init__ = new_init  # type: ignore[misc]

        for name, method in inspect.getmembers(cls, inspect.isfunction):
            if not name.startswith("_"):
                setattr(
                    cls,
                    name,
                    _wrap_method_with_invariant(method, name, condition, message),
                )

        return cls

    return class_decorator
