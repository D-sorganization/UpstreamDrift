from __future__ import annotations

from typing import Any

from shared.python._contracts_exceptions import PostconditionError, PreconditionError
from shared.python._contracts_level import ContractLevel, _ContractState
from shared.python._contracts_primitives import require


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


def set_contracts_enabled(enabled: bool) -> None:
    """Enable or disable contract checking globally.

    This is a convenience wrapper around :func:`set_contract_level` that
    maps ``True`` to ``ENFORCE`` and ``False`` to ``OFF``, preserving
    backward compatibility with satellite modules.
    """
    from shared.python._contracts_level import set_contract_level

    set_contract_level(ContractLevel.ENFORCE if enabled else ContractLevel.OFF)


def require_positive(value: float, name: str = "value") -> None:
    """Require that *value* is strictly positive.

    Raises:
        PreconditionError: If *value* ``<= 0``.
    """
    if _ContractState.level == ContractLevel.OFF:
        return
    if value <= 0:
        raise PreconditionError(f"{name} must be positive (got {value})")


def require_finite(array: Any, name: str = "array") -> None:
    """Require all elements of *array* to be finite (no NaN / Inf).

    Raises:
        PreconditionError: If any element is NaN or Inf.
    """
    import numpy as np

    if _ContractState.level == ContractLevel.OFF:
        return
    if not np.all(np.isfinite(array)):
        raise PreconditionError(f"{name} contains NaN or Inf values")


def require_unit_vector(vector: Any, name: str = "vector", tol: float = 1e-6) -> None:
    """Require *vector* to have unit length.

    Raises:
        PreconditionError: If the norm deviates from 1.0 by more than *tol*.
    """
    import numpy as np

    if _ContractState.level == ContractLevel.OFF:
        return
    norm = np.linalg.norm(vector)
    if abs(norm - 1.0) > tol:
        raise PreconditionError(f"{name} must be a unit vector (norm = {norm})")


def ensure_valid_result(result: Any) -> None:
    """Ensure a ``ValidationResult``-like object is valid.

    Raises:
        PostconditionError: If ``result.is_valid`` is falsy.
    """
    if _ContractState.level == ContractLevel.OFF:
        return
    if not result.is_valid:
        errors = "; ".join(result.get_error_messages())
        raise PostconditionError(f"Validation failed: {errors}")


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
