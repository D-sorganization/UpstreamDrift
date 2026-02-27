"""Design by Contract Decorators for URDF Model Generation.

This module re-exports from the canonical contracts implementation
at ``src/shared/python/contracts.py`` for backward compatibility.

All contract enforcement, decorators, exceptions, convenience functions,
and condition predicates are defined in the single source of truth.
"""

from __future__ import annotations

from contracts import (  # noqa: F401
    CONTRACTS_ENABLED,
    ContractViolationError,
    InvariantError,
    PostconditionError,
    PreconditionError,
    contract,
    ensure_valid_result,
    has_finite_elements,
    is_non_negative,
    is_positive,
    is_valid_result,
    postcondition,
    precondition,
    require_finite,
    require_positive,
    require_unit_vector,
    set_contracts_enabled,
)
from contracts import (
    class_invariant as invariant,
)

# Backward-compatible alias: the old module exposed ``ContractViolation``
# as its base exception name.
ContractViolation = ContractViolationError

__all__ = [
    # Decorators
    "precondition",
    "postcondition",
    "contract",
    "invariant",
    # Exceptions
    "ContractViolation",
    "ContractViolationError",
    "PreconditionError",
    "PostconditionError",
    "InvariantError",
    # Convenience functions
    "require_positive",
    "require_finite",
    "require_unit_vector",
    "ensure_valid_result",
    "set_contracts_enabled",
    # Condition functions
    "is_positive",
    "is_non_negative",
    "is_valid_result",
    "has_finite_elements",
    # Global flag
    "CONTRACTS_ENABLED",
]
