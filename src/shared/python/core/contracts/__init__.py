"""Design by Contract (DbC) decorators and utilities.

This package provides formal contract enforcement for the UpstreamDrift platform,
implementing preconditions, postconditions, and class invariants.

Enforcement Levels (controlled via ``DBC_LEVEL`` environment variable):
  - ``enforce`` (default): Raise contract violation errors on failure.
  - ``warn``: Log violations at WARNING level but do not raise.
  - ``off``: Skip all contract checks (maximum performance).

Sub-modules:
  - ``level``: Contract enforcement level configuration.
  - ``exceptions``: Contract violation exception hierarchy.
  - ``primitives``: Core ``require()`` / ``ensure()`` functions.
  - ``decorators``: ``@precondition``, ``@postcondition``, ``@require_state``, etc.
  - ``validators``: Numeric / array validation helpers.
  - ``invariants``: ``ContractChecker`` mixin and ``@invariant`` class decorator.

All public names are re-exported here for backward compatibility.
"""

from .decorators import (
    finite_result,
    non_empty_result,
    postcondition,
    precondition,
    require_state,
    requires_initialized,
    requires_model_loaded,
)
from .exceptions import (
    ContractViolationError,
    InvariantError,
    PostconditionError,
    PreconditionError,
    StateError,
)
from .invariants import (
    ContractChecker,
    invariant,
    invariant_checked,
)
from .level import (
    CONTRACTS_ENABLED,
    DBC_LEVEL,
    ContractLevel,
    _resolve_contract_level,
    contracts_enabled,
    disable_contracts,
    enable_contracts,
    get_contract_level,
    set_contract_level,
)
from .primitives import (
    ensure,
    require,
)
from .validators import (
    check_finite,
    check_non_negative,
    check_positive,
    check_positive_definite,
    check_shape,
    check_symmetric,
)

__all__ = [
    # Level
    "CONTRACTS_ENABLED",
    "ContractLevel",
    "DBC_LEVEL",
    "_resolve_contract_level",
    "contracts_enabled",
    "disable_contracts",
    "enable_contracts",
    "get_contract_level",
    "set_contract_level",
    # Exceptions
    "ContractViolationError",
    "InvariantError",
    "PostconditionError",
    "PreconditionError",
    "StateError",
    # Primitives
    "ensure",
    "require",
    # Decorators
    "finite_result",
    "non_empty_result",
    "postcondition",
    "precondition",
    "require_state",
    "requires_initialized",
    "requires_model_loaded",
    # Validators
    "check_finite",
    "check_non_negative",
    "check_positive",
    "check_positive_definite",
    "check_shape",
    "check_symmetric",
    # Invariants
    "ContractChecker",
    "invariant",
    "invariant_checked",
]
