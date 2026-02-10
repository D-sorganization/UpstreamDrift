"""Backward compatibility shim - module moved to core.contracts."""

import sys as _sys

from .core import contracts as _real_module  # noqa: E402
from .core.contracts import (  # noqa: F401
    CONTRACTS_ENABLED,
    ContractChecker,
    ContractViolationError,
    F,
    InvariantError,
    PostconditionError,
    PreconditionError,
    StateError,
    T,
    check_finite,
    check_non_negative,
    check_positive,
    check_positive_definite,
    check_shape,
    check_symmetric,
    contracts_enabled,
    disable_contracts,
    enable_contracts,
    finite_result,
    invariant_checked,
    logger,
    non_empty_result,
    postcondition,
    precondition,
    require_state,
    requires_initialized,
    requires_model_loaded,
)

_sys.modules[__name__] = _real_module
