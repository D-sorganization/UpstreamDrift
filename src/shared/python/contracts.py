# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

"""Design by Contract (DbC) enforcement for the Tools platform.

This is the **canonical** DbC implementation for ``src/shared/python``.
Domain-specific shims in sub-packages (e.g.
``humanoid_character_builder.contracts`` and
``model_generation.core.contracts``) re-export from this module for
backward compatibility.  All new code should import directly from here::

    from src.shared.python.contracts import require, ensure, precondition

Relationship to ``src/shared/python/core/contracts/``:
    That package is a separately evolved contracts implementation used by
    the ``src/shared/python/core/`` sub-system.  The two implementations
    are intentionally distinct; this module is the authoritative source
    for the wider platform.

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

import sys

from src.shared.python._contracts_decorators import (
    F,
    _check_class_invariant,
    _evaluate_precondition,
    _wrap_method_with_invariant,
    class_invariant,
    contract,
    postcondition,
    precondition,
)
from src.shared.python._contracts_exceptions import (
    _VIOLATION_CLASSES,
    ContractEvaluationError,
    ContractViolationError,
    InvariantError,
    PostconditionError,
    PreconditionError,
    _handle_violation,
)
from src.shared.python._contracts_invariant_mixin import (
    ContractChecker,
    invariant_checked,
)
from src.shared.python._contracts_level import (
    CONTRACTS_ENABLED,
    DBC_LEVEL,
    ContractLevel,
    _ContractState,
    _resolve_contract_level,
    get_contract_level,
    set_contract_level,
)
from src.shared.python._contracts_primitives import ensure, invariant, require
from src.shared.python._contracts_validators import (
    check_non_negative,
    check_positive,
    check_pressure,
    check_range,
    check_temperature,
    ensure_valid_result,
    has_finite_elements,
    is_non_negative,
    is_positive,
    is_valid_result,
    require_finite,
    require_positive,
    require_unit_vector,
    set_contracts_enabled,
)

__all__ = [
    "F",
    "CONTRACTS_ENABLED",
    "ContractChecker",
    "ContractLevel",
    "ContractViolationError",
    "DBC_LEVEL",
    "InvariantError",
    "PostconditionError",
    "PreconditionError",
    "_VIOLATION_CLASSES",
    "_ContractState",
    "_check_class_invariant",
    "_evaluate_precondition",
    "_handle_violation",
    "_resolve_contract_level",
    "_wrap_method_with_invariant",
    "check_non_negative",
    "check_positive",
    "check_pressure",
    "check_range",
    "check_temperature",
    "class_invariant",
    "contract",
    "ensure",
    "ensure_valid_result",
    "get_contract_level",
    "has_finite_elements",
    "invariant",
    "invariant_checked",
    "is_non_negative",
    "is_positive",
    "is_valid_result",
    "postcondition",
    "precondition",
    "require",
    "require_finite",
    "require_positive",
    "require_unit_vector",
    "set_contract_level",
    "set_contracts_enabled",
    "ContractEvaluationError",
]

# Backward-compatibility module aliases.
#
# ``src/shared/python`` and its parent ``src`` are both on ``sys.path`` (see
# ``[tool.pytest.ini_options] pythonpath`` in ``pyproject.toml``), so this one
# physical file is reachable under three dotted names:
#
#   * ``src.shared.python.contracts``  -- the canonical name (this module).
#   * ``shared.python.contracts``      -- legacy, deprecated.
#   * ``contracts``                    -- legacy top-level, deprecated.
#
# Python treats each name as a *distinct* module object unless they are
# explicitly unified.  Without unification, exception classes such as
# ``PreconditionError`` would have two identities and ``pytest.raises`` /
# ``except`` across import boundaries would silently fail on Python 3.11+.
#
# The previous implementation mutated ``sys.modules`` with a blind ``for`` loop
# (including a self-referential entry for the canonical name).  This replaces it
# with explicit, static registrations of the *legacy* aliases only.  New code
# must import from ``src.shared.python.contracts``; the legacy names remain
# importable for backward compatibility but should be considered deprecated.
sys.modules["contracts"] = sys.modules[__name__]
sys.modules["shared.python.contracts"] = sys.modules[__name__]
