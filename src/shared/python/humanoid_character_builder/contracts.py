"""Design by Contract decorators for Humanoid Character Builder.

This module re-exports from the canonical contracts implementation
at ``src/shared/python/contracts.py`` for backward compatibility.

All contract enforcement, decorators, and exceptions are defined
in the single source of truth.

.. note::
    Import directly from ``src.shared.python.contracts`` for new code.
    This shim exists only for backward compatibility within the
    ``humanoid_character_builder`` package.
"""

from __future__ import annotations

from src.shared.python.contracts import (  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401
    ContractViolationError,
)
from src.shared.python.contracts import class_invariant as invariant
from src.shared.python.contracts import (  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401
    postcondition,
    precondition,
)

__all__ = [
    "ContractViolationError",
    "invariant",
    "postcondition",
    "precondition",
]
