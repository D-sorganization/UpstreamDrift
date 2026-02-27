"""Design by Contract decorators for Humanoid Character Builder.

This module re-exports from the canonical contracts implementation
at ``src/shared/python/contracts.py`` for backward compatibility.

All contract enforcement, decorators, and exceptions are defined
in the single source of truth.
"""

from __future__ import annotations

from contracts import (  # noqa: F401
    ContractViolationError,
    postcondition,
    precondition,
)
from contracts import (
    class_invariant as invariant,
)

__all__ = [
    "ContractViolationError",
    "invariant",
    "postcondition",
    "precondition",
]
