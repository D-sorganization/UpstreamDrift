"""Design by Contract decorators for Humanoid Character Builder.

This module re-exports from the canonical contracts implementation
at ``src/shared/python/contracts.py`` for backward compatibility.

All contract enforcement, decorators, and exceptions are defined
in the single source of truth.
"""

from __future__ import annotations

from src.shared.python.contracts import (
    ContractViolationError,
    postcondition,
    precondition,
)
from src.shared.python.contracts import (
    class_invariant as invariant,
)

__all__ = [
    "ContractViolationError",
    "invariant",
    "postcondition",
    "precondition",
]
