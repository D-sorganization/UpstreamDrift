"""Unified Model Generation Package for URDF and Physics Simulation.

This package provides comprehensive tools for creating, editing, and
converting robot models in URDF and other formats.

Lazy-loading strategy (see issue #611):
    All heavy imports are deferred to first access via ``__getattr__``.
    This breaks the circular import chain:
        __init__ -> builders.base_builder -> core.contracts
        -> core.validation -> core.contracts
    Only lightweight constants are imported eagerly.

Refactored in issue #1696:
    The lazy-import dispatch table has been moved to ``_lazy_map.py`` and
    the convenience functions to ``_convenience.py`` to keep this file
    below 120 lines.
"""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.1.0"
__author__ = "Golf Modeling Suite"

# --- Only lightweight constants are imported eagerly ---
from src.shared.python.model_generation._convenience import (
    _PRESETS as _HUMANOID_PRESETS,
    quick_build,
    quick_urdf,
)
from src.shared.python.model_generation._lazy_map import LAZY_IMPORTS
from src.shared.python.model_generation.core.constants import (
    DEFAULT_DENSITY_KG_M3,
    DEFAULT_HEIGHT_M,
    DEFAULT_INERTIA_KG_M2,
    DEFAULT_MASS_KG,
    GRAVITY_M_S2,
)

__all__ = [
    # Version
    "__version__",
    # Constants
    "GRAVITY_M_S2",
    "DEFAULT_DENSITY_KG_M3",
    "DEFAULT_INERTIA_KG_M2",
    "DEFAULT_HEIGHT_M",
    "DEFAULT_MASS_KG",
    # Convenience functions
    "quick_urdf",
    "quick_build",
    # All lazy-loaded names
    *LAZY_IMPORTS.keys(),
]


def __getattr__(name: str) -> Any:
    """Lazy-load attributes on first access (see issue #611)."""
    if name in LAZY_IMPORTS:
        module_path, attr_name = LAZY_IMPORTS[name]
        # nosemgrep: python.lang.security.audit.non-literal-import.non-literal-import
        module = importlib.import_module(module_path)
        value = getattr(module, attr_name)
        # Cache on the module so subsequent accesses are fast
        globals()[name] = value
        return value
    raise AttributeError(f"module 'model_generation' has no attribute {name!r}")
