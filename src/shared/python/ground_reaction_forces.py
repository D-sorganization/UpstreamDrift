"""Backward compatibility shim - module moved to physics.ground_reaction_forces."""
import sys as _sys

from .physics import ground_reaction_forces as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
