"""Backward compatibility shim - module moved to physics.equipment."""

import sys as _sys

from .physics import equipment as _real_module  # noqa: E402
from .physics.equipment import (  # noqa: F401
    CLUB_CONFIGS,
    get_club_config,
)

_sys.modules[__name__] = _real_module
