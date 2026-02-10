"""Backward compatibility shim - module moved to core.version."""

import sys as _sys

from .core import version as _real_module  # noqa: E402
from .core.version import (  # noqa: F401
    FEATURES,
    PROFESSIONAL_FEATURES,
    SUPPORTED_ENGINES,
)

_sys.modules[__name__] = _real_module
