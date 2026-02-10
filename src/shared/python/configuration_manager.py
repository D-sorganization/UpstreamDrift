"""Backward compatibility shim - module moved to config.configuration_manager."""

import sys as _sys

from .config import configuration_manager as _real_module  # noqa: E402
from .config.configuration_manager import (  # noqa: F401
    ConfigurationManager,
    SimulationConfig,
)

_sys.modules[__name__] = _real_module
