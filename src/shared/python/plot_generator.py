"""Backward compatibility shim - module moved to gui_pkg.plot_generator."""

import sys as _sys

from .gui_pkg import plot_generator as _real_module  # noqa: E402
from .gui_pkg.plot_generator import (  # noqa: F401
    ALL_PLOT_TYPES,
    PlotConfig,
    PlotGenerator,
    PlotType,
    SimulationData,
    logger,
)

_sys.modules[__name__] = _real_module
