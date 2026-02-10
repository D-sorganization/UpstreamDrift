"""Backward compatibility shim - module moved to data_io.output_manager."""

import sys as _sys

from .data_io import output_manager as _real_module  # noqa: E402
from .data_io.output_manager import (  # noqa: F401
    OutputFormat,
    OutputManager,
    SimulationResultDict,
    SimulationResults,
    SimulationResultScalar,
    SimulationResultValue,
    load_results,
    logger,
    save_results,
)

_sys.modules[__name__] = _real_module
