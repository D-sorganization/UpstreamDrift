"""Core utilities: errors, constants, logging, and version.

This package groups fundamental modules:
- _core: logging setup, base error re-exports
- constants: suite-wide path constants
- exceptions: deprecated exception aliases (use error_utils)
- error_utils: consolidated exception hierarchy
- numerical_constants: numerical tolerances and thresholds
- physics_constants: physical constants for golf simulation
- contracts: design-by-contract framework
- type_utils: type helper utilities
- version: suite version info
- error_decorators: error-handling decorators
- datetime_utils: date/time utilities
"""

from typing import Any

from ._core import (
    DataFormatError,
    EngineNotFoundError,
    GolfModelingError,
    get_logger,
    setup_logging,
    setup_structured_logging,
)
from .constants import (
    DEFAULT_TIME_STEP,
    DRAKE_ROOT,
    ENGINES_ROOT,
    MATLAB_2D_ROOT,
    MATLAB_3D_ROOT,
    MUJOCO_ROOT,
    OUTPUT_ROOT,
    PENDULUM_ROOT,
    PINOCCHIO_ROOT,
    SHARED_ROOT,
    SUITE_ROOT,
)
from .version import __version__

__all__ = [
    "DEFAULT_TIME_STEP",
    "DRAKE_ROOT",
    "DataFormatError",
    "ENGINES_ROOT",
    "EngineNotFoundError",
    "GolfModelingError",
    "MATLAB_2D_ROOT",
    "MATLAB_3D_ROOT",
    "MUJOCO_ROOT",
    "OUTPUT_ROOT",
    "PENDULUM_ROOT",
    "PINOCCHIO_ROOT",
    "SHARED_ROOT",
    "SUITE_ROOT",
    "__version__",
    "get_logger",
    "setup_logging",
    "setup_structured_logging",
]


def __getattr__(name: str) -> Any:
    """Delegate attribute lookup to _core for backward compatibility.

    The original core.py exported names from exceptions.py via __getattr__
    (deprecated aliases). This ensures those lookups still work through
    the core/ package.
    """
    from . import _core

    try:
        return getattr(_core, name)
    except AttributeError:
        pass

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
