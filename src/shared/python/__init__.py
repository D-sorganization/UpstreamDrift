"""Golf Modeling Suite - Shared Python Utilities.

Source of Truth: UpstreamDrift (this repository)
Cross-repo package: upstream-drift-shared (see pyproject.toml in this directory)
Consumers:
    - Tools repo: src/shared/theme/, src/shared/python/plot_theme/
    - Gasification_Model: vendor/ud-tools/

This module provides common functionality shared across all physics engines
and modeling approaches in the Golf Modeling Suite.
"""

__version__ = "1.0.0"
__author__ = "Golf Modeling Suite Team"

# Heavy dependencies (matplotlib, numpy, pandas) are NOT imported here
# to prevent immediate failures when they're not installed.
# Each module that needs them should import them directly.
# This allows the launcher to run and provide helpful error messages
# about missing dependencies only when specific features are used.

from typing import Any

from .core import GolfModelingError, setup_logging

# Suite-wide constants and paths are now in constants.py
from .core.constants import (
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
from .engine_core.engine_manager import EngineManager, EngineStatus, EngineType

# Heavy imports are made available through lazy loading
_HEAVY_IMPORTS = {
    "ComparativeSwingAnalyzer": (
        "validation_pkg.comparative_analysis",
        "ComparativeSwingAnalyzer",
    ),
    "ComparativePlotter": (
        "validation_pkg.comparative_plotting",
        "ComparativePlotter",
    ),
}


def __getattr__(name: str) -> Any:
    """Lazy import for heavy dependencies to avoid immediate scipy/matplotlib dependency."""
    if name in _HEAVY_IMPORTS:
        module_name, class_name = _HEAVY_IMPORTS[name]
        try:
            module = __import__(f"shared.python.{module_name}", fromlist=[class_name])
            return getattr(module, class_name)
        except ImportError as e:
            raise ImportError(
                f"Failed to import {name}. This may be due to missing dependencies "
                f"or NumPy compatibility issues. Original error: {e}"
            ) from e

    # Handle module imports
    if name == "pose_estimation":
        try:
            module = __import__("shared.python.pose_estimation", fromlist=[""])
            return module
        except ImportError as e:
            raise ImportError(
                f"Failed to import {name}. This may be due to missing dependencies "
                f"or NumPy compatibility issues. Original error: {e}"
            ) from e

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "SUITE_ROOT",
    "ENGINES_ROOT",
    "SHARED_ROOT",
    "OUTPUT_ROOT",
    "MUJOCO_ROOT",
    "DRAKE_ROOT",
    "PINOCCHIO_ROOT",
    "MATLAB_2D_ROOT",
    "MATLAB_3D_ROOT",
    "PENDULUM_ROOT",
    "EngineManager",
    "EngineType",
    "EngineStatus",
    "GolfModelingError",
    "setup_logging",
    "ComparativeSwingAnalyzer",  # Available via lazy import
    "ComparativePlotter",  # Available via lazy import
]
