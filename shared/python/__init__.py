"""Golf Modeling Suite - Shared Python Utilities.

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

from pathlib import Path  # Lightweight, always available
from typing import Any

# Suite-wide constants
SUITE_ROOT = Path(__file__).parent.parent.parent
ENGINES_ROOT = SUITE_ROOT / "engines"
SHARED_ROOT = SUITE_ROOT / "shared"
OUTPUT_ROOT = SUITE_ROOT / "output"

# Physics engine paths
MUJOCO_ROOT = ENGINES_ROOT / "physics_engines" / "mujoco"
DRAKE_ROOT = ENGINES_ROOT / "physics_engines" / "drake"
PINOCCHIO_ROOT = ENGINES_ROOT / "physics_engines" / "pinocchio"
MATLAB_2D_ROOT = ENGINES_ROOT / "Simscape_Multibody_Models" / "2D_Golf_Model"
MATLAB_3D_ROOT = ENGINES_ROOT / "Simscape_Multibody_Models" / "3D_Golf_Model"
PENDULUM_ROOT = ENGINES_ROOT / "pendulum_models"

# Import key classes for easier access
# NOTE: We now import from .core to avoid heavy dependencies in __init__
from .core import GolfModelingError, setup_logging  # noqa: E402
from .engine_manager import EngineManager, EngineStatus, EngineType  # noqa: E402

# Heavy imports are made available through lazy loading
_HEAVY_IMPORTS = {
    "ComparativeSwingAnalyzer": ("comparative_analysis", "ComparativeSwingAnalyzer"),
    "ComparativePlotter": ("comparative_plotting", "ComparativePlotter"),
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
