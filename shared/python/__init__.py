"""Golf Modeling Suite - Shared Python Utilities.

This module provides common functionality shared across all physics engines
and modeling approaches in the Golf Modeling Suite.
"""

__version__ = "1.0.0"
__author__ = "Golf Modeling Suite Team"

# Common imports for all engines
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
from .engine_manager import EngineManager, EngineType, EngineStatus
from .common_utils import GolfModelingError, setup_logging

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
]
