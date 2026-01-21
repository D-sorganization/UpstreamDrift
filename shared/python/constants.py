"""Project-wide constants and configuration.

Physics constants are delegated to shared.python.physics_constants.
"""

from pathlib import Path

from shared.python.physics_constants import *  # noqa: F403

# Project Root Paths
SUITE_ROOT = Path(__file__).resolve().parent.parent.parent
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

# Numerical constants (Solver specific)
EPSILON: float = 1e-15
MAX_ITERATIONS: int = 10000
CONVERGENCE_TOLERANCE: float = 1e-6
DEFAULT_TIME_STEP = PhysicalConstant(  # noqa: F405
    0.001, "s", "Standard", "Default simulation time step"
)

# Reproducibility
DEFAULT_RANDOM_SEED: int = 42

# Project Config
MUJOCO_LAUNCHER_SCRIPT: Path = Path(
    "engines/physics_engines/mujoco/python/mujoco_humanoid_golf/advanced_gui.py"
)
DRAKE_LAUNCHER_SCRIPT: Path = Path(
    "engines/physics_engines/drake/python/src/golf_gui.py"
)
PINOCCHIO_LAUNCHER_SCRIPT: Path = Path(
    "engines/physics_engines/pinocchio/python/pinocchio_golf/gui.py"
)
URDF_GENERATOR_SCRIPT: Path = Path("tools/urdf_generator/main.py")
C3D_VIEWER_SCRIPT: Path = Path(
    "engines/Simscape_Multibody_Models/3D_Golf_Model/python/src/apps/c3d_viewer.py"
)
GUI_LAUNCHER_SCRIPT: Path = Path("launchers/golf_launcher.py")
LOCAL_LAUNCHER_SCRIPT: Path = Path("launchers/golf_suite_launcher.py")
