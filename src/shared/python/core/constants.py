"""Project-wide constants and configuration.

Physics constants are delegated to shared.python.core.physics_constants.
Pre-computed float values are provided to avoid repeated conversions.
"""

from pathlib import Path

from . import physics_constants as _physics_constants
from .physics_constants import (
    AIR_DENSITY_SEA_LEVEL_KG_M3 as AIR_DENSITY_SEA_LEVEL_KG_M3,  # noqa: F401
)
from .physics_constants import (
    DEG_TO_RAD as DEG_TO_RAD,  # noqa: F401
)
from .physics_constants import (
    GOLF_BALL_DIAMETER_M,
    GOLF_BALL_MASS_KG,
    GOLF_BALL_MOMENT_OF_INERTIA_KG_M2,
    GOLF_BALL_RADIUS_M,
    GRAVITY_M_S2,
    PhysicalConstant,
)
from .physics_constants import (
    GOLF_BALL_DRAG_COEFFICIENT as GOLF_BALL_DRAG_COEFFICIENT,  # noqa: F401
)
from .physics_constants import (
    KG_TO_LB as KG_TO_LB,  # noqa: F401
)
from .physics_constants import (
    M_TO_FT as M_TO_FT,  # noqa: F401
)
from .physics_constants import (
    M_TO_YARD as M_TO_YARD,  # noqa: F401
)
from .physics_constants import (
    MPS_TO_KPH as MPS_TO_KPH,  # noqa: F401
)
from .physics_constants import (
    MPS_TO_MPH as MPS_TO_MPH,  # noqa: F401
)
from .physics_constants import (
    RAD_TO_DEG as RAD_TO_DEG,  # noqa: F401
)

# Backward compatibility: re-export physics constants from canonical module.
__all__ = getattr(  # noqa: PLE0605 - dynamically re-exported from physics_constants
    _physics_constants,
    "__all__",
    [name for name in dir(_physics_constants) if not name.startswith("_")],
)
globals().update({name: getattr(_physics_constants, name) for name in __all__})

# Pre-computed float values for commonly used constants
# (Avoids repeated float() conversions from PhysicalConstant in multiple modules)
GRAVITY_FLOAT: float = float(GRAVITY_M_S2)
GRAVITY: float = 9.81  # Approximate gravity for general simulation use
GOLF_BALL_MASS_FLOAT: float = float(GOLF_BALL_MASS_KG)
GOLF_BALL_RADIUS_FLOAT: float = float(GOLF_BALL_RADIUS_M)
GOLF_BALL_DIAMETER_FLOAT: float = float(GOLF_BALL_DIAMETER_M)
GOLF_BALL_MOMENT_INERTIA_FLOAT: float = float(GOLF_BALL_MOMENT_OF_INERTIA_KG_M2)

# ============================================================
# Display / Rendering Constants
# ============================================================
# Standard video resolutions used across video export, GUI rendering, and
# offscreen render contexts. Centralizing these eliminates the 21+ instances
# of bare 1920/1080/1024 literals scattered across the codebase.

HD_WIDTH: int = 1920
"""Full HD width in pixels (1080p)."""
HD_HEIGHT: int = 1080
"""Full HD height in pixels (1080p)."""
OFFSCREEN_RENDER_SIZE: int = 1024
"""Default offscreen render buffer size in pixels (square)."""
SHADOW_MAP_SIZE: int = 4096
"""Default MuJoCo shadow map resolution in pixels."""
DEFAULT_FPS: int = 60
"""Default frames per second for video export and real-time rendering."""
MAX_COLOR_CHANNEL: int = 255
"""Maximum value for an 8-bit color channel (RGB)."""
FULL_ROTATION_DEG: int = 360
"""Full rotation in degrees."""
HALF_ROTATION_DEG: int = 180
"""Half rotation in degrees (pi radians)."""
DEFAULT_MAX_DISPLAY_JOINTS: int = 6
"""Default maximum number of joints shown in a single plot (for readability)."""

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
URDF_GENERATOR_SCRIPT: Path = Path("src/tools/model_explorer/launch_model_explorer.py")
C3D_VIEWER_SCRIPT: Path = Path(
    "src/engines/Simscape_Multibody_Models/3D_Golf_Model/python/src/apps/c3d_viewer.py"
)
GUI_LAUNCHER_SCRIPT: Path = Path("src/launchers/golf_launcher.py")
# LOCAL_LAUNCHER_SCRIPT is deprecated - use GUI_LAUNCHER_SCRIPT instead
# The main launcher now supports both local and Docker modes
LOCAL_LAUNCHER_SCRIPT: Path = Path("src/launchers/golf_launcher.py")
