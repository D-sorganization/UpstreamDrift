"""BunkerShot3D: A 3-D simulation of a golf bunker shot.

Re-exports the public API from all subpackages so consumers can import
directly from ``bunkershot3d`` instead of reaching into submodules.
"""

__version__ = "0.1.0"

# Ball model
from .ball import (
    BallLie,
    BallLieType,
    BallProperties,
    BunkerShotState,
    compute_bunker_launch,
    to_post_impact_state,
)

# Backend drivers
from .backends import ChronoDriver, LiggghtsDriver, MPMDriver

# Calibration
from .calibration import (
    AngleOfReposeExperiment,
    CalibrationOptimizer,
    DrainedShearCellExperiment,
)

# Exceptions
from .exceptions import BackendNotImplementedError

# Geometry
from .geometry import ClubheadGenerator

# I/O
from .io import BunkerShotResultReader, BunkerShotResultWriter

# Kinematics
from .kinematics import (
    CoSimulator,
    CoupledDoublePendulum,
    SwingTrajectory,
    generate_reference_trajectory,
)

# Post-processing
from .postproc import WrenchTrace

__all__: list[str] = [
    "AngleOfReposeExperiment",
    "BackendNotImplementedError",
    "BallLie",
    "BallLieType",
    "BallProperties",
    "BunkerShotResultReader",
    "BunkerShotResultWriter",
    "BunkerShotState",
    "CalibrationOptimizer",
    "ChronoDriver",
    "ClubheadGenerator",
    "CoSimulator",
    "DrainedShearCellExperiment",
    "LiggghtsDriver",
    "MPMDriver",
    "CoupledDoublePendulum",
    "SwingTrajectory",
    "WrenchTrace",
    "__version__",
    "compute_bunker_launch",
    "generate_reference_trajectory",
    "to_post_impact_state",
]
