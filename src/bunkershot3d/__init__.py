"""BunkerShot3D: A 3-D simulation of a golf bunker shot.

Re-exports the public API from all subpackages so consumers can import
directly from \`bunkershot3d\` instead of reaching into submodules.
"""

__version__ = "0.1.0"

# Backend drivers
from .backends import ChronoDriver, LiggghtsDriver, MPMDriver

# Calibration
from .calibration import AngleOfReposeExperiment, CalibrationOptimizer, DrainedShearCellExperiment

# Geometry
from .geometry import ClubheadGenerator

# I/O
from .io import BunkerShotResultReader, BunkerShotResultWriter

# Kinematics
from .kinematics import CoSimulator, MockDoublePendulum, SwingTrajectory, generate_reference_trajectory

# Post-processing
from .postproc import WrenchTrace

__all__: list[str] = [
    "AngleOfReposeExperiment",
    "BunkerShotResultReader",
    "BunkerShotResultWriter",
    "CalibrationOptimizer",
    "ChronoDriver",
    "ClubheadGenerator",
    "CoSimulator",
    "DrainedShearCellExperiment",
    "LiggghtsDriver",
    "MPMDriver",
    "MockDoublePendulum",
    "SwingTrajectory",
    "WrenchTrace",
    "__version__",
    "generate_reference_trajectory",
]
