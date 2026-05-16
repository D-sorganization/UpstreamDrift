"""Muscle models, kinematics, and swing analysis."""

from __future__ import annotations

from .activation_dynamics import ActivationDynamics
from .biomechanics_data import BiomechanicalData
from .fsp_integration import (
    FspResult,
    compute_swing_fsp,
    detect_md_mf_window,
    extract_clubhead_trajectory,
)
from .hill_muscle import HillMuscleModel, MuscleParameters, MuscleState
from .kinematic_sequence import SegmentPeak, SegmentTimingResult
from .multi_muscle import MuscleAttachment, MuscleGroup
from .muscle_equilibrium import EquilibriumSolver
from .swing_comparison import ComparisonMetric, DTWResult, SwingComparator
from .swing_plane_analysis import SwingPlaneAnalyzer, SwingPlaneMetrics, fit_plane
from .ztcf import ZTCFResult

# Optional modules with external dependencies (dynamic_com depends on
# humanoid_character_builder which may not be available in all environments)
try:
    from .dynamic_com import BiomechanicalModel, SegmentDefinition
except ImportError:
    BiomechanicalModel = None  # type: ignore[assignment,misc]
    SegmentDefinition = None  # type: ignore[assignment,misc]

# Optional modules with external dependencies (continued)
try:
    from .grf_visualization import plot_grf_and_com_3d
except ImportError:
    plot_grf_and_com_3d = None  # type: ignore[assignment]

try:
    from .humanoid_urdf_contracts import ContractViolation, ValidationReport, describe
except ImportError:
    ContractViolation = None  # type: ignore[assignment,misc]
    ValidationReport = None  # type: ignore[assignment,misc]
    describe = None  # type: ignore[assignment]

try:
    from .muscle_analysis import MuscleSynergyAnalyzer, SynergyResult
except ImportError:
    MuscleSynergyAnalyzer = None  # type: ignore[assignment,misc]
    SynergyResult = None  # type: ignore[assignment,misc]

try:
    from .myoconverter_integration import MyoConverter
except ImportError:
    MyoConverter = None  # type: ignore[assignment,misc]

try:
    from .myosuite_adapter import MuscleDrivenEnv
except ImportError:
    MuscleDrivenEnv = None  # type: ignore[assignment,misc]

try:
    from .rust_muscle import f_l, is_rust_available
except ImportError:
    f_l = None  # type: ignore[assignment]
    is_rust_available = None  # type: ignore[assignment]

try:
    from .swing_plane_visualization import SwingPlaneVisualization
except ImportError:
    SwingPlaneVisualization = None  # type: ignore[assignment,misc]

__all__: list[str] = [
    # fsp_integration
    "FspResult",
    "compute_swing_fsp",
    "detect_md_mf_window",
    "extract_clubhead_trajectory",
    # activation_dynamics
    "ActivationDynamics",
    # biomechanics_data
    "BiomechanicalData",
    # hill_muscle
    "HillMuscleModel",
    "MuscleParameters",
    "MuscleState",
    # kinematic_sequence
    "SegmentPeak",
    "SegmentTimingResult",
    # multi_muscle
    "MuscleAttachment",
    "MuscleGroup",
    # muscle_equilibrium
    "EquilibriumSolver",
    # swing_comparison
    "ComparisonMetric",
    "DTWResult",
    "SwingComparator",
    # swing_plane_analysis
    "SwingPlaneAnalyzer",
    "SwingPlaneMetrics",
    "fit_plane",
    # ztcf
    "ZTCFResult",
]

# Append optional symbols when available
if BiomechanicalModel is not None:
    __all__.extend(["BiomechanicalModel", "SegmentDefinition"])
if ContractViolation is not None:
    __all__.extend(["ContractViolation", "ValidationReport", "describe"])
if f_l is not None:
    __all__.extend(["f_l", "is_rust_available"])
if MuscleSynergyAnalyzer is not None:
    __all__.extend(["MuscleSynergyAnalyzer", "SynergyResult"])
if MyoConverter is not None:
    __all__.append("MyoConverter")
if MuscleDrivenEnv is not None:
    __all__.append("MuscleDrivenEnv")
if SwingPlaneVisualization is not None:
    __all__.append("SwingPlaneVisualization")
if plot_grf_and_com_3d is not None:
    __all__.append("plot_grf_and_com_3d")
