"""MuJoCo golf swing biomechanical analysis package.

This package provides professional-grade robotics analysis tools for golf swing
simulation and optimization, including:

- Biomechanical analysis with force/torque extraction
- Advanced kinematics (Jacobians, manipulability, IK)
- Multiple control schemes (impedance, admittance, hybrid)
- Trajectory optimization
- Parallel mechanism analysis (constraint Jacobians)
- Motion primitive libraries
- Motion capture integration and retargeting
- Kinematic-dependent force analysis (Coriolis, centrifugal)
- Inverse dynamics solvers
- Telemetry capture and reporting
"""

# Control system
# Motion capture and force analysis
# Advanced robotics modules
# Core modules
# Core modules
try:
    from . import (
        advanced_control,
        advanced_kinematics,
        biomechanics,
        control_system,
        inverse_dynamics,
        kinematic_forces,
        models,
        motion_capture,
        motion_optimization,
        plotting,
        urdf_io,
    )
    from .control_system import ActuatorControl, ControlSystem, ControlType
except (ImportError, OSError):
    # If core modules fail to load (e.g. strict headless environments with no GL),
    # we allow partial loading.
    advanced_control = None  # type: ignore
    advanced_kinematics = None  # type: ignore
    biomechanics = None  # type: ignore
    control_system = None  # type: ignore
    inverse_dynamics = None  # type: ignore
    kinematic_forces = None  # type: ignore
    models = None  # type: ignore
    motion_capture = None  # type: ignore
    motion_optimization = None  # type: ignore
    plotting = None  # type: ignore
    urdf_io = None  # type: ignore
    ActuatorControl = None  # type: ignore
    ControlSystem = None  # type: ignore
    ControlType = None  # type: ignore

# Telemetry
from .telemetry import TelemetryRecorder, TelemetryReport

# GUI modules - optional, may fail in headless environments
_has_gui = False
try:
    from . import advanced_gui, sim_widget

    _has_gui = True
except ImportError:
    # GUI not available (e.g., in headless CI environments)
    # Don't expose these in __all__ to avoid AttributeErrors
    advanced_gui = None  # type: ignore[assignment]
    sim_widget = None  # type: ignore[assignment]

__version__ = "2.1.0"

# Build __all__ conditionally based on available modules
__all__ = [
    "ActuatorControl",
    "ControlSystem",
    "ControlType",
    "TelemetryRecorder",
    "TelemetryReport",
    "advanced_control",
    "advanced_kinematics",
    "biomechanics",
    "control_system",
    "inverse_dynamics",
    "kinematic_forces",
    "models",
    "motion_capture",
    "motion_optimization",
    "plotting",
    "urdf_io",
]

# Only include GUI modules in __all__ if they're available
if _has_gui:
    __all__ += ["advanced_gui", "sim_widget"]
