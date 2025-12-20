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
