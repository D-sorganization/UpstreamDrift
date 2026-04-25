# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.  # noqa: E501
# It requires domain-aware structural extraction to isolate its internal classes appropriately.  # noqa: E501

"""Advanced control schemes for robotics applications.

This module implements state-of-the-art control strategies including:
- Impedance control (position-based)
- Admittance control (force-based)
- Hybrid force-position control
- Computed torque control (inverse dynamics)
- Task-space control with nullspace projection
- Operational space control
"""

from ._advanced_controller import AdvancedController
from ._control_types import ControlMode, HybridControlMask, ImpedanceParameters
from ._trajectory_generator import TrajectoryGenerator

__all__ = [
    "AdvancedController",
    "ControlMode",
    "HybridControlMask",
    "ImpedanceParameters",
    "TrajectoryGenerator",
]
