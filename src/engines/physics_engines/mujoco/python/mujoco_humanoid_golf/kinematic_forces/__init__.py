"""Kinematic-dependent force analysis for golf swing biomechanics.

This module computes motion-dependent forces that can be calculated from
kinematics alone, WITHOUT requiring full inverse dynamics:

- Coriolis forces
- Centrifugal forces
- Centripetal accelerations
- Velocity-dependent forces
- Gravitational forces (configuration-dependent)

These forces are critical for understanding swing dynamics and can be computed
even for parallel mechanisms where full inverse dynamics is challenging.

UNIT CONVENTIONS (Addresses Assessment B-006)
==============================================
This module uses SI units throughout unless otherwise noted:
... (refer to original docstring if needed)
"""

from ..mujoco_version import MjDataContext, _check_mujoco_version  # noqa: F401
from .analyzer import KinematicForceAnalyzer
from .export import export_kinematic_forces_to_csv
from .types import KinematicForceData

# Re-export MjDataContext so existing imports from this module continue to work
__all__ = [
    "KinematicForceData",
    "KinematicForceAnalyzer",
    "MjDataContext",
    "export_kinematic_forces_to_csv",
]
