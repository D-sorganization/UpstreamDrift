"""Backward-compatibility shim for the canonical pendulum physics module.

This shim re-exports the public API of the **canonical** double-pendulum
implementation so that callers that previously reached deep into the
``double_pendulum_model`` package can migrate to the canonical source
incrementally.

Canonical source (closes #3056):
    ``src/shared/python/pendulum_simulator/physics.py``

Migration guide
---------------
Old import (non-canonical)::

    from src.engines.pendulum_models.python.double_pendulum_model.physics._compat_shim import (
        PendulumParams,
        equations_of_motion,
        forward_kinematics,
        mass_matrix,
    )

New import (canonical)::

    from src.shared.python.pendulum_simulator.physics import (
        PendulumParams,
        equations_of_motion,
        forward_kinematics,
        mass_matrix,
    )
"""

from __future__ import annotations

from src.shared.python.pendulum_simulator.physics import (
    JointLimits,
    JointLimitsNDOF,
    PendulumParams,
    TorqueClamp,
    clamp_torque,
    coriolis_vector,
    equations_of_motion,
    forward_kinematics,
    gravity_vector,
    joint_limit_torque,
    joint_limit_torque_ndof,
    joint_velocities,
    mass_matrix,
    mass_matrix_components,
)

__all__ = [
    "JointLimits",
    "JointLimitsNDOF",
    "PendulumParams",
    "TorqueClamp",
    "clamp_torque",
    "coriolis_vector",
    "equations_of_motion",
    "forward_kinematics",
    "gravity_vector",
    "joint_limit_torque",
    "joint_limit_torque_ndof",
    "joint_velocities",
    "mass_matrix",
    "mass_matrix_components",
]
