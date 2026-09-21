"""Pendulum motion matching package."""

from __future__ import annotations

from .adapters import (
    MODEL_ID_ANALYTICAL,
    MODEL_ID_TOOLS,
    DoublePendulumAdapter,
    check_dynamics_parity,
    forward_kinematics_2d,
    params_analytical_to_tools,
    params_tools_to_analytical,
)
from .provider import PendulumFitSwingProvider
from .torque_optimization import (
    COEFFS_PER_JOINT,
    BernsteinTorqueProfile,
    fit_bounded_double_pendulum,
    integrate_double_pendulum_rollout,
)

__all__ = [
    "COEFFS_PER_JOINT",
    "MODEL_ID_ANALYTICAL",
    "MODEL_ID_TOOLS",
    "BernsteinTorqueProfile",
    "DoublePendulumAdapter",
    "PendulumFitSwingProvider",
    "check_dynamics_parity",
    "fit_bounded_double_pendulum",
    "forward_kinematics_2d",
    "integrate_double_pendulum_rollout",
    "params_analytical_to_tools",
    "params_tools_to_analytical",
]
