"""Standalone driven double pendulum educational toolkit."""

from .physics.double_pendulum import (
    DEFAULT_ARM_CENTER_OF_MASS_RATIO,
    DEFAULT_ARM_INERTIA_SCALING,
    DEFAULT_ARM_LENGTH_M,
    DEFAULT_ARM_MASS_KG,
    DEFAULT_CLUBHEAD_MASS_KG,
    DEFAULT_DAMPING_SHOULDER,
    DEFAULT_DAMPING_WRIST,
    DEFAULT_PLANE_INCLINATION_DEG,
    DEFAULT_SHAFT_COM_RATIO,
    DEFAULT_SHAFT_LENGTH_M,
    DEFAULT_SHAFT_MASS_KG,
    DoublePendulumDynamics,
    DoublePendulumParameters,
    DoublePendulumState,
    ExpressionFunction,
    compile_forcing_functions,
)

__all__ = [
    "DEFAULT_ARM_CENTER_OF_MASS_RATIO",
    "DEFAULT_ARM_INERTIA_SCALING",
    "DEFAULT_ARM_LENGTH_M",
    "DEFAULT_ARM_MASS_KG",
    "DEFAULT_CLUBHEAD_MASS_KG",
    "DEFAULT_DAMPING_SHOULDER",
    "DEFAULT_DAMPING_WRIST",
    "DEFAULT_PLANE_INCLINATION_DEG",
    "DEFAULT_SHAFT_COM_RATIO",
    "DEFAULT_SHAFT_LENGTH_M",
    "DEFAULT_SHAFT_MASS_KG",
    "DoublePendulumDynamics",
    "DoublePendulumParameters",
    "DoublePendulumState",
    "ExpressionFunction",
    "compile_forcing_functions",
]
