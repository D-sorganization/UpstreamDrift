"""Backward compatibility shim - module moved to physics.ball_flight_physics."""

import sys as _sys

from .physics import ball_flight_physics as _real_module  # noqa: E402
from .physics.ball_flight_physics import (  # noqa: F401
    MAX_LIFT_COEFFICIENT,
    MIN_SPEED_THRESHOLD,
    NUMERICAL_EPSILON,
    BallFlightSimulator,
    BallProperties,
    EnhancedBallFlightSimulator,
    EnvironmentalConditions,
    LaunchConditions,
    TrajectoryPoint,
    logger,
)

_sys.modules[__name__] = _real_module
