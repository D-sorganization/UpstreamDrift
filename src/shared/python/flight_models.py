"""Backward compatibility shim - module moved to physics.flight_models."""

import sys as _sys

from .physics import flight_models as _real_module  # noqa: E402
from .physics.flight_models import (  # noqa: F401
    GOLF_BALL_MASS,
    GOLF_BALL_RADIUS,
    MIN_SPEED_THRESHOLD,
    STD_AIR_DENSITY,
    STD_GRAVITY,
    BallFlightModel,
    ConstantCoefficientModel,
    ConstantCoefficientSpec,
    FlightModelRegistry,
    FlightModelType,
    FlightResult,
    MacDonaldHanzelyModel,
    TrajectoryPoint,
    UnifiedLaunchConditions,
    WaterlooPennerModel,
    compare_models,
    logger,
)

_sys.modules[__name__] = _real_module
