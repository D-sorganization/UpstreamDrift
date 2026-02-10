"""Backward compatibility shim - module moved to physics.ground_reaction_forces."""

import sys as _sys

from .physics import ground_reaction_forces as _real_module  # noqa: E402
from .physics.ground_reaction_forces import (  # noqa: F401
    ANGULAR_IMPULSE_TOLERANCE,
    COP_POSITION_TOLERANCE_MM,
    GRAVITY_MAGNITUDE,
    GRF_MAGNITUDE_TOLERANCE,
    FootSide,
    GRFAnalyzer,
    GRFSummary,
    GRFTimeSeries,
    GroundReactionForce,
    ImpulseMetrics,
    compute_angular_impulse,
    compute_cop_from_grf,
    compute_cop_trajectory_length,
    compute_linear_impulse,
    extract_grf_from_contacts,
    logger,
    validate_grf_cross_engine,
)

_sys.modules[__name__] = _real_module
