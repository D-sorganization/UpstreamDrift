"""Backward compatibility shim - module moved to physics.impact_model."""

import sys as _sys

from .physics import impact_model as _real_module  # noqa: E402
from .physics.impact_model import (  # noqa: F401
    DEFAULT_CONTACT_DURATION,
    DEFAULT_COR,
    GOLF_BALL_MASS,
    GOLF_BALL_MOMENT_INERTIA,
    GOLF_BALL_RADIUS,
    FiniteTimeImpactModel,
    ImpactEvent,
    ImpactModel,
    ImpactModelType,
    ImpactParameters,
    ImpactRecorder,
    ImpactSolverAPI,
    PostImpactState,
    PreImpactState,
    RigidBodyImpactModel,
    SpringDamperImpactModel,
    compute_gear_effect_spin,
    create_impact_model,
    logger,
    validate_energy_balance,
)

_sys.modules[__name__] = _real_module
