"""Impact physics: data models, collision solvers, and helper functions.

.. deprecated::
    This flat module is a **thin re-export shim** over the canonical
    :mod:`src.shared.python.physics.impact_model` package (#7053). The
    implementation previously lived here as a full copy of
    ``impact_model/{models,types,utils}.py``, guarded only by the parity
    test (#7015). The duplicate bodies were deleted to satisfy DRY; every
    symbol below is the *same object* as the canonical definition, so
    ``_impact_physics.RigidBodyImpactModel is
    impact_model.models.RigidBodyImpactModel``.

    New code should import from ``impact_model`` directly.
"""

from __future__ import annotations

from .impact_model.models import (
    FiniteTimeImpactModel,
    ImpactModel,
    RigidBodyImpactModel,
    SpringDamperImpactModel,
    create_impact_model,
)
from .impact_model.types import (
    ImpactModelType,
    ImpactParameters,
    PostImpactState,
    PreImpactState,
)
from .impact_model.utils import (
    compute_gear_effect_spin,
    validate_energy_balance,
)

__all__ = [
    "FiniteTimeImpactModel",
    "ImpactModel",
    "ImpactModelType",
    "ImpactParameters",
    "PostImpactState",
    "PreImpactState",
    "RigidBodyImpactModel",
    "SpringDamperImpactModel",
    "compute_gear_effect_spin",
    "create_impact_model",
    "validate_energy_balance",
]
