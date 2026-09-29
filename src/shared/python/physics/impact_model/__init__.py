"""Modular Impact Model Module.

Guideline K3 Implementation: Modular Impact Model (MuJoCo).

Provides standalone impact solver for ball-clubface collision including:
- Rigid body collision (coefficient of restitution)
- Compliant contact (spring-damper, Kelvin-Voigt)
- Finite-time contact (impulse-momentum with contact duration)
- Spin generation models (gear effect, offset impact)

The module is engine-agnostic with Python API for external solvers.

.. note::
    Contact parameters (COR, friction) have Rust kernel equivalents in
    ``src.shared.python.physics.rust_kernel``. New code should use
    ``create_contact_parameters()`` from the adapter instead of hardcoded values.
"""

from .models import (
    FiniteTimeImpactModel,
    ImpactModel,
    RigidBodyImpactModel,
    SpringDamperImpactModel,
    create_impact_model,
)
from .solver import (
    ImpactRecorder,
    ImpactSolverAPI,
)
from .types import (
    ImpactEvent,
    ImpactModelType,
    ImpactParameters,
    PostImpactState,
    PreImpactState,
)
from .utils import (
    compute_gear_effect_spin,
    validate_energy_balance,
)

__all__ = [
    "ImpactModelType",
    "PreImpactState",
    "PostImpactState",
    "ImpactParameters",
    "ImpactEvent",
    "ImpactModel",
    "RigidBodyImpactModel",
    "SpringDamperImpactModel",
    "FiniteTimeImpactModel",
    "create_impact_model",
    "compute_gear_effect_spin",
    "validate_energy_balance",
    "ImpactRecorder",
    "ImpactSolverAPI",
]
