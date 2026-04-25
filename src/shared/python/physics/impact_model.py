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

Physics implementations are split across:
- _impact_physics.py: data models, collision solvers, helper functions
- _impact_recorder.py: ImpactEvent, ImpactRecorder, ImpactSolverAPI
"""

from __future__ import annotations

from ._impact_physics import (
    FiniteTimeImpactModel,
    ImpactModel,
    ImpactModelType,
    ImpactParameters,
    PostImpactState,
    PreImpactState,
    RigidBodyImpactModel,
    SpringDamperImpactModel,
    compute_gear_effect_spin,
    create_impact_model,
    validate_energy_balance,
)
from ._impact_recorder import ImpactEvent, ImpactRecorder, ImpactSolverAPI

__all__ = [
    "FiniteTimeImpactModel",
    "ImpactEvent",
    "ImpactModel",
    "ImpactModelType",
    "ImpactParameters",
    "ImpactRecorder",
    "ImpactSolverAPI",
    "PostImpactState",
    "PreImpactState",
    "RigidBodyImpactModel",
    "SpringDamperImpactModel",
    "compute_gear_effect_spin",
    "create_impact_model",
    "validate_energy_balance",
]
