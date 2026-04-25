"""Flexible Beam Shaft Module — backward-compatible re-export.

Guideline B5 Implementation: Flexible Beam Shaft.

Provides shaft flexibility modeling options:
- Rigid shaft (baseline)
- Finite element beam model (distributed compliance)
- Modal representation (dominant bending modes)

Shaft properties:
- Stiffness distribution (EI profile along shaft)
- Mass distribution
- Damping characteristics

This module provides the mathematical framework and data structures
for shaft modeling. Physics engine integration is separate.

Planned enhancement: implement torsional dynamics (current Euler-Bernoulli beam model ignores torsional twisting).
Planned enhancement: support asymmetric cross-sections (modeling spine alignment and manufacturing tolerances).

Implementation is split across:
- shaft_params.py     — enums, dataclasses, constants
- shaft_model.py      — ShaftModel ABC, RigidShaftModel, ModalShaftModel, geometry helpers
- shaft_integrator.py — FiniteElementShaftModel, Newmark-beta integration, factory
"""

from src.shared.python.physics.shaft_integrator import (  # noqa: F401
    FiniteElementShaftModel,
    compute_static_deflection,
    create_shaft_model,
)
from src.shared.python.physics.shaft_model import (  # noqa: F401
    ModalShaftModel,
    RigidShaftModel,
    ShaftModel,
    compute_EI_profile,
    compute_mass_profile,
    compute_section_area,
    compute_section_inertia,
    create_standard_shaft,
)
from src.shared.python.physics.shaft_params import (  # noqa: F401
    GRAPHITE_DENSITY,
    GRAPHITE_E,
    SHAFT_LENGTH_DRIVER,
    SHAFT_LENGTH_IRON,
    STEEL_DENSITY,
    STEEL_E,
    BeamElement,
    ShaftFlexModel,
    ShaftMaterial,
    ShaftMode,
    ShaftProperties,
    ShaftState,
)
