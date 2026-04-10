"""Flexible Beam Shaft Module.

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
"""

from __future__ import annotations

from src.shared.python.logging_pkg.logging_config import get_logger

from ._shaft_data import (
    BeamElement,
    ShaftFlexModel,
    ShaftMaterial,
    ShaftMode,
    ShaftProperties,
    ShaftState,
)
from ._shaft_fem import FiniteElementShaftModel
from ._shaft_model import ModalShaftModel, RigidShaftModel, ShaftModel
from ._shaft_properties import (
    GRAPHITE_DENSITY,
    GRAPHITE_E,
    SHAFT_LENGTH_DRIVER,
    SHAFT_LENGTH_IRON,
    STEEL_DENSITY,
    STEEL_E,
    compute_EI_profile,
    compute_mass_profile,
    compute_section_area,
    compute_section_inertia,
    create_standard_shaft,
)
from ._shaft_utils import compute_static_deflection, create_shaft_model

logger = get_logger(__name__)

__all__ = [
    "BeamElement",
    "FiniteElementShaftModel",
    "GRAPHITE_DENSITY",
    "GRAPHITE_E",
    "ModalShaftModel",
    "RigidShaftModel",
    "SHAFT_LENGTH_DRIVER",
    "SHAFT_LENGTH_IRON",
    "STEEL_DENSITY",
    "STEEL_E",
    "ShaftFlexModel",
    "ShaftMaterial",
    "ShaftMode",
    "ShaftModel",
    "ShaftProperties",
    "ShaftState",
    "compute_EI_profile",
    "compute_mass_profile",
    "compute_section_area",
    "compute_section_inertia",
    "compute_static_deflection",
    "create_shaft_model",
    "create_standard_shaft",
]
