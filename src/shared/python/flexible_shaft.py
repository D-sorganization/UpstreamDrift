"""Backward compatibility shim - module moved to physics.flexible_shaft."""

import sys as _sys

from .physics import flexible_shaft as _real_module  # noqa: E402
from .physics.flexible_shaft import (  # noqa: F401
    GRAPHITE_DENSITY,
    GRAPHITE_E,
    SHAFT_LENGTH_DRIVER,
    SHAFT_LENGTH_IRON,
    STEEL_DENSITY,
    STEEL_E,
    BeamElement,
    FiniteElementShaftModel,
    ModalShaftModel,
    RigidShaftModel,
    ShaftFlexModel,
    ShaftMaterial,
    ShaftMode,
    ShaftModel,
    ShaftProperties,
    ShaftState,
    compute_EI_profile,
    compute_mass_profile,
    compute_section_area,
    compute_section_inertia,
    compute_static_deflection,
    create_shaft_model,
    create_standard_shaft,
    logger,
)

_sys.modules[__name__] = _real_module
