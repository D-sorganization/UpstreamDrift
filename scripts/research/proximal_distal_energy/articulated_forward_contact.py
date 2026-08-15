"""Public facade for bounded articulated bilateral-contact evidence.

The implementation is split into contract, single-trajectory integration, and
atlas orchestration modules so each scientific responsibility remains small and
independently testable.
"""

from scripts.research.proximal_distal_energy.articulated_forward_atlas import (
    DATA_DIR,
    run_articulated_forward_contact_atlas,
)
from scripts.research.proximal_distal_energy.articulated_forward_contract import (
    ArticulatedForwardContactConfig,
    ForwardVariant,
    mechanical_energy,
    registered_variants,
)
from scripts.research.proximal_distal_energy.articulated_forward_integration import (
    ForwardIntegrationCase,
    integrate_articulated_contact,
)

__all__ = [
    "ArticulatedForwardContactConfig",
    "DATA_DIR",
    "ForwardIntegrationCase",
    "ForwardVariant",
    "integrate_articulated_contact",
    "mechanical_energy",
    "registered_variants",
    "run_articulated_forward_contact_atlas",
]
