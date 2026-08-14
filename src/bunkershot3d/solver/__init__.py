"""F0 tier: Dynamic Resistive Force Theory solver (issue #8611, ADR-0032).

The default solver for design iteration. RFT is the only per-geometry method
cheap enough for a design loop (~ms/shot vs 30-90 min for MPM).

Modules:
    coefficients: 20-term polynomial table from Agarwal, Goldman and Kamrin
    material: Material scaling from bulk density and friction angle
    envelope: Validity envelope enforcement
    drft: The complete DRFT solver
"""

from .coefficients import (
    DRFT_COEFFICIENTS,
    compute_alpha_components,
    compute_f_values,
    compute_term_basis,
)
from .drft import DRFTResult, DRFTSolver, FidelityTier
from .envelope import (
    EnvelopeStatus,
    ValidityEnvelope,
    ValidityVerdict,
    compute_froude_number,
    compute_micro_inertial,
)
from .material import (
    GRAVITY_M_S2,
    compute_f_hat,
    compute_xi_n,
    mu_from_friction_angle_deg,
)

__all__ = [
    "DRFT_COEFFICIENTS",
    "DRFTResult",
    "DRFTSolver",
    "EnvelopeStatus",
    "FidelityTier",
    "GRAVITY_M_S2",
    "ValidityEnvelope",
    "ValidityVerdict",
    "compute_alpha_components",
    "compute_f_hat",
    "compute_f_values",
    "compute_froude_number",
    "compute_micro_inertial",
    "compute_term_basis",
    "compute_xi_n",
    "mu_from_friction_angle_deg",
]
