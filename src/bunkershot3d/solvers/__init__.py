"""Granular solvers for BunkerShot3D (issue #8611, epic #8607, ADR-0032).

The **F0 tier -- dynamic 3D Resistive Force Theory -- is the default
solver of the whole tool.** Every other tier exists to calibrate or
cross-check it, because RFT is the only method that is genuinely
per-geometry: the force on an intruder is an integral of a local stress
response over the swept surface, so changing the sole shape changes the
answer without re-running a granular solve. That is exactly the design
question, at ~ms per shot instead of 30-90 minutes.

Layout
------

* :mod:`.protocol` -- the one ``GranularSolver`` interface every tier
  implements, plus ``Wrench``, ``IntrusionState`` and ``SolverResult``.
  A result cannot be constructed without a validity verdict.
* :mod:`.envelope` -- the dimensionless groups, the refusal rules and
  the standing caveats. The most important module here: at 25 m/s a
  bunker shot is ~60x outside 3D-RFT's stated Froude limit and ~20x
  beyond any published validation.
* :mod:`.coefficients` -- the 20-term 3D-RFT polynomial, the material
  scaling cubic, and the provenance of every fitted constant. All
  borrowed; none measured on golf bunker sand.
* :mod:`.structural` -- the DRFT structural correction ``delta_h``,
  which is *not* optional and whose wedge form is unknown.
* :mod:`.elements` -- the structure-of-arrays surface discretisation.
* :mod:`.drft` -- the solver.
* :mod:`.shot` -- time-marching one shot, under the 50 ms budget.

Honesty summary
---------------

Nothing in this package is calibrated against golf bunker sand. The
polynomial is fitted to a generic frictional-plastic medium, ``lambda``
comes from plate-drag and wheel experiments, the material-scaling cubic
comes from the same source, and ``delta_h`` has no wedge-specific form at
all. Every result says so, in the verdict it carries.
"""

from __future__ import annotations

from .coefficients import (
    LAMBDA_BY_MOTION,
    PLATE_DRAG_LAMBDA,
    RFT_COEFFICIENT_PROVENANCE,
    RFT_POLYNOMIAL_COEFFICIENTS,
    VERTICAL_PLATE_ALPHA_Z,
    MaterialResponse,
    generic_alpha,
    internal_friction_mu,
    material_scaling_pa_per_m,
    polynomial_terms,
    scaling_shape_function,
)
from .drft import DEFAULT_FEATURE_SCALES_M, DRFTSolver, ElementResponse
from .elements import SurfaceElements
from .envelope import (
    GRAVITY_M_S2,
    MAX_VALIDATED_SPEED_M_S,
    MIN_CONTINUUM_SIZE_RATIO,
    RFT_FROUDE_LIMIT,
    RFT_INERTIAL_NUMBER_LIMIT,
    RFT_QUASI_STATIC_FROUDE_CEILING,
    STANDING_CAVEATS,
    Caveat,
    DimensionlessGroups,
    EnvelopeContext,
    EnvelopeStatus,
    FeatureScale,
    RefusalPolicy,
    ValidityVerdict,
    dimensionless_groups,
    evaluate_envelope,
    worst_of,
)
from .exceptions import (
    CalibrationError,
    OutOfEnvelopeError,
    ShotTruncatedError,
    SolverError,
    SolverInputError,
)
from .protocol import (
    FidelityTier,
    GranularSolver,
    IntrusionState,
    SolverResult,
    Wrench,
)
from .shot import HeadKinematics, ShotResult, ShotSettings, simulate_shot
from .structural import (
    CrossoverSaturatingDepression,
    DepressionInputs,
    StructuralCorrection,
    WheelAnalogueDepression,
    ZeroDepression,
    default_structural_correction,
)

__all__ = [
    "DEFAULT_FEATURE_SCALES_M",
    "GRAVITY_M_S2",
    "LAMBDA_BY_MOTION",
    "MAX_VALIDATED_SPEED_M_S",
    "MIN_CONTINUUM_SIZE_RATIO",
    "PLATE_DRAG_LAMBDA",
    "RFT_COEFFICIENT_PROVENANCE",
    "RFT_FROUDE_LIMIT",
    "RFT_INERTIAL_NUMBER_LIMIT",
    "RFT_POLYNOMIAL_COEFFICIENTS",
    "RFT_QUASI_STATIC_FROUDE_CEILING",
    "STANDING_CAVEATS",
    "VERTICAL_PLATE_ALPHA_Z",
    "CalibrationError",
    "Caveat",
    "CrossoverSaturatingDepression",
    "DRFTSolver",
    "DepressionInputs",
    "DimensionlessGroups",
    "ElementResponse",
    "EnvelopeContext",
    "EnvelopeStatus",
    "FeatureScale",
    "FidelityTier",
    "GranularSolver",
    "HeadKinematics",
    "IntrusionState",
    "MaterialResponse",
    "OutOfEnvelopeError",
    "RefusalPolicy",
    "ShotResult",
    "ShotSettings",
    "ShotTruncatedError",
    "SolverError",
    "SolverInputError",
    "SolverResult",
    "StructuralCorrection",
    "SurfaceElements",
    "ValidityVerdict",
    "WheelAnalogueDepression",
    "Wrench",
    "ZeroDepression",
    "default_structural_correction",
    "dimensionless_groups",
    "evaluate_envelope",
    "generic_alpha",
    "internal_friction_mu",
    "material_scaling_pa_per_m",
    "polynomial_terms",
    "scaling_shape_function",
    "simulate_shot",
    "worst_of",
]
