"""Verification, Validation and the credibility statement (issue #8616, W9).

Three separate things, kept separate, in this order
---------------------------------------------------

1. **Code verification** -- is the maths right?  :mod:`.conservation`,
   :mod:`.convergence` and :mod:`.cases`.  **Never uses experimental
   data.**  Conservation residuals split by class, order of accuracy
   under refinement of the DRFT surface discretisation, and analytic
   limit cases with closed-form answers.
2. **Solution verification** -- how big is the numerical error?
   :mod:`.gci`.  Grid Convergence Index with Richardson extrapolation per
   Celik et al. (2008), producing ``u_num``.
3. **Validation** -- is it the right maths?  :mod:`.validation` and
   :mod:`.reference_data`.  The ASME V&V 20 metric, and a register of the
   quantities for which **no published measurement exists** so that the
   suite refuses rather than inventing one.

:mod:`.credibility` then states, in NASA-STD-7009B's framing, what is
verified, what is validated, against what, over what domain, with what
uncertainty -- reporting the achieved level *and* the gap to the
threshold for every factor.

What this suite refuses to claim
--------------------------------

The literature search behind this issue found no published value anywhere
for ball launch angle, speed or spin from a splash shot, for clubhead
deceleration in sand, for the energy split, or for ejecta mass.  The
package therefore validates none of them, and
:func:`~bunkershot3d.vandv.reference_data.require_measurable` raises if
anybody tries.
"""

from __future__ import annotations

from .cases import (
    CylinderVerificationCase,
    FlatPlateVerificationCase,
    asymmetric_body_elements,
    cylinder_case,
    cylinder_inertial_force_n,
    cylinder_side_elements,
    flat_plate_depth_force_n,
    flat_plate_elements,
    flat_plate_inertial_force_n,
    leading_edge_fraction,
    quasi_static_plate_case,
    quasi_static_solver,
    spiral_body_elements,
)
from .conservation import (
    ROUND_OFF_TOLERANCE,
    ConservationClass,
    ConservationResidual,
    element_moment_residual,
    energy_work_residual,
    inertial_power_is_dissipative,
    linear_impulse_residual,
    moment_transfer_residual,
    residual_series,
)
from .convergence import (
    ObservedOrder,
    RefinementLevel,
    observed_order_from_errors,
    observed_order_from_residuals,
    refinement_errors,
)
from .credibility import (
    CREDIBILITY_ASSESSMENT,
    DESIGN_FEATURE_LENGTH_M,
    DESIGN_SPEED_M_S,
    MAX_CREDIBILITY_LEVEL,
    CredibilityFactor,
    EnvelopeExceedance,
    FactorAssessment,
    credibility_table_markdown,
    domain_of_applicability,
    domain_table_markdown,
    envelope_exceedance,
)
from .exceptions import (
    ConservationClassError,
    NoReferenceDataError,
    SolutionVerificationError,
    VandVError,
    VerificationError,
)
from .gci import (
    COMFORTABLE_REFINEMENT_RATIO,
    FACTOR_OF_SAFETY_THREE_GRID,
    FACTOR_OF_SAFETY_TWO_GRID,
    GCI_COVERAGE_FACTOR,
    ApparentOrder,
    ConvergenceType,
    GCIResult,
    GCIStudy,
    GridSolution,
    apparent_order,
    error_amplification,
    grid_convergence_index,
    richardson_extrapolate,
    two_grid_gci,
)
from .reference_data import (
    GRANULAR_INTRUSION_BENCHMARK,
    REFERENCE_DATASETS,
    UNMEASURED_QUANTITIES,
    WIVOU_2016,
    DomainOverlap,
    IntrusionBenchmark,
    ReferenceDataset,
    ReferenceRange,
    domain_overlap,
    reference_dataset,
    require_measurable,
)
from .studies import (
    QUIKRETE_ANALOGUE_SOURCE,
    QUIKRETE_BULK_DENSITY_KG_M3,
    QUIKRETE_FRICTION_ANGLE_DEG,
    QUIKRETE_MEASURED_ALPHA_Z_N_PER_CM3,
    WIVOU_CONTROLLED_VARIABLES,
    carry_correlation_comparison,
    correlation_standard_uncertainty,
    friction_angle_leverage_per_degree,
    plate_response_comparisons,
    predicted_alpha_z_n_per_cm3,
    surface_refinement_study,
)
from .validation import (
    COVERAGE_FACTOR,
    NumericalUncertainty,
    ValidationComparison,
    ValidationReport,
    ValidationResult,
    validate,
    validation_report,
)

__all__ = [
    "COMFORTABLE_REFINEMENT_RATIO",
    "COVERAGE_FACTOR",
    "CREDIBILITY_ASSESSMENT",
    "DESIGN_FEATURE_LENGTH_M",
    "DESIGN_SPEED_M_S",
    "FACTOR_OF_SAFETY_THREE_GRID",
    "FACTOR_OF_SAFETY_TWO_GRID",
    "GCI_COVERAGE_FACTOR",
    "GRANULAR_INTRUSION_BENCHMARK",
    "MAX_CREDIBILITY_LEVEL",
    "QUIKRETE_ANALOGUE_SOURCE",
    "QUIKRETE_BULK_DENSITY_KG_M3",
    "QUIKRETE_FRICTION_ANGLE_DEG",
    "QUIKRETE_MEASURED_ALPHA_Z_N_PER_CM3",
    "REFERENCE_DATASETS",
    "ROUND_OFF_TOLERANCE",
    "UNMEASURED_QUANTITIES",
    "WIVOU_2016",
    "WIVOU_CONTROLLED_VARIABLES",
    "ApparentOrder",
    "ConservationClass",
    "ConservationClassError",
    "ConservationResidual",
    "ConvergenceType",
    "CredibilityFactor",
    "CylinderVerificationCase",
    "DomainOverlap",
    "EnvelopeExceedance",
    "FactorAssessment",
    "FlatPlateVerificationCase",
    "GCIResult",
    "GCIStudy",
    "GridSolution",
    "IntrusionBenchmark",
    "NoReferenceDataError",
    "NumericalUncertainty",
    "ObservedOrder",
    "ReferenceDataset",
    "ReferenceRange",
    "RefinementLevel",
    "SolutionVerificationError",
    "VandVError",
    "ValidationComparison",
    "ValidationReport",
    "ValidationResult",
    "VerificationError",
    "apparent_order",
    "asymmetric_body_elements",
    "carry_correlation_comparison",
    "correlation_standard_uncertainty",
    "credibility_table_markdown",
    "cylinder_case",
    "cylinder_inertial_force_n",
    "cylinder_side_elements",
    "domain_of_applicability",
    "domain_overlap",
    "domain_table_markdown",
    "element_moment_residual",
    "energy_work_residual",
    "envelope_exceedance",
    "error_amplification",
    "flat_plate_depth_force_n",
    "flat_plate_elements",
    "flat_plate_inertial_force_n",
    "friction_angle_leverage_per_degree",
    "grid_convergence_index",
    "inertial_power_is_dissipative",
    "leading_edge_fraction",
    "linear_impulse_residual",
    "moment_transfer_residual",
    "observed_order_from_errors",
    "observed_order_from_residuals",
    "plate_response_comparisons",
    "predicted_alpha_z_n_per_cm3",
    "quasi_static_plate_case",
    "quasi_static_solver",
    "reference_dataset",
    "refinement_errors",
    "require_measurable",
    "residual_series",
    "richardson_extrapolate",
    "spiral_body_elements",
    "surface_refinement_study",
    "two_grid_gci",
    "validate",
    "validation_report",
]
