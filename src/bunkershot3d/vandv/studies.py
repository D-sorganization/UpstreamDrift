"""The concrete V&V studies this package can actually run (issue #8616).

Everything above this module is machinery.  This is where it is pointed
at the F0 solver, and where the answers stop being flattering.

What can be run
---------------

* :func:`surface_refinement_study` -- solution verification.  A three-
  or four-grid GCI over the cylinder case, giving ``u_num`` for the
  surface quadrature.
* :func:`plate_response_comparisons` -- the **only** validation that can
  be formed at all: the material-scaling cubic's prediction of the
  vertical-plate response against the 2.02 N/cm^3 measured on the
  Quikrete medium-sand analogue.  It comes out **noise-limited**, and it
  stays noise-limited even if the measurement is granted zero
  uncertainty, because the response is 12-13% per degree of friction
  angle and the friction angle is not known to a degree.
* :func:`carry_correlation_comparison` -- the machinery for the Wivou
  et al. (2016) carry correlations, ready for a model correlation that
  **does not yet exist**.  Nothing in this package computes carry, so
  this validation has not been performed and the credibility statement
  says so.

What cannot be run, and is not faked
------------------------------------

Ball launch angle, ball speed, ball spin, clubhead deceleration in sand,
the energy split, ejecta mass and the coefficient of restitution through
a sand layer.  There is no published measurement of any of them.
:func:`~bunkershot3d.vandv.reference_data.require_measurable` refuses
each one, so a comparison against them cannot even be constructed.

The plate-drag law ``F = K|z| + lambda rho A v^2`` with ``K = 580 N/m``
is *also* not usable, for a duller reason: ``K`` is proportional to the
plate area and the area it was measured on is not recorded in the
research digest.  A comparison against it would be a comparison against
an unknown scale factor.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from ..solvers import (
    VERTICAL_PLATE_ALPHA_Z,
    DRFTSolver,
    MaterialResponse,
    internal_friction_mu,
    material_scaling_pa_per_m,
)
from .cases import cylinder_case
from .exceptions import VerificationError
from .gci import GCIStudy, GridSolution, grid_convergence_index
from .reference_data import WIVOU_2016
from .validation import NumericalUncertainty, ValidationComparison

__all__ = [
    "QUIKRETE_ANALOGUE_SOURCE",
    "QUIKRETE_BULK_DENSITY_KG_M3",
    "QUIKRETE_FRICTION_ANGLE_DEG",
    "QUIKRETE_FRICTION_ANGLE_UNCERTAINTY_DEG",
    "QUIKRETE_MEASURED_ALPHA_Z_N_PER_CM3",
    "QUIKRETE_PACKING_FRACTION_UNCERTAINTY",
    "WIVOU_CONTROLLED_VARIABLES",
    "carry_correlation_comparison",
    "correlation_standard_uncertainty",
    "friction_angle_leverage_per_degree",
    "plate_response_comparisons",
    "predicted_alpha_z_n_per_cm3",
    "quikrete_internal_friction_mu",
    "surface_refinement_study",
]

QUIKRETE_MEASURED_ALPHA_Z_N_PER_CM3 = 2.02
"""The measured vertical-plate response of the Quikrete medium-sand analogue."""

QUIKRETE_FRICTION_ANGLE_DEG = 34.0
"""Internal friction angle of the same analogue."""

QUIKRETE_PACKING_FRACTION = 0.6
"""Packing fraction of the same analogue."""

QUIKRETE_GRAIN_DENSITY_KG_M3 = 2600.0
"""Grain density of the same analogue."""

QUIKRETE_BULK_DENSITY_KG_M3 = QUIKRETE_PACKING_FRACTION * QUIKRETE_GRAIN_DENSITY_KG_M3
"""``rho_c = phi rho_grain`` for the analogue: 1560 kg/m^3."""

QUIKRETE_FRICTION_ANGLE_UNCERTAINTY_DEG = 1.0
"""Standard uncertainty assumed on the friction angle.

The source quotes ``Phi = 34 deg`` with no error bar. One degree is the
band spanned by the independent quartz values in the research digest
(``phi_crit`` about 33 deg, Richefeu et al. about 33 deg for sand), and
it is stated here as an assumption rather than smuggled in as a fact."""

QUIKRETE_PACKING_FRACTION_UNCERTAINTY = 0.02
"""Standard uncertainty assumed on the packing fraction, likewise stated."""

QUIKRETE_ANALOGUE_SOURCE = (
    "Agarwal, Senatore, Zhang, Kingsbury, Iagnemma, Goldman & Kamrin, "
    "J. Terramechanics (2019), arXiv:1901.10667: alpha_z(0, pi/2) = "
    "2.02 N/cm^3 measured on Quikrete medium sand, 0.3-0.8 mm"
)

WIVOU_CONTROLLED_VARIABLES = 2
"""Variables assumed held constant behind Wivou's r = -0.98.

The paper says "with other variables held constant" and does not say how
many. Two is an assumption; it costs two degrees of freedom and it is
recorded here rather than buried."""


def predicted_alpha_z_n_per_cm3(
    *, bulk_density_kg_m3: float, friction_angle_deg: float
) -> float:
    """The material-scaling cubic's prediction of the plate response.

    ``xi_n alpha_z_generic(0, pi/2, 0)``, converted to the N/cm^3 unit the
    measurement is published in.

    Args:
        bulk_density_kg_m3: ``rho_c``.
        friction_angle_deg: ``Phi``.

    Returns:
        The predicted response in N/cm^3.
    """
    scale = material_scaling_pa_per_m(
        bulk_density_kg_m3=bulk_density_kg_m3, friction_angle_deg=friction_angle_deg
    )
    return scale * VERTICAL_PLATE_ALPHA_Z / 1e6


def _alpha_z_input_uncertainty() -> float:
    """Propagate the analogue's input spread through the scaling cubic.

    Central differences on the two inputs, combined in quadrature because
    packing fraction and friction angle are measured independently.  The
    friction-angle term dominates by a factor of about four: the cubic
    moves roughly 12-13% per degree, which is the substance of the
    finding rather than an incidental detail.

    Returns:
        The standard uncertainty on the predicted response, N/cm^3.
    """
    nominal = {
        "bulk_density_kg_m3": QUIKRETE_BULK_DENSITY_KG_M3,
        "friction_angle_deg": QUIKRETE_FRICTION_ANGLE_DEG,
    }
    density_step = QUIKRETE_PACKING_FRACTION_UNCERTAINTY * QUIKRETE_GRAIN_DENSITY_KG_M3
    density_term = 0.5 * (
        predicted_alpha_z_n_per_cm3(
            bulk_density_kg_m3=nominal["bulk_density_kg_m3"] + density_step,
            friction_angle_deg=nominal["friction_angle_deg"],
        )
        - predicted_alpha_z_n_per_cm3(
            bulk_density_kg_m3=nominal["bulk_density_kg_m3"] - density_step,
            friction_angle_deg=nominal["friction_angle_deg"],
        )
    )
    angle_step = QUIKRETE_FRICTION_ANGLE_UNCERTAINTY_DEG
    angle_term = 0.5 * (
        predicted_alpha_z_n_per_cm3(
            bulk_density_kg_m3=nominal["bulk_density_kg_m3"],
            friction_angle_deg=nominal["friction_angle_deg"] + angle_step,
        )
        - predicted_alpha_z_n_per_cm3(
            bulk_density_kg_m3=nominal["bulk_density_kg_m3"],
            friction_angle_deg=nominal["friction_angle_deg"] - angle_step,
        )
    )
    return math.hypot(density_term, angle_term)


def plate_response_comparisons() -> tuple[ValidationComparison, ValidationComparison]:
    """The only validation this package can form, stated twice.

    The first comparison is the honest one: the source reports **no**
    uncertainty on the 2.02 N/cm^3 measurement, so ``u_exp`` is unknown
    and the verdict is *indeterminate*.  Treating an unreported ``u_exp``
    as zero would claim the measurement was exact.

    The second is the same comparison with ``u_exp = 0`` granted
    deliberately -- the most favourable assumption available to the model.
    It still comes out **noise-limited**, which is a stronger statement
    than the first: the comparison carries no information about model
    error even under the assumption that most flatters it.

    ``u_h`` is zero for both, and legitimately so: the flat-plate
    traction is uniform, so the surface quadrature is exact at any
    discretisation and there is no discretisation error to estimate.

    Returns:
        ``(as_published, with_measurement_granted_exact)``.
    """
    predicted = predicted_alpha_z_n_per_cm3(
        bulk_density_kg_m3=QUIKRETE_BULK_DENSITY_KG_M3,
        friction_angle_deg=QUIKRETE_FRICTION_ANGLE_DEG,
    )
    numerical = NumericalUncertainty(u_h=0.0, u_it=0.0, u_ro=0.0)
    u_input = _alpha_z_input_uncertainty()
    shared_notes = (
        "u_h is exactly zero because the flat-plate traction is uniform, so "
        "the surface quadrature is exact at any discretisation.",
        "u_input is dominated by the friction angle: the scaling cubic moves "
        "about 12-13% per degree, and Phi is quoted without an error bar.",
        "This validates the material-scaling cubic against a laboratory "
        "analogue sand. It says nothing about golf bunker sand, about a "
        "wedge, or about any speed above quasi-static.",
    )
    as_published = ValidationComparison(
        quantity="vertical_plate_alpha_z_n_per_cm3",
        unit="N/cm^3",
        simulation_value=predicted,
        experiment_value=QUIKRETE_MEASURED_ALPHA_Z_N_PER_CM3,
        numerical=numerical,
        u_input=u_input,
        u_exp=None,
        reference=QUIKRETE_ANALOGUE_SOURCE,
        notes=(
            *shared_notes,
            "The source reports no uncertainty on the measurement, so u_val "
            "cannot be formed and the verdict is indeterminate.",
        ),
    )
    granted_exact = ValidationComparison(
        quantity="vertical_plate_alpha_z_n_per_cm3",
        unit="N/cm^3",
        simulation_value=predicted,
        experiment_value=QUIKRETE_MEASURED_ALPHA_Z_N_PER_CM3,
        numerical=numerical,
        u_input=u_input,
        u_exp=0.0,
        reference=QUIKRETE_ANALOGUE_SOURCE + " (u_exp granted as exactly zero)",
        notes=(
            *shared_notes,
            "u_exp = 0 is deliberately the most favourable assumption "
            "available to the model. The comparison is noise-limited even so.",
        ),
    )
    return (as_published, granted_exact)


def surface_refinement_study(
    solver: DRFTSolver,
    material: MaterialResponse,
    *,
    facet_counts: Sequence[int] = (64, 128, 256, 512),
    speed_m_s: float = 25.0,
) -> GCIStudy:
    """Solution verification of the DRFT surface quadrature.

    Args:
        solver: The solver under study.
        material: Sand response constants, for the exact reference.
        facet_counts: Circumferential facet counts, ascending, each a
            multiple of four.
        speed_m_s: Intrusion speed.

    Returns:
        The GCI study for the streamwise inertial force.

    Raises:
        VerificationError: If fewer than three refinement levels are
            given, since a three-grid GCI needs three.
    """
    if len(facet_counts) < 3:
        raise VerificationError(
            f"a grid-convergence study needs at least three levels, got "
            f"{len(facet_counts)}"
        )
    solutions = []
    for count in facet_counts:
        case = cylinder_case(material, n_facets=count, speed_m_s=speed_m_s)
        force = solver.solve(case.state()).inertial_force_n[0]
        solutions.append(GridSolution(case.cell_size_m, float(force), f"N={count}"))
    return GCIStudy(
        (grid_convergence_index(solutions, quantity="cylinder inertial force, x"),)
    )


def correlation_standard_uncertainty(
    *, correlation_r: float, n_samples: int, n_controlled_variables: int = 0
) -> float:
    """Standard uncertainty on a Pearson ``r`` via the Fisher ``z`` transform.

    ``se_z = 1/sqrt(n - 3 - k)`` and ``dr/dz = 1 - r^2``, so
    ``u_r = (1 - r^2) se_z``.  The delta-method step is an approximation
    and is worst exactly where ``|r|`` is near one, which is where the
    published bunker correlations sit -- so the value is conservative in
    the wrong direction and should be read as indicative.

    Args:
        correlation_r: The correlation.
        n_samples: Number of paired observations.
        n_controlled_variables: Variables held constant, each costing a
            degree of freedom.

    Returns:
        The standard uncertainty on ``r``.

    Raises:
        VerificationError: If ``|r| >= 1`` or the degrees of freedom are
            not positive.
    """
    correlation = float(correlation_r)
    if not math.isfinite(correlation) or abs(correlation) >= 1.0:
        raise VerificationError(
            f"a correlation must lie strictly inside (-1, 1) for the Fisher "
            f"transform, got {correlation_r!r}"
        )
    degrees = int(n_samples) - 3 - int(n_controlled_variables)
    if degrees <= 0:
        raise VerificationError(
            f"n = {n_samples} with {n_controlled_variables} controlled "
            "variable(s) leaves no degrees of freedom for a Fisher interval"
        )
    return (1.0 - correlation**2) / math.sqrt(degrees)


def carry_correlation_comparison(
    *,
    factor: str,
    model_correlation_r: float,
    u_num: float = 0.0,
    u_input: float = 0.0,
) -> ValidationComparison:
    """Compare a model carry correlation with Wivou et al. (2016).

    **Nothing in this package computes a model carry correlation yet**, so
    this comparison has not been performed.  The machinery exists so that
    it can be, and so that the credibility statement can name precisely
    what is missing rather than gesturing at "future work".

    Args:
        factor: ``entry_distance_behind_ball_m`` or ``divot_depth_m``.
        model_correlation_r: The correlation this model produces.
        u_num: Numerical uncertainty on the model correlation.
        u_input: Input uncertainty on the model correlation.

    Returns:
        The comparison, ready for :func:`~bunkershot3d.vandv.validate`.

    Raises:
        VerificationError: If the factor has no published correlation.
    """
    published = WIVOU_2016.correlations.get(factor)
    if published is None:
        raise VerificationError(
            f"Wivou et al. (2016) publishes no carry correlation for "
            f"{factor!r}; available factors are "
            f"{sorted(WIVOU_2016.correlations)}"
        )
    samples = WIVOU_2016.n_samples or 0
    u_exp = correlation_standard_uncertainty(
        correlation_r=published,
        n_samples=samples,
        n_controlled_variables=WIVOU_CONTROLLED_VARIABLES,
    )
    return ValidationComparison(
        quantity=f"carry_correlation_{factor}",
        unit="dimensionless",
        simulation_value=float(model_correlation_r),
        experiment_value=float(published),
        numerical=NumericalUncertainty(u_h=float(u_num)),
        u_input=float(u_input),
        u_exp=u_exp,
        reference=WIVOU_2016.citation,
        notes=(
            f"u_exp is a Fisher-z interval on n = {samples} with "
            f"{WIVOU_CONTROLLED_VARIABLES} controlled variable(s) assumed; "
            "the paper states neither the degrees of freedom nor an interval.",
            "Lower magnitude is more forgiving, so this comparison is a "
            "design baseline as well as a validation target.",
            "The model correlation must come from a carry pipeline. This "
            "package does not have one, so this comparison has not been run.",
        ),
    )


def friction_angle_leverage_per_degree() -> float:
    """Fractional change in the predicted plate response per degree of ``Phi``.

    Reported because it is the reason the one available validation is
    noise-limited: a model whose answer moves 12-13% per degree cannot be
    confirmed or refuted by a 5% agreement with a measurement whose
    friction angle is quoted to the nearest degree.

    Returns:
        The fractional sensitivity per degree.
    """
    nominal = predicted_alpha_z_n_per_cm3(
        bulk_density_kg_m3=QUIKRETE_BULK_DENSITY_KG_M3,
        friction_angle_deg=QUIKRETE_FRICTION_ANGLE_DEG,
    )
    raised = predicted_alpha_z_n_per_cm3(
        bulk_density_kg_m3=QUIKRETE_BULK_DENSITY_KG_M3,
        friction_angle_deg=QUIKRETE_FRICTION_ANGLE_DEG + 1.0,
    )
    lowered = predicted_alpha_z_n_per_cm3(
        bulk_density_kg_m3=QUIKRETE_BULK_DENSITY_KG_M3,
        friction_angle_deg=QUIKRETE_FRICTION_ANGLE_DEG - 1.0,
    )
    return 0.5 * (raised - lowered) / nominal


def quikrete_internal_friction_mu() -> float:
    """``mu_int = tan(Phi)`` for the analogue, for reports."""
    return internal_friction_mu(QUIKRETE_FRICTION_ANGLE_DEG)
