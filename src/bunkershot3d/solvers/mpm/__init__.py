"""The F1 tier: a 2-D plane-strain material-point sand solver (ADR-0033).

F0 (:mod:`bunkershot3d.solvers.drft`) integrates an empirical resistive
stress over the intruder's swept surface.  It never forms a sand
velocity, at any resolution, so no amount of post-processing yields the
*fields* Track B of epic #8699 consumes.  A continuum solve produces them
by construction: every cell carries a velocity, a density and a stress
because those are the solution variables.

What this package is
--------------------

* :mod:`.constitutive` -- rate-independent **Drucker-Prager**
  elastoplasticity with a compressive cap read off the sand's own packing
  state, and the reason ``mu(I)`` was rejected (it is ill-posed at both
  low and high inertial number, so refinement makes it worse).
* :mod:`.grid` -- the Eulerian background grid, quadratic B-splines and
  the **APIC** transfers, which conserve linear *and* angular momentum
  where PIC damps and FLIP rings.
* :mod:`.state` -- particles, bed initialisation and the domain walls.
* :mod:`.body` -- the clubhead as a rigid moving plane-strain section,
  with the contact treatment that stops material tunnelling through it.
* :mod:`.ball` -- the ball as a second such section, an **infinite
  cylinder rather than a sphere**, whose below-equator / face-side
  split is reported qualitatively and whose launch stays refused.
* :mod:`.contact` -- the sand against **several** bodies in one step, and
  the stated projection order that makes a shared node's answer
  independent of the caller's argument list.
* :mod:`.step` -- the scheme itself, one step at a time, so a whole-shot
  march can drive it without reaching into the solver.
* :mod:`.solver` -- :class:`~bunkershot3d.solvers.mpm.solver.PlaneStrainMPMSolver`,
  which implements the :class:`~bunkershot3d.solvers.protocol.GranularSolver`
  protocol so F1 is swappable with F0.
* :mod:`.envelope` -- the F1 validity verdict, its caveats, and the
  quantities ADR-0033 refuses outright.
* :mod:`.verification` -- conservation residuals, the analytic case, the
  grid-convergence study and the F0 cross-check.

Verified is not validated
-------------------------

The :mod:`.verification` suite shows this solver **solves its equations
correctly**: mass exactly, momentum to round-off, energy at first order
in the step, a 1.6% match to a closed-form consolidation answer, and
monotone grid convergence with a 1.1% GCI.  None of that is evidence that
the equations describe golf bunker sand.  Validation for this package
stands at NASA-STD-7009B level **0 of 4** because issue #8616 found no
published measurement of any quantity it produces, so every verdict this
tier issues is :attr:`~bunkershot3d.solvers.envelope.EnvelopeStatus.BEYOND_VALIDATION`
and cannot be better.

What it is not
--------------

Plane strain has **no out-of-plane flow**: sand moving heel-to-toe along
the face does not exist in this model, and no refinement adds it.  F1 is
specified at bulk resolution (``dx ~ 1-2 mm``) for the 10-100 mm flow
features, so the ~0.5 mm leading edge is deliberately under-resolved and
**club force remains F0's to report** --
:func:`~bunkershot3d.solvers.mpm.envelope.require_quotable` raises rather
than leaving that to documentation.
"""

from __future__ import annotations

from .ball import (
    BALL_DIAMETER_M,
    BALL_RADIUS_M,
    DEFAULT_BALL_FACETS,
    MIN_BALL_FACETS,
    PLANE_STRAIN_BALL_NOTE,
    BallContactSplit,
    BallSection,
    circular_section,
    n_facets_for_cell_size,
)
from .body import ContactImpulse, RigidSection, convex_hull_2d, plane_torque_about_y
from .contact import (
    BodyContact,
    apply_body_contacts,
    contact_order,
    push_out_bodies,
)
from .constitutive import (
    HARDIN_RICHART_ANGULAR_COEFFICIENT_KPA,
    HARDIN_RICHART_ROUND_COEFFICIENT_KPA,
    PLANE_STRAIN_DIMENSION,
    SAND_POISSON_RATIO,
    SandContinuum,
    drucker_prager_alpha,
    hencky_kirchhoff_principal,
    principal_stretches,
    project_to_yield_surface,
    reconstruct,
    yield_function,
)
from .envelope import (
    F1_STANDING_CAVEATS,
    MIN_CELLS_PER_GRAIN,
    MIN_CELLS_PER_RESOLVED_FEATURE,
    RefusedQuantity,
    evaluate_f1_envelope,
    require_quotable,
)
from .grid import (
    NODES_PER_PARTICLE,
    STENCIL_WIDTH,
    GridInterpolation,
    PlaneStrainGrid,
    apic_angular_momentum,
    cross_2d,
)
from .solver import (
    DEFAULT_CFL_NUMBER,
    MPMRun,
    MPMSetup,
    PlaneStrainMPMSolver,
    StepDiagnostics,
    cfl_time_step_s,
)
from .state import (
    DomainWalls,
    ParticleState,
    SurfaceDepression,
    WallCondition,
    settled_bed,
    surface_depression,
    surface_profile_m,
)
from .step import StepContext, advance_step
from .verification import (
    ColumnEquilibrium,
    F0CrossCheck,
    column_grid_convergence,
    cross_check_against_f0,
    elastic_column_equilibrium,
    energy_residuals,
    free_fall_residuals,
)

__all__ = [
    "BALL_DIAMETER_M",
    "BALL_RADIUS_M",
    "BallContactSplit",
    "BallSection",
    "BodyContact",
    "ColumnEquilibrium",
    "ContactImpulse",
    "DEFAULT_BALL_FACETS",
    "DEFAULT_CFL_NUMBER",
    "DomainWalls",
    "F0CrossCheck",
    "F1_STANDING_CAVEATS",
    "GridInterpolation",
    "HARDIN_RICHART_ANGULAR_COEFFICIENT_KPA",
    "HARDIN_RICHART_ROUND_COEFFICIENT_KPA",
    "MIN_BALL_FACETS",
    "MIN_CELLS_PER_GRAIN",
    "MIN_CELLS_PER_RESOLVED_FEATURE",
    "MPMRun",
    "MPMSetup",
    "NODES_PER_PARTICLE",
    "PLANE_STRAIN_BALL_NOTE",
    "PLANE_STRAIN_DIMENSION",
    "ParticleState",
    "PlaneStrainGrid",
    "PlaneStrainMPMSolver",
    "RefusedQuantity",
    "RigidSection",
    "SAND_POISSON_RATIO",
    "STENCIL_WIDTH",
    "SandContinuum",
    "StepContext",
    "StepDiagnostics",
    "SurfaceDepression",
    "WallCondition",
    "advance_step",
    "apic_angular_momentum",
    "apply_body_contacts",
    "cfl_time_step_s",
    "circular_section",
    "column_grid_convergence",
    "contact_order",
    "convex_hull_2d",
    "cross_2d",
    "cross_check_against_f0",
    "drucker_prager_alpha",
    "elastic_column_equilibrium",
    "energy_residuals",
    "evaluate_f1_envelope",
    "free_fall_residuals",
    "hencky_kirchhoff_principal",
    "n_facets_for_cell_size",
    "plane_torque_about_y",
    "principal_stretches",
    "project_to_yield_surface",
    "push_out_bodies",
    "reconstruct",
    "require_quotable",
    "settled_bed",
    "surface_depression",
    "surface_profile_m",
    "yield_function",
]
