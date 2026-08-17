"""Code verification for the F1 tier, through the existing V&V machinery.

ADR-0033 requires F1 to "produce a GCI study rather than a
single-resolution result", and the NASA-STD-7009B self-assessment in
:mod:`bunkershot3d.vandv.credibility` records **validation at level 0 of
4**.  A new solver that cannot demonstrate its own correctness makes that
score worse, not better, so this module exists before any picture does.

Nothing here uses experimental data.  It is *code verification* -- is the
maths right -- and it reuses
:mod:`bunkershot3d.vandv.conservation`,
:mod:`bunkershot3d.vandv.convergence` and :mod:`bunkershot3d.vandv.gci`
rather than growing a second Richardson extrapolation beside the Celik
implementation that is already there.

The four checks, and why each is the one it is
----------------------------------------------

**Free fall.**  A closed bed under gravity with no intruder and no walls.
Mass is conserved exactly (particles never change mass, and the B-spline
partition of unity makes the P2G sum exact), and total momentum after
``n`` steps is exactly ``M g n dt`` because the internal forces sum to
zero for *any* stress field.  Both are identities of the scheme, so both
are round-off class.

**Elastic column.**  An unstressed column relaxed under its own weight to
static equilibrium, against the closed-form 1-D consolidation answer
``<sigma_zz> = -rho g H / 2``.  This is a real test rather than a
restatement of the initialisation: the solver's own bed seeding uses the
geostatic state, but this case **starts unstressed and has to find it**.
The equilibrium state is verified to be elastic, so the analytic elastic
answer is the exact solution of the elastoplastic model here, not an
approximation to it.

**Why not Bagnold.**  The obvious granular analytic case is the Bagnold
profile of steady inclined flow, and it is unavailable to this tier *by
construction*: it is a consequence of the ``mu(I)`` rate dependence, and
F1's constitutive model is deliberately rate-independent (see
:mod:`bunkershot3d.solvers.mpm.constitutive` for why).  A
rate-independent plastic solid on an incline gives plug flow over a basal
shear band, not a 3/2 power law.  Using Bagnold anyway would be testing a
rheology this solver does not have.

**Energy.**  Truncation class, not round-off: symplectic Euler conserves
energy only to ``O(dt)``, and the *order of the decay* is the test.  A
truncation residual can always be made small by shrinking ``dt``, which
says nothing, so :mod:`bunkershot3d.vandv.conservation` refuses to let it
be judged by a fixed tolerance and this module obeys that.

**F0 cross-check.**  ADR-0033 is explicit that this is a consistency
check between two uncalibrated models and **not** a validation: agreement
raises neither tier's validation level above 0, because neither is being
compared to a measurement.  What it can do is falsify, so the comparison
is reported as numbers -- including the ratio and where it comes from --
rather than reduced to a pass.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ...vandv.conservation import ConservationClass, ConservationResidual
from ...vandv.convergence import ObservedOrder, observed_order_from_errors
from ...vandv.gci import GCIResult, GridSolution, grid_convergence_index
from ..drft import DRFTSolver
from ..envelope import GRAVITY_M_S2, RefusalPolicy
from ..exceptions import SolverInputError
from ..protocol import IntrusionState, SolverResult
from .constitutive import SandContinuum, principal_stretches
from .grid import PlaneStrainGrid
from .solver import MPMRun, PlaneStrainMPMSolver
from .state import (
    DomainWalls,
    ParticleState,
    SurfaceDepression,
    WallCondition,
    settled_bed,
)

__all__ = [
    "ColumnEquilibrium",
    "F0CrossCheck",
    "column_grid_convergence",
    "cross_check_against_f0",
    "elastic_column_equilibrium",
    "energy_residuals",
    "free_fall_residuals",
    "mean_vertical_stress_pa",
]

_OPEN_DOMAIN = DomainWalls(
    lower_x=WallCondition.FREE,
    upper_x=WallCondition.FREE,
    lower_z=WallCondition.FREE,
    upper_z=WallCondition.FREE,
)
"""No walls at all: the closed-system configuration the momentum identity
is an identity in."""

_COLUMN_DOMAIN = DomainWalls(
    lower_x=WallCondition.SLIP,
    upper_x=WallCondition.SLIP,
    lower_z=WallCondition.STICKY,
    upper_z=WallCondition.FREE,
)
"""The 1-D consolidation column: laterally confined, fixed base, free top."""


def _free_solver(
    material: SandContinuum,
    cell_size_m: float,
    *,
    walls: DomainWalls,
    gravity_m_s2: float,
    max_steps: int,
) -> PlaneStrainMPMSolver:
    """A solver configured for a verification case rather than a shot."""
    return PlaneStrainMPMSolver(
        material=material,
        cell_size_m=cell_size_m,
        effective_width_m=1.0,
        bed_depth_m=1.0,
        walls=walls,
        gravity_m_s2=gravity_m_s2,
        refusal_policy=RefusalPolicy.REPORT,
        max_steps=max_steps,
    )


def _particles(
    material: SandContinuum,
    *,
    cell_size_m: float,
    width_m: float,
    height_m: float,
    particles_per_cell_axis: int = 2,
    geostatic: bool = False,
) -> ParticleState:
    """A rectangular block of particles occupying ``[0, W] x [0, H]``."""
    return settled_bed(
        material,
        x_bounds_m=(0.0, width_m),
        free_surface_height_m=height_m,
        depth_m=height_m,
        cell_size_m=cell_size_m,
        particles_per_cell_axis=particles_per_cell_axis,
        geostatic=geostatic,
    )


def _open_grid(
    *, cell_size_m: float, width_m: float, height_m: float
) -> PlaneStrainGrid:
    """A grid with room on every side, for a case that has no walls at all."""
    return PlaneStrainGrid.covering(
        (0.0, 0.0), (width_m, height_m), cell_size_m, pad_cells=4
    )


def _column_grid(
    *, cell_size_m: float, width_m: float, height_m: float
) -> PlaneStrainGrid:
    """A grid whose wall bands land **on** the column's own boundaries.

    This is the whole of the boundary condition and it is easy to get
    silently wrong.  :class:`~bunkershot3d.solvers.mpm.state.DomainWalls`
    acts over the outermost two node layers of the grid, so a grid padded
    generously on every side puts the "fixed base" two cells *below* the
    column and the "confining" side walls two cells *outside* it.  The
    column then stands in free space and simply falls -- with an entirely
    plausible-looking run and a mean stress of zero.

    One cell of padding on the left, the right and the bottom puts the
    inner wall layer exactly on ``x = 0``, ``x = W`` and ``z = 0``, which
    are the 1-D consolidation conditions the analytic answer assumes.
    """
    size = float(cell_size_m)
    count_x = int(math.ceil(width_m / size)) + 3
    count_z = int(math.ceil(height_m / size)) + 4
    return PlaneStrainGrid((-size, -size), size, (count_x, count_z))


# --------------------------------------------------------------- free fall


def free_fall_residuals(
    material: SandContinuum,
    *,
    cell_size_m: float = 0.004,
    width_m: float = 0.024,
    height_m: float = 0.024,
    n_steps: int = 40,
    gravity_m_s2: float = GRAVITY_M_S2,
) -> tuple[ConservationResidual, ...]:
    """Mass and momentum residuals of a closed bed in free fall.

    An unstressed block with no intruder and no walls.  Nothing can
    generate stress -- ``F`` stays the identity because the grid velocity
    field is uniform -- so the only force is gravity and both residuals
    are exact identities of the scheme:

    * ``sum_p m_p`` never changes, and P2G sums it exactly by the
      partition of unity;
    * ``sum_p m_p v_p`` after ``n`` steps is exactly ``M g n dt``, because
      ``sum_i grad w_ip = 0`` makes the internal forces sum to zero for
      any stress field whatsoever.

    Both are therefore **round-off class**, and neither carries a step
    size: :class:`~bunkershot3d.vandv.conservation.ConservationResidual`
    refuses a step size on a round-off residual precisely so nobody fits
    an order to floating-point noise.

    Args:
        material: The continuum.
        cell_size_m: Grid ``dx``.
        width_m: Block width.
        height_m: Block height.
        n_steps: Steps to fall.
        gravity_m_s2: Gravitational acceleration.

    Returns:
        ``(mass_residual, vertical_momentum_residual)``.
    """
    particles = _particles(
        material,
        cell_size_m=cell_size_m,
        width_m=width_m,
        height_m=height_m,
    )
    grid = _open_grid(cell_size_m=cell_size_m, width_m=width_m, height_m=height_m)
    solver = _free_solver(
        material,
        cell_size_m,
        walls=_OPEN_DOMAIN,
        gravity_m_s2=gravity_m_s2,
        max_steps=max(n_steps, 1),
    )
    initial_mass = particles.total_mass_kg
    step_s = 0.4 * cell_size_m / material.elastic_wave_speed_m_s
    run = solver.march(
        particles,
        None,
        grid,
        n_steps=n_steps,
        time_step_s=step_s,
        free_surface_height_m=height_m,
        bed_x_bounds_m=(0.0, width_m),
    )

    final = run.steps[-1]
    expected_momentum = -initial_mass * gravity_m_s2 * n_steps * step_s
    return (
        ConservationResidual(
            name="F1 free fall: total particle mass",
            conservation_class=ConservationClass.ROUND_OFF,
            residual=abs(final.total_mass_kg_per_m - initial_mass),
            scale=initial_mass,
        ),
        ConservationResidual(
            name="F1 free fall: vertical linear momentum",
            conservation_class=ConservationClass.ROUND_OFF,
            residual=abs(float(final.linear_momentum_kg_m_s[1]) - expected_momentum),
            scale=abs(expected_momentum),
        ),
    )


# ------------------------------------------------------------------ energy


def energy_residuals(
    material: SandContinuum,
    *,
    courant_numbers: Sequence[float] = (0.4, 0.2, 0.1),
    cell_size_m: float = 0.004,
    width_m: float = 0.024,
    height_m: float = 0.024,
    duration_s: float = 4.0e-4,
    gravity_m_s2: float = GRAVITY_M_S2,
) -> tuple[ConservationResidual, ...]:
    """Energy residuals of a closed bed in free fall, one per timestep.

    What this covers, and what it deliberately does not
    ---------------------------------------------------

    It measures the **time integrator's** energy behaviour.  Symplectic
    Euler applied to ``v <- v + g dt``, ``x <- x + v dt`` loses exactly
    ``M g^2 dt^2 / 2`` of ``KE + m g z`` per step, so over a fixed window
    ``T`` the drift is ``M g^2 dt T / 2`` -- first order in ``dt``, with a
    closed form, and with no plasticity anywhere because the grid
    velocity field is uniform and ``F`` stays the identity.

    It does **not** cover the elastic energy exchange, and the reason is
    a real property of the material rather than an oversight.  The
    obvious case -- a pre-compressed block released to oscillate -- does
    not work for cohesionless sand: the block rebounds past its rest
    state into tension, the Drucker-Prager return map correctly
    annihilates the deviatoric strain at the cone tip, and the resulting
    energy loss is *physical plastic dissipation*, not truncation error.
    Measured on a first attempt at this case: 129 of 144 particles
    yielding per step.  Fitting an order to that would be fitting an
    order to the plasticity.  The elastic pathway is covered instead by a
    static check --
    :attr:`ColumnEquilibrium.analytic_elastic_energy_j_per_m` -- which
    needs no conservation argument.

    Args:
        material: The continuum.
        courant_numbers: Courant numbers to run, giving the step series.
        cell_size_m: Grid ``dx``.
        width_m: Block width.
        height_m: Block height.
        duration_s: Physical time each run covers, held fixed so the runs
            compare the same event at different steps.
        gravity_m_s2: Gravitational acceleration.

    Returns:
        One truncation-class residual per Courant number, each carrying
        its step size because the order of the decay *is* its test.

    Raises:
        SolverInputError: If any particle yields, which would make the
            residual physical dissipation rather than truncation error.
    """
    residuals: list[ConservationResidual] = []
    for number in courant_numbers:
        particles = _particles(
            material,
            cell_size_m=cell_size_m,
            width_m=width_m,
            height_m=height_m,
        )
        grid = _open_grid(cell_size_m=cell_size_m, width_m=width_m, height_m=height_m)
        step_s = float(number) * cell_size_m / material.elastic_wave_speed_m_s
        n_steps = max(int(round(duration_s / step_s)), 2)
        solver = _free_solver(
            material,
            cell_size_m,
            walls=_OPEN_DOMAIN,
            gravity_m_s2=gravity_m_s2,
            max_steps=n_steps,
        )
        run = solver.march(
            particles,
            None,
            grid,
            n_steps=n_steps,
            time_step_s=step_s,
            free_surface_height_m=height_m,
            bed_x_bounds_m=(0.0, width_m),
        )
        yielded = sum(step.n_yielded for step in run.steps)
        if yielded:
            raise SolverInputError(
                f"{yielded} particle(s) yielded during the free-fall energy "
                "case. A uniformly falling block cannot deform, so this means "
                "the transfer is not reproducing a uniform velocity field and "
                "the residual would no longer be truncation error"
            )
        total = np.array(
            [
                step.kinetic_energy_j_per_m
                + step.elastic_energy_j_per_m
                + step.gravitational_energy_j_per_m
                for step in run.steps
            ]
        )
        scale = float(np.abs(total).max())
        residuals.append(
            ConservationResidual(
                name=f"F1 free fall: total energy at C={number:g}",
                conservation_class=ConservationClass.TRUNCATION,
                residual=float(np.abs(total - total[0]).max()),
                scale=scale if scale > 0.0 else 1.0,
                step_size_s=step_s,
            )
        )
    return tuple(residuals)


# ---------------------------------------------------------- elastic column


def mean_vertical_stress_pa(material: SandContinuum, particles: ParticleState) -> float:
    """Volume-weighted mean vertical Cauchy stress over a particle set."""
    left, stretches, _ = principal_stretches(particles.deformation_gradient)
    strain = np.log(stretches)
    trace = strain.sum(axis=1)
    kirchhoff = (
        2.0 * material.shear_modulus_pa * strain
        + material.lame_lambda_pa * trace[:, None]
    )
    jacobian = stretches.prod(axis=1)
    principal = kirchhoff / jacobian[:, None]
    vertical = np.einsum("nk,nk,nk->n", left[:, 1, :], principal, left[:, 1, :])
    volume = particles.initial_volume_m2 * jacobian
    return float((vertical * volume).sum() / volume.sum())


@dataclass(frozen=True, slots=True)
class ColumnEquilibrium:
    """One relaxed elastic column, against its closed-form answer.

    Attributes:
        cell_size_m: The grid the column was solved on.
        mean_vertical_stress_pa: Solved volume-weighted ``<sigma_zz>``.
        analytic_mean_vertical_stress_pa: ``-rho g H / 2``.
        residual_kinetic_energy_j_per_m: What was left moving at the end.
        initial_kinetic_scale_j_per_m: The largest kinetic energy the
            relaxation passed through, so the residual has something to be
            judged against.
        n_yielded: Particles the return map moved on the final step. Must
            be zero for the elastic answer to be the exact one.
        n_steps: Steps taken.
        n_particles: Particles in the column.
    """

    cell_size_m: float
    mean_vertical_stress_pa: float
    analytic_mean_vertical_stress_pa: float
    elastic_energy_j_per_m: float
    analytic_elastic_energy_j_per_m: float
    residual_kinetic_energy_j_per_m: float
    initial_kinetic_scale_j_per_m: float
    n_yielded: int
    n_steps: int
    n_particles: int

    @property
    def elastic_energy_relative_error(self) -> float:
        """Error in the stored strain energy against ``W (rho g)^2 H^3 / (6 M)``.

        The elastic pathway's check.  It is static, so it needs no
        conservation argument -- which matters because the conservative
        elastic case a cohesionless sand would need does not exist (see
        :func:`energy_residuals`).
        """
        return abs(
            self.elastic_energy_j_per_m - self.analytic_elastic_energy_j_per_m
        ) / abs(self.analytic_elastic_energy_j_per_m)

    @property
    def absolute_error_pa(self) -> float:
        """Absolute error against the analytic answer."""
        return abs(self.mean_vertical_stress_pa - self.analytic_mean_vertical_stress_pa)

    @property
    def relative_error(self) -> float:
        """Error as a fraction of the analytic answer."""
        return self.absolute_error_pa / abs(self.analytic_mean_vertical_stress_pa)

    @property
    def relaxed_fraction(self) -> float:
        """How far the kinetic energy fell from its peak. Near zero is settled."""
        if self.initial_kinetic_scale_j_per_m <= 0.0:
            return 0.0
        return self.residual_kinetic_energy_j_per_m / self.initial_kinetic_scale_j_per_m

    def summary(self) -> str:
        """A line fit for a verification report."""
        return (
            f"dx={self.cell_size_m * 1e3:.4g} mm, {self.n_particles} particles: "
            f"<sigma_zz> = {self.mean_vertical_stress_pa:.6g} Pa against "
            f"{self.analytic_mean_vertical_stress_pa:.6g} Pa analytic "
            f"({self.relative_error:.3%}), relaxed to "
            f"{self.relaxed_fraction:.2e} of peak KE"
        )


def elastic_column_equilibrium(
    material: SandContinuum,
    *,
    cell_size_m: float,
    width_m: float = 0.024,
    height_m: float = 0.048,
    gravity_m_s2: float = GRAVITY_M_S2,
    damping_per_step: float = 0.02,
    n_steps: int | None = None,
) -> ColumnEquilibrium:
    """Relax an unstressed column under its own weight and compare.

    Laterally confined by slip walls and fixed at the base, so the exact
    solution is the 1-D consolidation state
    ``sigma_zz(z) = -rho g (H - z)``, whose volume average is
    ``-rho g H / 2``.  That state is inside the Drucker-Prager cone for
    any sand this package can build -- the ``K_0`` stress ratio
    ``nu / (1 - nu)`` puts it well below the yield surface -- so the
    *elastic* answer is the exact solution of the elastoplastic model and
    not an approximation to it.  ``n_yielded`` is reported so that claim
    is checked rather than assumed.

    Args:
        material: The continuum.
        cell_size_m: Grid ``dx``.
        width_m: Column width.
        height_m: Column height.
        gravity_m_s2: Gravitational acceleration.
        damping_per_step: Nodal velocity damping. Relaxation to a
            *static* answer needs it; it vanishes as the velocity does,
            so it cannot move the equilibrium it is converging to.
        n_steps: Steps to relax. Defaults to enough elastic transits of
            the column for the damping to have acted.

    Returns:
        The relaxed column.
    """
    particles = _particles(
        material,
        cell_size_m=cell_size_m,
        width_m=width_m,
        height_m=height_m,
    )
    grid = _column_grid(cell_size_m=cell_size_m, width_m=width_m, height_m=height_m)
    step_s = 0.4 * cell_size_m / material.elastic_wave_speed_m_s
    transit_s = height_m / material.elastic_wave_speed_m_s
    steps = (
        int(math.ceil(24.0 * transit_s / step_s)) if n_steps is None else int(n_steps)
    )
    solver = _free_solver(
        material,
        cell_size_m,
        walls=_COLUMN_DOMAIN,
        gravity_m_s2=gravity_m_s2,
        max_steps=steps,
    )
    run = solver.march(
        particles,
        None,
        grid,
        n_steps=steps,
        time_step_s=step_s,
        free_surface_height_m=height_m,
        bed_x_bounds_m=(0.0, width_m),
        damping_per_step=damping_per_step,
    )
    peak_kinetic = max(step.kinetic_energy_j_per_m for step in run.steps)
    unit_weight = material.density_kg_m3 * gravity_m_s2
    return ColumnEquilibrium(
        cell_size_m=float(cell_size_m),
        mean_vertical_stress_pa=mean_vertical_stress_pa(material, particles),
        analytic_mean_vertical_stress_pa=-unit_weight * height_m / 2.0,
        elastic_energy_j_per_m=run.steps[-1].elastic_energy_j_per_m,
        # U = W int_0^H sigma_zz^2 / (2 M) dz = W (rho g)^2 H^3 / (6 M)
        analytic_elastic_energy_j_per_m=(
            width_m * unit_weight**2 * height_m**3 / (6.0 * material.p_wave_modulus_pa)
        ),
        residual_kinetic_energy_j_per_m=run.steps[-1].kinetic_energy_j_per_m,
        initial_kinetic_scale_j_per_m=peak_kinetic,
        n_yielded=run.steps[-1].n_yielded,
        n_steps=run.n_steps,
        n_particles=particles.n_particles,
    )


def column_grid_convergence(
    material: SandContinuum,
    *,
    cell_sizes_m: Sequence[float] = (0.006, 0.004, 0.003),
    width_m: float = 0.024,
    height_m: float = 0.048,
    gravity_m_s2: float = GRAVITY_M_S2,
) -> tuple[tuple[ColumnEquilibrium, ...], ObservedOrder, GCIResult]:
    """Refine the elastic column and report the order and the GCI.

    The default cell sizes divide the column height exactly at two
    particles per cell per axis, which matters: a size that leaves a
    partial particle layer changes the *discrete* column's height by a
    rounding rather than by the discretisation, and a rounding artefact
    is not a smooth function of ``dx``, so the observed order would be
    measuring the ``round`` call.

    Args:
        material: The continuum.
        cell_sizes_m: Grid sizes, coarsest first. Three at least, since
            Celik's apparent order needs three levels.
        width_m: Column width.
        height_m: Column height.
        gravity_m_s2: Gravitational acceleration.

    Returns:
        ``(levels, observed_order, gci)``.

    Raises:
        SolverInputError: If fewer than three sizes are supplied.
    """
    sizes = [float(size) for size in cell_sizes_m]
    if len(sizes) < 3:
        raise SolverInputError(
            f"a GCI study needs at least three grids, got {len(sizes)}"
        )
    levels = tuple(
        elastic_column_equilibrium(
            material,
            cell_size_m=size,
            width_m=width_m,
            height_m=height_m,
            gravity_m_s2=gravity_m_s2,
        )
        for size in sizes
    )
    order = observed_order_from_errors(
        [level.cell_size_m for level in levels],
        [level.absolute_error_pa for level in levels],
    )
    gci = grid_convergence_index(
        [
            GridSolution(
                cell_size_m=level.cell_size_m,
                value=level.mean_vertical_stress_pa,
                label=f"dx={level.cell_size_m * 1e3:.3g} mm",
            )
            for level in levels
        ],
        quantity="F1 elastic column mean vertical stress",
    )
    return levels, order, gci


# ----------------------------------------------------------- F0 comparison


@dataclass(frozen=True, slots=True)
class F0CrossCheck:
    """F1 and F0 on the quantities both produce, and where they diverge.

    ADR-0033 states plainly what this is: **a consistency check between
    two uncalibrated models, not a validation.**  Agreement raises
    neither tier's NASA-STD-7009B validation level above 0, because
    neither is being compared to a measurement.  Disagreement beyond a
    declared band, on the other hand, means at least one of them is
    wrong, and that is worth knowing.

    Attributes:
        speed_m_s: Intrusion speed the pair was run at.
        f0_force_n: ``(3,)`` F0 resultant force.
        f1_force_n: ``(3,)`` F1 resultant force, at the declared width.
        f0_depth_force_n: F0's depth-linear part.
        f0_inertial_force_n: F0's dynamic part.
        f1_stress_force_n: F1's stress-and-weight part.
        f1_flux_force_n: F1's momentum-flux part.
        submerged_depth_m: Deepest submerged element of the **shared
            query**, read off the geometry and therefore identical for
            both tiers. It exists because ``f0_max_depth_m`` and
            ``f1_max_depth_m`` are *not* the same measurement, despite
            being the same protocol field: F0 reports its deepest
            **engaged** element -- leading-edge and submerged, the contact
            diagnostic issue #8701 warned against conflating with a
            geometric depth -- while F1 reports the deepest submerged
            element outright. On the flat sole section this cross-check
            was written for they coincide; on a lofted head, most of whose
            elements never lead, they differ by an order of magnitude.
            Anything comparing depth *across* the tiers must use this
            field and not those two.
        f0_max_depth_m: F0's own ``SolverResult.max_depth_m``: its deepest
            **engaged** element. Not monotone, and zero whenever nothing
            meets the engagement criterion.
        f1_max_depth_m: F1's own ``SolverResult.max_depth_m``: the deepest
            submerged element, engaged or not.
        f1_divot: F1's free-surface depression -- depth *and* section area
            -- which F0 cannot produce at all. F0's own "divot" is the
            swept lower envelope of the head, so the two are different
            measurements of the same word and the comparison has to say so
            (issue #8713).
        effective_width_m: The declared width F1's magnitude rests on.
    """

    speed_m_s: float
    f0_force_n: NDArray[np.float64]
    f1_force_n: NDArray[np.float64]
    f0_depth_force_n: NDArray[np.float64]
    f0_inertial_force_n: NDArray[np.float64]
    f1_stress_force_n: NDArray[np.float64]
    f1_flux_force_n: NDArray[np.float64]
    submerged_depth_m: float
    f0_max_depth_m: float
    f1_max_depth_m: float
    f1_divot: SurfaceDepression
    effective_width_m: float

    @property
    def f1_divot_depth_m(self) -> float:
        """Deepest the sand's own free surface fell [m]."""
        return self.f1_divot.max_depth_m

    @property
    def f1_divot_section_area_m2(self) -> float:
        """Section of sand F1 removed from below the original surface [m^2].

        Per unit out-of-plane width, so it compares directly against
        :attr:`~bunkershot3d.metrics.divot.DivotMetrics.section_area_m2`
        without either side declaring a width. The mass does need one; see
        :meth:`~bunkershot3d.solvers.mpm.state.SurfaceDepression.displaced_mass_kg`.
        """
        return self.f1_divot.section_area_m2

    @property
    def magnitude_ratio(self) -> float:
        """``|F1| / |F0|``. Meaningful only alongside the declared width."""
        f0 = float(np.linalg.norm(self.f0_force_n))
        if f0 <= 0.0:
            return math.inf
        return float(np.linalg.norm(self.f1_force_n)) / f0

    @property
    def direction_agreement(self) -> float:
        """Cosine between the two resultant force directions.

        The most transferable part of the comparison: it does not depend
        on the effective width at all, so it is the one number here that
        is not conditional on a modelling assumption.
        """
        f0_norm = float(np.linalg.norm(self.f0_force_n))
        f1_norm = float(np.linalg.norm(self.f1_force_n))
        if f0_norm <= 0.0 or f1_norm <= 0.0:
            return 0.0
        return float(self.f0_force_n @ self.f1_force_n) / (f0_norm * f1_norm)

    @property
    def f0_inertial_fraction(self) -> float:
        """Share of F0's force carried by its dynamic term."""
        depth = float(np.linalg.norm(self.f0_depth_force_n))
        inertial = float(np.linalg.norm(self.f0_inertial_force_n))
        return inertial / (depth + inertial) if depth + inertial > 0.0 else 0.0

    @property
    def f1_flux_fraction(self) -> float:
        """Share of F1's reaction carried by momentum flux."""
        stress = float(np.linalg.norm(self.f1_stress_force_n))
        flux = float(np.linalg.norm(self.f1_flux_force_n))
        return flux / (stress + flux) if stress + flux > 0.0 else 0.0

    def summary(self) -> str:
        """A paragraph fit for a verification report."""
        return (
            f"v={self.speed_m_s:.4g} m/s, width={self.effective_width_m * 1e3:.4g} mm: "
            f"|F0|={float(np.linalg.norm(self.f0_force_n)):.4g} N, "
            f"|F1|={float(np.linalg.norm(self.f1_force_n)):.4g} N, "
            f"ratio={self.magnitude_ratio:.3g}, "
            f"direction cos={self.direction_agreement:.4f}; "
            f"inertial share F0={self.f0_inertial_fraction:.3f} vs flux share "
            f"F1={self.f1_flux_fraction:.3f}; submerged depth "
            f"{self.submerged_depth_m * 1e3:.3g} mm (F0 engaged "
            f"{self.f0_max_depth_m * 1e3:.3g} mm, F1 reports "
            f"{self.f1_max_depth_m * 1e3:.3g} mm); F1 divot "
            f"{self.f1_divot_depth_m * 1e3:.3g} mm deep, "
            f"{self.f1_divot_section_area_m2 * 1e4:.3g} cm^2 in section "
            "(F0 moves no sand, so it produces neither)"
        )


def cross_check_against_f0(
    state: IntrusionState,
    f0_solver: DRFTSolver,
    f1_solver: PlaneStrainMPMSolver,
) -> F0CrossCheck:
    """Run both tiers on one query and report where they diverge.

    Args:
        state: The intrusion query, given to both tiers unchanged.
        f0_solver: The F0 solver. Its refusal policy should be permissive,
            since a bunker shot is far outside its stated envelope and a
            strict policy would refuse before producing anything to
            compare.
        f1_solver: The F1 solver.

    Returns:
        The comparison.
    """
    f0_result: SolverResult = f0_solver.solve(state)
    run = f1_solver.run(state)
    window = min(f1_solver.averaging_window_s, run.duration_s)
    stress_part, flux_part = run.force_split(window)
    width = f1_solver.effective_width_m
    total = (stress_part + flux_part) * width
    depths = -state.element_depths_m()
    submerged = max(float(depths.max()) if depths.size else 0.0, 0.0)

    return F0CrossCheck(
        speed_m_s=state.speed_m_s,
        f0_force_n=np.asarray(f0_result.wrench.force_n, dtype=np.float64),
        f1_force_n=np.array([total[0], 0.0, total[1]]),
        f0_depth_force_n=np.asarray(f0_result.depth_force_n, dtype=np.float64),
        f0_inertial_force_n=np.asarray(f0_result.inertial_force_n, dtype=np.float64),
        f1_stress_force_n=np.array(
            [stress_part[0] * width, 0.0, stress_part[1] * width]
        ),
        f1_flux_force_n=np.array([flux_part[0] * width, 0.0, flux_part[1] * width]),
        submerged_depth_m=submerged,
        f0_max_depth_m=f0_result.max_depth_m,
        f1_max_depth_m=max(float(depths.max()) if depths.size else 0.0, 0.0),
        f1_divot=run.surface_depression(),
        effective_width_m=width,
    )


def _unused(run: MPMRun) -> None:  # pragma: no cover - typing anchor
    """Keep :class:`MPMRun` imported for annotations under ``from __future__``."""
    del run
