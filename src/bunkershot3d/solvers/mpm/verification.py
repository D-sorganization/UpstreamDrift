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
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ...vandv.conservation import ConservationClass, ConservationResidual
from ...vandv.convergence import ObservedOrder, observed_order_from_errors
from ...vandv.gci import GCIResult, GridSolution, grid_convergence_index
from ..drft import DRFTSolver
from ..elements import SurfaceElements
from ..envelope import GRAVITY_M_S2, RefusalPolicy
from ..exceptions import SolverInputError
from ..protocol import IntrusionState, SolverResult
from .constitutive import (
    PLANE_STRAIN_DIMENSION,
    SandContinuum,
    principal_stretches,
)
from .grid import PlaneStrainGrid
from .solver import MPMRun, PlaneStrainMPMSolver
from .state import (
    DomainWalls,
    ParticleState,
    SurfaceDepression,
    WallCondition,
    settled_bed,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to a checker
    from .body import RigidSection

__all__ = [
    "DEFAULT_MANUFACTURED_FIELD",
    "DESIGN_ORDER_SPATIAL",
    "ColumnEquilibrium",
    "F0CrossCheck",
    "ManufacturedField",
    "ManufacturedLevel",
    "ManufacturedSolutionStudy",
    "PassiveWallLimit",
    "RankineLimits",
    "TemporalLevel",
    "TemporalStudy",
    "column_grid_convergence",
    "column_temporal_convergence",
    "cross_check_against_f0",
    "elastic_column_equilibrium",
    "energy_residuals",
    "free_fall_residuals",
    "manufactured_solution_convergence",
    "mean_vertical_stress_pa",
    "passive_earth_pressure_limit",
    "rankine_limits",
    "uniform_stress_patch_residual",
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


def _march_open_domain(
    solver: PlaneStrainMPMSolver,
    particles: ParticleState,
    grid: PlaneStrainGrid,
    *,
    n_steps: int,
    time_step_s: float,
    width_m: float,
    height_m: float,
) -> MPMRun:
    """March a free column with no intruder over an open domain.

    The conservation and energy-order cases differ only in how they choose
    ``n_steps`` and the step size; the march itself is the same call. Sharing
    it keeps the two studies measuring the same integrator, so a change to the
    boundary arguments cannot silently apply to one case and not the other.
    """
    return solver.march(
        particles,
        None,
        grid,
        n_steps=n_steps,
        time_step_s=time_step_s,
        free_surface_height_m=height_m,
        bed_x_bounds_m=(0.0, width_m),
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
    run = _march_open_domain(
        solver,
        particles,
        grid,
        n_steps=n_steps,
        time_step_s=step_s,
        width_m=width_m,
        height_m=height_m,
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
        run = _march_open_domain(
            solver,
            particles,
            grid,
            n_steps=n_steps,
            time_step_s=step_s,
            width_m=width_m,
            height_m=height_m,
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
    def f1_divot_fully_resolved(self) -> bool:
        """Whether every surface bin held sand, so the area is complete."""
        return self.f1_divot.fully_resolved

    @property
    def f1_divot_bins(self) -> tuple[int, int]:
        """``(empty, total)`` surface bins the divot was profiled on."""
        return (self.f1_divot.n_empty_bins, self.f1_divot.n_bins)

    def f1_divot_mass_kg(self, *, width_m: float, bulk_density_kg_m3: float) -> float:
        """F1's removed section as a mass, at a **declared** width.

        Args:
            width_m: Declared out-of-plane width [m].
            bulk_density_kg_m3: Sand bulk density [kg/m^3].

        Returns:
            The displaced mass [kg].

        Raises:
            SolverInputError: If either argument is not positive.
        """
        return self.f1_divot.displaced_mass_kg(
            width_m=width_m, bulk_density_kg_m3=bulk_density_kg_m3
        )

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


# ------------------------------------------------------- Rankine limit state


_TINY = float(np.finfo(np.float64).tiny)
"""Smallest positive normal double, as a divide-by-zero floor."""

_PASSIVE_PAD_CELLS = 10
"""Cells of grid to the left of the layer, so the wall has room to start."""

_PASSIVE_STEP_CAP = 100000
"""Step cap for the quasi-static push; a limit load needs many small steps."""


_SLIP_LAYER_DOMAIN = DomainWalls(
    lower_x=WallCondition.FREE,
    upper_x=WallCondition.SLIP,
    lower_z=WallCondition.SLIP,
    upper_z=WallCondition.FREE,
)
"""The Rankine layer: no wall on the loaded side, no shear on the base.

A frictionless base is not a convenience.  It is what makes the Rankine
field ``sigma_xz = 0``, ``sigma_zz = rho g (H - z)``,
``sigma_xx = K sigma_zz`` a *statically admissible* limit field for the
whole layer rather than only for an assumed wedge: with no basal shear,
horizontal equilibrium reduces to ``d sigma_xx / dx = 0``, which that
field satisfies identically.  A sticky base adds an unmodelled shear
traction and was measured to raise the thrust by 24% on the same grid.
"""

_MIN_PASSIVE_LENGTH_RATIO = 3.0
"""Bed length, in wall heights, below which the layer is too short.

The Rankine passive wedge behind a wall of height ``H`` reaches
``H tan(45 + phi/2)`` -- about ``1.7 H`` here -- so a bed shorter than
twice that is being compressed against its far wall rather than pushed
to a limit state."""

_MAX_QUASI_STATIC_RATIO = 1.0e-3
"""Largest ``v_wall / c`` a static limit load may be read at."""

_MIN_MOBILISED_FRACTION = 0.5
"""Smallest share of the bed that must be at yield for a *limit* load."""


@dataclass(frozen=True, slots=True)
class RankineLimits:
    """The plane-strain Coulomb limits of *this* cone, not of the input angle.

    F1's yield surface is written on the **two in-plane** principal
    Kirchhoff stresses, ``||dev(tau)|| + alpha (tr(tau) - tr(tau)_tip)
    <= 0`` with ``dev`` and ``tr`` taken over ``d = 2`` components.  In
    that space a two-dimensional state ``(tau_1, tau_2)`` has
    ``||dev(tau)|| = |tau_1 - tau_2| / sqrt(2)``, so the limit condition
    is

    ``|tau_1 - tau_2| / sqrt(2) = alpha (tr(tau)_tip - tau_1 - tau_2)``

    which, written in the soil-mechanics sign convention with
    ``s = sqrt(2) alpha``, is exactly Mohr-Coulomb at an equivalent
    plane-strain friction angle ``phi* = asin(s)``:

    ``sigma_h = K sigma_v -/+ s T / (1 -/+ s)``,
    ``K_a = (1 - s)/(1 + s)``, ``K_p = (1 + s)/(1 - s)``.

    ``phi*`` is **not** the friction angle handed to
    :func:`~bunkershot3d.solvers.mpm.constitutive.drucker_prager_alpha`.
    That function fits the inner (compressive-meridian) cone to
    Mohr-Coulomb in *three* dimensions; used in plane strain the same
    ``alpha`` corresponds to a softer angle, 31.94 deg for the 34 deg
    input every sand preset in this package carries.  A verification
    case that compared against ``(1 - sin 34)/(1 + sin 34)`` would be
    comparing against a criterion this solver does not enforce, and
    would report a 4% "error" that is in fact a definition.

    Attributes:
        alpha: The cone slope the limits were derived from.
        sin_phi_star: ``sqrt(2) alpha``, the sine of the equivalent
            plane-strain friction angle.
        active_coefficient: ``K_a``.
        passive_coefficient: ``K_p``.
        cone_tip_stress_pa: ``tr(tau)`` at the cone tip, ``(2 mu + d
            lambda) tr(eps)_tip``, zero for a cohesionless sand.
    """

    alpha: float
    sin_phi_star: float
    active_coefficient: float
    passive_coefficient: float
    cone_tip_stress_pa: float

    @property
    def phi_star_deg(self) -> float:
        """The equivalent plane-strain friction angle in degrees."""
        return math.degrees(math.asin(self.sin_phi_star))

    @property
    def passive_cohesive_intercept_pa(self) -> float:
        """``s T / (1 - s)``: the model's ``2 c sqrt(K_p)``."""
        return self.sin_phi_star * self.cone_tip_stress_pa / (1.0 - self.sin_phi_star)

    @property
    def active_cohesive_intercept_pa(self) -> float:
        """``s T / (1 + s)``: the model's ``2 c sqrt(K_a)``."""
        return self.sin_phi_star * self.cone_tip_stress_pa / (1.0 + self.sin_phi_star)

    def passive_thrust_n_per_m(
        self, *, height_m: float, unit_weight_n_m3: float
    ) -> float:
        """``K_p gamma H^2 / 2 + (s T / (1 - s)) H`` per unit width.

        Args:
            height_m: Retained height ``H``.
            unit_weight_n_m3: ``rho g``.

        Returns:
            The limit thrust in N/m.
        """
        height = float(height_m)
        return (
            0.5 * self.passive_coefficient * float(unit_weight_n_m3) * height * height
            + self.passive_cohesive_intercept_pa * height
        )

    def active_thrust_n_per_m(
        self, *, height_m: float, unit_weight_n_m3: float
    ) -> float:
        """``K_a gamma H^2 / 2 - (s T / (1 + s)) H`` per unit width.

        Args:
            height_m: Retained height ``H``.
            unit_weight_n_m3: ``rho g``.

        Returns:
            The limit thrust in N/m. Reported for completeness; the
            verification case here is the passive one, because a
            retreating wall has to be *followed* by the sand and a
            measured active thrust is therefore contaminated by however
            fast the wedge can fall. Measured on this solver: the active
            plateau came out at 0.78 of the closed form on a slip base
            and 0.32 on a sticky one, against 1.06 for the passive case
            on the same grid.
        """
        height = float(height_m)
        return (
            0.5 * self.active_coefficient * float(unit_weight_n_m3) * height * height
            - self.active_cohesive_intercept_pa * height
        )

    def summary(self) -> str:
        """A line fit for a verification report."""
        return (
            f"cone alpha={self.alpha:.5f} -> plane-strain phi*="
            f"{self.phi_star_deg:.3f} deg, K_a={self.active_coefficient:.5f}, "
            f"K_p={self.passive_coefficient:.5f}, cone tip "
            f"{self.cone_tip_stress_pa:.4g} Pa"
        )


def rankine_limits(material: SandContinuum) -> RankineLimits:
    """The active and passive limits the F1 cone actually enforces.

    Rankine, "On the stability of loose earth", *Phil. Trans. R. Soc.
    Lond.* **147**:9-27 (1857), specialised to this solver's
    two-dimensional Drucker-Prager surface as derived in
    :class:`RankineLimits`.

    Args:
        material: The continuum.

    Returns:
        The limits.

    Raises:
        SolverInputError: If ``sqrt(2) alpha >= 1``, where the cone is
            steeper than any Mohr-Coulomb criterion and no Rankine state
            exists: ``K_p`` would be negative or infinite.
    """
    sin_phi_star = math.sqrt(2.0) * float(material.alpha)
    if not math.isfinite(sin_phi_star) or sin_phi_star >= 1.0:
        raise SolverInputError(
            f"sqrt(2) alpha = {sin_phi_star:.6g} for alpha = {material.alpha!r}, so "
            "the cone has no plane-strain Mohr-Coulomb equivalent: no Rankine "
            "limit state exists and K_p would be negative or infinite"
        )
    dimension = PLANE_STRAIN_DIMENSION
    bulk = 2.0 * material.shear_modulus_pa + dimension * material.lame_lambda_pa
    return RankineLimits(
        alpha=float(material.alpha),
        sin_phi_star=sin_phi_star,
        active_coefficient=(1.0 - sin_phi_star) / (1.0 + sin_phi_star),
        passive_coefficient=(1.0 + sin_phi_star) / (1.0 - sin_phi_star),
        cone_tip_stress_pa=bulk * float(material.tip_volumetric_strain),
    )


@dataclass(frozen=True, slots=True)
class PassiveWallLimit:
    """One measured passive thrust against its closed-form limit load.

    Attributes:
        cell_size_m: Grid ``dx``.
        height_m: Retained height ``H``.
        length_m: Bed length.
        wall_speed_m_s: How fast the wall was driven.
        travel_m: How far it went.
        thrust_n_per_m: The plateau of the solver's own wall reaction.
        plateau_spread: Standard deviation of the plateau window as a
            fraction of its mean. The plateau is a plastic flow, not a
            static answer, so it fluctuates; reporting the fluctuation
            keeps a single averaged number from reading as a settled one.
        analytic_thrust_n_per_m: ``K_p gamma H^2 / 2`` plus the cone-tip
            intercept.
        passive_coefficient: ``K_p`` the analytic value used.
        cohesive_share: Fraction of the analytic thrust carried by the
            cone tip rather than by the friction cone. Near one means the
            case is testing cohesion, not friction.
        yielded_fraction: Share of particles the return map moved on the
            final step.
        quasi_static_ratio: ``v_wall / c``.
        n_particles: Particles in the bed.
        n_steps: Steps marched.
    """

    cell_size_m: float
    height_m: float
    length_m: float
    wall_speed_m_s: float
    travel_m: float
    thrust_n_per_m: float
    plateau_spread: float
    analytic_thrust_n_per_m: float
    passive_coefficient: float
    cohesive_share: float
    yielded_fraction: float
    quasi_static_ratio: float
    n_particles: int
    n_steps: int

    @property
    def absolute_error_n_per_m(self) -> float:
        """Absolute error against the closed-form limit load."""
        return abs(self.thrust_n_per_m - self.analytic_thrust_n_per_m)

    @property
    def relative_error(self) -> float:
        """Error as a fraction of the closed-form limit load."""
        return self.absolute_error_n_per_m / abs(self.analytic_thrust_n_per_m)

    def summary(self) -> str:
        """A line fit for a verification report."""
        return (
            f"dx={self.cell_size_m * 1e3:.4g} mm, H={self.height_m * 1e3:.4g} mm, "
            f"{self.n_particles} particles: P_p = {self.thrust_n_per_m:.6g} N/m "
            f"(+/-{self.plateau_spread:.1%} over the plateau) against "
            f"{self.analytic_thrust_n_per_m:.6g} N/m closed form "
            f"({self.relative_error:.3%}); K_p={self.passive_coefficient:.4f}, "
            f"cohesive share {self.cohesive_share:.3f}, "
            f"{self.yielded_fraction:.1%} of the bed at yield, "
            f"v/c = {self.quasi_static_ratio:.2e}"
        )


def _rankine_wall_section(
    solver: PlaneStrainMPMSolver,
    *,
    height_m: float,
    cell_size_m: float,
    speed_m_s: float,
) -> RigidSection:
    """The smooth rigid wall, built through the solver's own public route.

    The plate is taken past the toe *and* past the free surface so that
    no sand can escape under or over it, which is what the closed form
    assumes.  It is built by handing
    :meth:`~bunkershot3d.solvers.mpm.solver.PlaneStrainMPMSolver.section_from_state`
    a four-point body rather than by reaching into
    :mod:`~bunkershot3d.solvers.mpm.body`, so the case keeps working
    whatever the section class does internally.
    """
    thickness = 3.0 * cell_size_m
    overshoot = 2.0 * cell_size_m
    corners = np.array(
        [
            [-thickness, 0.0, -height_m - overshoot],
            [0.0, 0.0, -height_m - overshoot],
            [0.0, 0.0, overshoot],
            [-thickness, 0.0, overshoot],
        ]
    )
    normals = np.tile([1.0, 0.0, 0.0], (corners.shape[0], 1))
    areas = np.full(corners.shape[0], cell_size_m * cell_size_m)
    state = IntrusionState(
        SurfaceElements(corners, normals, areas),
        (speed_m_s, 0.0, 0.0),
        free_surface_height_m=0.0,
    )
    return solver.section_from_state(state)


def passive_earth_pressure_limit(
    material: SandContinuum,
    *,
    cell_size_m: float = 0.003,
    height_m: float = 0.030,
    length_ratio: float = 5.0,
    wall_speed_m_s: float = 0.02,
    travel_ratio: float = 0.006,
    plateau_fraction: float = 0.2,
    gravity_m_s2: float = GRAVITY_M_S2,
) -> PassiveWallLimit:
    """Push a smooth rigid wall into a layer and read its limit load.

    The plastic-limit case issue #8733 section 4 asks for.  Everything
    else in this module is elastic, kinematic, or a restatement of the
    yield function's own identity; this one drives the whole layer to the
    Drucker-Prager limit and compares the wall reaction **the solver
    computes for itself** -- ``StepDiagnostics.contact_force_n_per_m``,
    the same traction integration a club head is loaded through --
    against a closed form.

    The closed form and what it assumes
    -----------------------------------

    ``P_p = K_p rho g H^2 / 2 + (s T / (1 - s)) H`` per unit width, with
    ``K_p`` and the intercept from :func:`rankine_limits`.  It is exact
    for this constitutive model under five conditions, each of which is
    arranged here and reported rather than assumed:

    1. **Smooth wall.**  ``contact_friction = 0``, so the wall carries no
       shear and the principal directions stay axis-aligned.
    2. **Frictionless base.**  ``sigma_xz = 0`` everywhere makes
       horizontal equilibrium ``d sigma_xx / dx = 0``, so the Rankine
       field is statically admissible over the *whole* layer and the
       answer does not depend on where the failure surface goes.  This is
       also why the bed length barely matters: measured 1.0615 at 5H
       against 1.0573 at 8H, a 0.4% difference.
    3. **Quasi-static.**  ``v_wall / c = 1.6e-4`` at the defaults, and
       halving the wall speed moved the plateau by 0.25%.
    4. **Fully mobilised.**  Reported as ``yielded_fraction``; a load
       read off a bed that is still elastic is a stiffness, not a limit.
    5. **No escape route.**  The plate runs past the toe and past the
       free surface.

    Passive rather than active, and why
    -----------------------------------

    The active limit needs the sand to *follow* a retreating wall, so
    what is measured is the wedge's ability to fall rather than its
    strength.  Measured here: the active plateau reached 0.78 of the
    closed form on a frictionless base and 0.32 on a sticky one, while
    the passive case on the same grid reached 1.06.  The active number is
    reported by :meth:`RankineLimits.active_thrust_n_per_m` but is not
    the verification case.

    Args:
        material: The continuum. A cohesionless one puts all of the load
            on the friction cone; ``cohesive_share`` says how much of it
            the cone tip carried.
        cell_size_m: Grid ``dx``.
        height_m: Retained height ``H``.
        length_ratio: Bed length in multiples of ``H``.
        wall_speed_m_s: Wall speed.
        travel_ratio: Wall travel in multiples of ``H``.
        plateau_fraction: Trailing share of the march the plateau is
            averaged over.
        gravity_m_s2: Gravitational acceleration.

    Returns:
        The measured limit load beside its closed form.

    Raises:
        SolverInputError: If the wall is not quasi-static, if the bed is
            too short to hold a passive wedge, if the plateau window is
            not a usable fraction, or if the bed never reached its limit
            -- each of which would turn a limit load into something else
            while still returning a number.
    """
    limits = rankine_limits(material)
    size = float(cell_size_m)
    height = float(height_m)
    length = float(length_ratio) * height
    speed = float(wall_speed_m_s)
    ratio = speed / material.elastic_wave_speed_m_s
    if not 0.0 < float(plateau_fraction) <= 1.0:
        raise SolverInputError(
            f"plateau_fraction must lie in (0, 1], got {plateau_fraction!r}"
        )
    if not math.isfinite(ratio) or ratio > _MAX_QUASI_STATIC_RATIO:
        raise SolverInputError(
            f"the wall runs at v/c = {ratio:.3g}, over the "
            f"{_MAX_QUASI_STATIC_RATIO:g} this case treats as quasi-static. "
            "Rankine is a *static* limit; at a finite fraction of the wave "
            "speed the measured thrust carries an inertial term the closed "
            "form does not have, and the comparison would be against the "
            "wrong problem"
        )
    if float(length_ratio) < _MIN_PASSIVE_LENGTH_RATIO:
        raise SolverInputError(
            f"a bed {length_ratio!r} wall-heights long cannot hold the passive "
            f"wedge, which reaches H tan(45 + phi*/2) = "
            f"{math.tan(math.radians(45.0 + limits.phi_star_deg / 2.0)):.3g} H; "
            f"use at least {_MIN_PASSIVE_LENGTH_RATIO:g}"
        )

    solver = PlaneStrainMPMSolver(
        material=material,
        cell_size_m=size,
        effective_width_m=1.0,
        bed_depth_m=height,
        gravity_m_s2=gravity_m_s2,
        contact_friction=0.0,
        walls=_SLIP_LAYER_DOMAIN,
        refusal_policy=RefusalPolicy.REPORT,
        max_steps=_PASSIVE_STEP_CAP,
    )
    particles = settled_bed(
        material,
        x_bounds_m=(0.0, length),
        free_surface_height_m=0.0,
        depth_m=height,
        cell_size_m=size,
        particles_per_cell_axis=2,
        gravity_m_s2=gravity_m_s2,
        geostatic=True,
    )
    pad = _PASSIVE_PAD_CELLS
    grid = PlaneStrainGrid(
        (-(pad + 1) * size, -height - size),
        size,
        (int(round(length / size)) + pad + 3, int(round(height / size)) + 4),
    )
    section = _rankine_wall_section(
        solver, height_m=height, cell_size_m=size, speed_m_s=speed
    )
    step_s = 0.4 * size / material.elastic_wave_speed_m_s
    travel = float(travel_ratio) * height
    n_steps = max(int(math.ceil(travel / (speed * step_s))), 2)
    run = solver.march(
        particles,
        section,
        grid,
        n_steps=n_steps,
        time_step_s=step_s,
        free_surface_height_m=0.0,
        bed_x_bounds_m=(0.0, length),
    )

    reaction = np.abs(np.array([step.contact_force_n_per_m[0] for step in run.steps]))
    window = max(int(round(float(plateau_fraction) * reaction.size)), 1)
    plateau = float(reaction[-window:].mean())
    unit_weight = material.density_kg_m3 * gravity_m_s2
    analytic = limits.passive_thrust_n_per_m(
        height_m=height, unit_weight_n_m3=unit_weight
    )
    yielded_fraction = run.steps[-1].n_yielded / particles.n_particles
    if yielded_fraction < _MIN_MOBILISED_FRACTION:
        raise SolverInputError(
            f"only {yielded_fraction:.1%} of the bed was at yield on the final "
            "step, so the wall reaction is an elastic stiffness rather than a "
            "limit load. Increase travel_ratio until the plateau appears"
        )
    return PassiveWallLimit(
        cell_size_m=size,
        height_m=height,
        length_m=length,
        wall_speed_m_s=speed,
        travel_m=travel,
        thrust_n_per_m=plateau,
        plateau_spread=float(reaction[-window:].std()) / max(plateau, _TINY),
        analytic_thrust_n_per_m=analytic,
        passive_coefficient=limits.passive_coefficient,
        cohesive_share=(
            limits.passive_cohesive_intercept_pa * height / analytic
            if analytic > 0.0
            else 0.0
        ),
        yielded_fraction=float(yielded_fraction),
        quasi_static_ratio=ratio,
        n_particles=particles.n_particles,
        n_steps=run.n_steps,
    )


# ------------------------------------------- method of manufactured solutions


DESIGN_ORDER_SPATIAL = 2.0
"""Formal spatial order of the scheme: quadratic B-splines, 2x2 particles.

The grid basis is a quadratic B-spline, so its gradient reproduces a
linear field exactly and a smooth one to ``O(dx^2)``.  The particles are
a regular sub-cell lattice at the quarter points, which is a midpoint
rule per sub-cell and therefore also second order.  Standard MPM
degrades to first order once particles drift away from that lattice; the
study below is taken at the *initial* lattice and in a single step, so it
measures the operator rather than the drift, and the honest reading of a
result near 2 is "the operator is second order", not "the solver is"."""

_PATCH_MARGIN_CELLS = 3
"""Cells of material a particle needs on every side for full support.

The quadratic B-spline stencil is three nodes wide, so a node needs
particles 1.5 cells out on each side and a particle needs its nodes to
have that.  Measured: at two cells of margin the uniform-stress patch
residual is 4.0e-3 relative -- an incomplete stencil, not a bug -- and at
three it is 1.1e-15."""


@dataclass(frozen=True, slots=True)
class ManufacturedField:
    """A smooth, admissible elastic state with a closed-form divergence.

    The deformation gradient is manufactured **diagonal**,
    ``F = diag(exp(a), exp(b))``, which makes the principal directions
    the coordinate axes and the whole stress field elementary:

    ``eps = diag(a, b)``,
    ``tau_xx = 2 mu a + lambda (a + b)``, ``tau_zz = 2 mu b + lambda (a + b)``,
    ``J = exp(a + b)``, ``sigma = tau / J``, ``sigma_xz = 0``

    so ``div sigma = (d sigma_xx / dx, d sigma_zz / dz)`` and each term
    differentiates in closed form.  With

    ``a = -c + A sin(k x) cos(k z)``, ``b = -c + B cos(k x) sin(k z)``

    the mean compression ``c`` holds the state inside the friction cone
    while the amplitudes ``A`` and ``B`` supply the deviatoric part that
    makes the divergence non-trivial.  Nothing here is computed by the
    solver, so the "exact" answer is genuinely independent of it.

    A manufactured ``F`` needs no compatibility: MPM carries ``F`` as
    per-particle state, and the scheme never asks whether a displacement
    field generating it exists.

    Attributes:
        size_m: Side of the square block the field lives on.
        amplitude_x: ``A``.
        amplitude_z: ``B``.
        mean_compression: ``c``, positive, subtracted from both strains.
    """

    size_m: float
    amplitude_x: float
    amplitude_z: float
    mean_compression: float

    def __post_init__(self) -> None:
        """Validate.

        Raises:
            SolverInputError: If the block or the amplitudes are unusable.
        """
        for name, value in (
            ("size_m", self.size_m),
            ("amplitude_x", self.amplitude_x),
            ("amplitude_z", self.amplitude_z),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise SolverInputError(f"{name} must be positive, got {value!r}")
        if not math.isfinite(self.mean_compression) or self.mean_compression < 0.0:
            raise SolverInputError(
                "mean_compression must be non-negative -- it is what keeps the "
                f"manufactured state inside the cone -- got "
                f"{self.mean_compression!r}"
            )

    @property
    def wavenumber_per_m(self) -> float:
        """``k = 2 pi / L``: one full period across the block."""
        return 2.0 * math.pi / self.size_m

    def hencky_strains(
        self, x_m: NDArray[np.float64], z_m: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``(a, b)``, the two principal logarithmic strains.

        Args:
            x_m: ``(n,)`` horizontal positions.
            z_m: ``(n,)`` vertical positions.

        Returns:
            The two ``(n,)`` strain fields.
        """
        wave = self.wavenumber_per_m
        first = -self.mean_compression + self.amplitude_x * np.sin(wave * x_m) * np.cos(
            wave * z_m
        )
        second = -self.mean_compression + self.amplitude_z * np.cos(
            wave * x_m
        ) * np.sin(wave * z_m)
        return first, second

    def deformation_gradient(
        self, x_m: NDArray[np.float64], z_m: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """``(n, 2, 2)`` diagonal deformation gradients of the field.

        Args:
            x_m: ``(n,)`` horizontal positions.
            z_m: ``(n,)`` vertical positions.

        Returns:
            The gradients.
        """
        first, second = self.hencky_strains(x_m, z_m)
        gradient = np.zeros(
            (first.size, PLANE_STRAIN_DIMENSION, PLANE_STRAIN_DIMENSION)
        )
        gradient[:, 0, 0] = np.exp(first)
        gradient[:, 1, 1] = np.exp(second)
        return gradient

    def stress_divergence_pa_per_m(
        self,
        x_m: NDArray[np.float64],
        z_m: NDArray[np.float64],
        *,
        shear_modulus_pa: float,
        lame_lambda_pa: float,
    ) -> NDArray[np.float64]:
        """``(n, 2)`` exact ``div sigma`` of the manufactured field.

        ``sigma = tau exp(-t)`` with ``t = a + b``, so
        ``d sigma_xx / dx = exp(-t) [d tau_xx / dx - tau_xx t_x]`` and
        likewise in ``z``.

        Args:
            x_m: ``(n,)`` horizontal positions.
            z_m: ``(n,)`` vertical positions.
            shear_modulus_pa: Lame ``mu``.
            lame_lambda_pa: Lame ``lambda``.

        Returns:
            The divergence in Pa/m.
        """
        wave = self.wavenumber_per_m
        sin_x, cos_x = np.sin(wave * x_m), np.cos(wave * x_m)
        sin_z, cos_z = np.sin(wave * z_m), np.cos(wave * z_m)
        first = -self.mean_compression + self.amplitude_x * sin_x * cos_z
        second = -self.mean_compression + self.amplitude_z * cos_x * sin_z
        first_x = self.amplitude_x * wave * cos_x * cos_z
        second_x = -self.amplitude_z * wave * sin_x * sin_z
        second_z = self.amplitude_z * wave * cos_x * cos_z
        first_z = -self.amplitude_x * wave * sin_x * sin_z
        trace = first + second
        trace_x = first_x + second_x
        trace_z = first_z + second_z
        kirchhoff_xx = 2.0 * shear_modulus_pa * first + lame_lambda_pa * trace
        kirchhoff_zz = 2.0 * shear_modulus_pa * second + lame_lambda_pa * trace
        scale = np.exp(-trace)
        return np.stack(
            [
                scale
                * (
                    2.0 * shear_modulus_pa * first_x
                    + lame_lambda_pa * trace_x
                    - kirchhoff_xx * trace_x
                ),
                scale
                * (
                    2.0 * shear_modulus_pa * second_z
                    + lame_lambda_pa * trace_z
                    - kirchhoff_zz * trace_z
                ),
            ],
            axis=1,
        )

    def exact_acceleration_m_s2(
        self,
        x_m: NDArray[np.float64],
        z_m: NDArray[np.float64],
        *,
        material: SandContinuum,
        gravity_m_s2: float,
    ) -> NDArray[np.float64]:
        """``(n, 2)`` exact ``div sigma / rho + g`` of the field.

        ``rho`` is the *current* density ``rho_0 / J``, so the ``exp(-t)``
        in the divergence cancels and the answer is a bounded elementary
        expression.

        Args:
            x_m: ``(n,)`` horizontal positions.
            z_m: ``(n,)`` vertical positions.
            material: The continuum.
            gravity_m_s2: Gravitational acceleration.

        Returns:
            The acceleration in m/s^2.
        """
        first, second = self.hencky_strains(x_m, z_m)
        divergence = self.stress_divergence_pa_per_m(
            x_m,
            z_m,
            shear_modulus_pa=material.shear_modulus_pa,
            lame_lambda_pa=material.lame_lambda_pa,
        )
        density = material.density_kg_m3 * np.exp(-(first + second))
        acceleration = divergence / density[:, None]
        acceleration[:, 1] -= float(gravity_m_s2)
        return acceleration


DEFAULT_MANUFACTURED_FIELD = ManufacturedField(
    size_m=0.048, amplitude_x=1.0e-4, amplitude_z=0.7e-4, mean_compression=8.5e-4
)
"""The field the study runs on.

``mean_compression`` is five times ``A + B``, which puts the whole field
about 26 kPa inside the cone for every sand preset in this package -- a
margin the study checks rather than assumes.  The amplitudes are small
enough that ``J`` stays within 0.2% of one, so the manufactured state is
a perturbation of a uniformly compressed block rather than a large
deformation the Hencky model would have to be argued for."""


@dataclass(frozen=True, slots=True)
class ManufacturedLevel:
    """One grid of the manufactured-solution study.

    Attributes:
        cell_size_m: Grid ``dx``.
        error_rms_m_s2: Root-mean-square of the acceleration error over
            the interior particles.
        exact_rms_m_s2: Root-mean-square of the exact acceleration over
            the same particles, so the error has a scale.
        n_particles: Particles in the block.
        n_interior: Particles at least
            :data:`_PATCH_MARGIN_CELLS` coarse cells from every face.
        worst_yield_pa: Largest yield-function value over the block. Must
            be negative for the manufactured state to be elastic.
        n_yielded: Particles the return map moved. Must be zero.
    """

    cell_size_m: float
    error_rms_m_s2: float
    exact_rms_m_s2: float
    n_particles: int
    n_interior: int
    worst_yield_pa: float
    n_yielded: int

    @property
    def relative_error(self) -> float:
        """Error as a fraction of the exact field's own magnitude."""
        return self.error_rms_m_s2 / self.exact_rms_m_s2

    def summary(self) -> str:
        """A line fit for a verification report."""
        return (
            f"dx={self.cell_size_m * 1e3:.4g} mm, {self.n_interior} of "
            f"{self.n_particles} particles interior: RMS acceleration error "
            f"{self.error_rms_m_s2:.6g} m/s^2 against {self.exact_rms_m_s2:.6g} "
            f"m/s^2 exact ({self.relative_error:.3%}); worst yield "
            f"{self.worst_yield_pa:.4g} Pa, {self.n_yielded} yielded"
        )


@dataclass(frozen=True, slots=True)
class ManufacturedSolutionStudy:
    """The MMS result, and an explicit statement of its reach.

    Attributes:
        levels: One entry per grid, coarsest first.
        observed_order: Fitted order of the error decay.
        design_order: :data:`DESIGN_ORDER_SPATIAL`.
        field: The manufactured field.
    """

    levels: tuple[ManufacturedLevel, ...]
    observed_order: ObservedOrder
    design_order: float
    field: ManufacturedField

    def summary(self) -> str:
        """A paragraph fit for a verification report.

        It names what the study does *not* cover, because an MMS that is
        quoted without its scope reads as a whole-code verification and
        this one is not.
        """
        return (
            f"F1 MMS on the elastic stress divergence and the transfer, "
            f"{len(self.levels)} grids: {self.observed_order.summary()} against a "
            f"design order of {self.design_order:g}. Covers the P2G stress "
            "integration, the nodal solve and the G2P gather, taken together, "
            "on the elastic branch at zero velocity. Does NOT cover the "
            "Drucker-Prager return map, the deformation-gradient update, the "
            "contact projection or the time integration; the plastic branch "
            "has its own case in passive_earth_pressure_limit and the step has "
            "its own in column_temporal_convergence.\n  "
            + "\n  ".join(level.summary() for level in self.levels)
        )


def _manufactured_block(
    material: SandContinuum,
    field: ManufacturedField,
    *,
    cell_size_m: float,
    gravity_m_s2: float,
) -> tuple[ParticleState, MPMRun, NDArray[np.float64]]:
    """Seed a block with the manufactured field and take exactly one step.

    One step from rest is the whole trick: the nodal update is
    ``v_i = dt f_i / m_i`` and the gather is
    ``v_p = sum_i w_ip v_i``, so ``v_p / dt`` **is** the discrete
    operator -- P2G of the stress, divide by the lumped nodal mass, G2P
    -- applied to the manufactured stress field, and it is independent of
    ``dt`` to the last bit.  The gravity term survives the same path
    exactly, because ``m_i g / m_i = g`` at every live node and the
    B-spline partition of unity gathers a constant without error.

    Raises:
        SolverInputError: If the manufactured state is not admissible, or
            if the return map moved a particle -- either of which would
            make the residual plastic rather than truncation error.
    """
    particles = settled_bed(
        material,
        x_bounds_m=(0.0, field.size_m),
        free_surface_height_m=field.size_m,
        depth_m=field.size_m,
        cell_size_m=cell_size_m,
        particles_per_cell_axis=2,
        geostatic=False,
    )
    horizontal = particles.position_m[:, 0]
    vertical = particles.position_m[:, 1]
    particles.deformation_gradient = field.deformation_gradient(horizontal, vertical)
    first, second = field.hencky_strains(horizontal, vertical)
    worst = float(material.yield_value(np.stack([first, second], axis=1)).max())
    if worst >= 0.0:
        raise SolverInputError(
            f"the manufactured field reaches a yield value of {worst:.4g} Pa, so "
            "part of it is outside the cone. The return map would then move the "
            "state and the residual would be plastic projection rather than "
            "truncation error; raise mean_compression or lower the amplitudes"
        )
    grid = PlaneStrainGrid.covering(
        (0.0, 0.0), (field.size_m, field.size_m), cell_size_m, pad_cells=4
    )
    solver = _free_solver(
        material,
        cell_size_m,
        walls=_OPEN_DOMAIN,
        gravity_m_s2=gravity_m_s2,
        max_steps=1,
    )
    step_s = 0.4 * cell_size_m / material.elastic_wave_speed_m_s
    run = solver.march(
        particles,
        None,
        grid,
        n_steps=1,
        time_step_s=step_s,
        free_surface_height_m=field.size_m,
        bed_x_bounds_m=(0.0, field.size_m),
    )
    if run.steps[-1].n_yielded:
        raise SolverInputError(
            f"{run.steps[-1].n_yielded} particle(s) yielded on the manufactured "
            "step, so the state left the elastic branch the closed form is "
            "written for"
        )
    return particles, run, particles.velocity_m_s / step_s


def _interior(
    particles: ParticleState, *, size_m: float, margin_m: float
) -> NDArray[np.bool_]:
    """Particles at least ``margin_m`` from every face of the block."""
    horizontal = particles.position_m[:, 0]
    vertical = particles.position_m[:, 1]
    return (
        (horizontal >= margin_m)
        & (horizontal <= size_m - margin_m)
        & (vertical >= margin_m)
        & (vertical <= size_m - margin_m)
    )


def uniform_stress_patch_residual(
    material: SandContinuum,
    *,
    cell_size_m: float = 0.004,
    volumetric_strain: float = -1.6e-3,
    deviatoric_strain: float = 4.0e-4,
    size_m: float = 0.048,
    gravity_m_s2: float = GRAVITY_M_S2,
) -> ConservationResidual:
    """The patch test: a uniform stress field must produce no net force.

    The companion to the manufactured study and the thing that keeps it
    honest.  For a **spatially constant** stress the discrete internal
    force at a fully supported node is ``-sigma sum_p V_p grad w_ip``,
    and the B-spline gradient is odd about the node while the particle
    lattice is symmetric about it, so the sum is exactly zero.  Every
    interior particle therefore has to see exactly ``(0, -g)`` -- an
    identity of the scheme, so **round-off class**, with no step size and
    no order test.

    That matters because the manufactured study measures a *truncation*
    residual on the same operator: this one fixes the floor.  If the
    linear-completeness of the transfer were broken, the manufactured
    error would still fall under refinement and would still fit an
    order, and only this residual would notice.

    Args:
        material: The continuum.
        cell_size_m: Grid ``dx``.
        volumetric_strain: ``a + b``, negative so the state is
            compressive and inside the cone.
        deviatoric_strain: Half the difference ``a - b``, so the stress
            is not isotropic and the test cannot pass on a pressure alone.
        size_m: Side of the block.
        gravity_m_s2: Gravitational acceleration.

    Returns:
        The residual, round-off class.

    Raises:
        SolverInputError: If the uniform state is outside the cone, if
            the interior is empty, or if the return map moved anything.
    """
    first = 0.5 * float(volumetric_strain) + float(deviatoric_strain)
    second = 0.5 * float(volumetric_strain) - float(deviatoric_strain)
    field = ManufacturedField(
        size_m=size_m,
        amplitude_x=abs(first) if first else _TINY,
        amplitude_z=abs(second) if second else _TINY,
        mean_compression=0.0,
    )
    particles = settled_bed(
        material,
        x_bounds_m=(0.0, field.size_m),
        free_surface_height_m=field.size_m,
        depth_m=field.size_m,
        cell_size_m=cell_size_m,
        particles_per_cell_axis=2,
        geostatic=False,
    )
    count = particles.n_particles
    strain = np.tile([first, second], (count, 1))
    worst = float(material.yield_value(strain).max())
    if worst >= 0.0:
        raise SolverInputError(
            f"the uniform patch state has a yield value of {worst:.4g} Pa, so it "
            "is outside the cone and the return map would move it"
        )
    gradient = np.zeros((count, PLANE_STRAIN_DIMENSION, PLANE_STRAIN_DIMENSION))
    gradient[:, 0, 0] = math.exp(first)
    gradient[:, 1, 1] = math.exp(second)
    particles.deformation_gradient = gradient
    grid = PlaneStrainGrid.covering(
        (0.0, 0.0), (field.size_m, field.size_m), cell_size_m, pad_cells=4
    )
    solver = _free_solver(
        material,
        cell_size_m,
        walls=_OPEN_DOMAIN,
        gravity_m_s2=gravity_m_s2,
        max_steps=1,
    )
    step_s = 0.4 * cell_size_m / material.elastic_wave_speed_m_s
    run = solver.march(
        particles,
        None,
        grid,
        n_steps=1,
        time_step_s=step_s,
        free_surface_height_m=field.size_m,
        bed_x_bounds_m=(0.0, field.size_m),
    )
    if run.steps[-1].n_yielded:
        raise SolverInputError(
            f"{run.steps[-1].n_yielded} particle(s) yielded on the patch step"
        )
    acceleration = particles.velocity_m_s / step_s
    acceleration[:, 1] += float(gravity_m_s2)
    inside = _interior(
        particles,
        size_m=field.size_m,
        margin_m=_PATCH_MARGIN_CELLS * float(cell_size_m),
    )
    if not bool(inside.any()):
        raise SolverInputError(
            f"no particle is {_PATCH_MARGIN_CELLS} cells from every face at "
            f"dx = {cell_size_m!r} on a {size_m!r} m block, so every particle "
            "has an incomplete stencil and the patch test would be measuring "
            "the free surface"
        )
    # The scale a *wrong* internal force would show up at: one stress unit
    # spread over one cell.  Without it the residual is a bare number and a
    # tighter grid would look like a better result.
    kirchhoff = 2.0 * material.shear_modulus_pa * first + material.lame_lambda_pa * (
        first + second
    )
    return ConservationResidual(
        name="F1 patch test: net force of a uniform stress field",
        conservation_class=ConservationClass.ROUND_OFF,
        residual=float(np.abs(acceleration[inside]).max()),
        scale=abs(kirchhoff) / (material.density_kg_m3 * float(cell_size_m)),
    )


def manufactured_solution_convergence(
    material: SandContinuum,
    *,
    cell_sizes_m: Sequence[float] = (0.004, 0.003, 0.002, 0.0015),
    field: ManufacturedField = DEFAULT_MANUFACTURED_FIELD,
    gravity_m_s2: float = GRAVITY_M_S2,
) -> ManufacturedSolutionStudy:
    """Manufacture an elastic state, derive its forcing, and fit the order.

    What is manufactured is the **stress field**, through the deformation
    gradient; what is derived is the acceleration it must produce,
    ``div sigma / rho + g``, in closed form and without the solver.  The
    solver is then asked for exactly one step from rest, which turns the
    particle velocity into ``dt`` times the discrete operator, and the
    two are compared in a volume-free RMS over the interior.

    Why the interior only
    ---------------------

    A particle within about 1.5 cells of a face gathers from nodes whose
    own support runs off the material, where the lumped mass is a
    fraction of a cell's worth and the discrete divergence is not an
    approximation of anything.  The comparison is therefore taken over a
    **fixed physical** sub-block -- three coarse cells in from every face
    -- so that every level measures the same region rather than a region
    that shrinks with ``dx``.

    An honest partial
    -----------------

    This is MMS for the *elastic* branch.  It exercises the stress
    divergence and the particle-grid transfer together, which is what
    issue #8733 asks for, and it does not touch the Drucker-Prager return
    map: a manufactured field that yielded would be measuring the
    projection rather than the discretisation, and the study refuses one
    outright.  A full MMS for the plastic branch would need a
    manufactured *plastic* state whose consistency condition is satisfied
    pointwise, which the non-associated flow rule here does not admit in
    closed form.  What is verified is stated in
    :meth:`ManufacturedSolutionStudy.summary`, including the negative
    half.

    Args:
        material: The continuum.
        cell_sizes_m: Grids, coarsest first. Each must divide the block
            exactly, or the particle lattice changes by a rounding and
            the observed order measures the rounding.
        field: The manufactured field.
        gravity_m_s2: Gravitational acceleration.

    Returns:
        The study.

    Raises:
        SolverInputError: If fewer than three grids are supplied, if a
            grid does not divide the block, or if the manufactured state
            is not elastic everywhere.
    """
    sizes = [float(size) for size in cell_sizes_m]
    if len(sizes) < 3:
        raise SolverInputError(
            f"an order-of-accuracy study needs at least three grids, got {len(sizes)}"
        )
    margin = _PATCH_MARGIN_CELLS * max(sizes)
    levels: list[ManufacturedLevel] = []
    for size in sizes:
        cells = field.size_m / size
        if abs(cells - round(cells)) > 1e-9:
            raise SolverInputError(
                f"dx = {size!r} m does not divide the {field.size_m!r} m block a "
                "whole number of times, so the particle lattice would change by "
                "a rounding rather than by the discretisation and the observed "
                "order would be measuring the round() call"
            )
        particles, run, discrete = _manufactured_block(
            material, field, cell_size_m=size, gravity_m_s2=gravity_m_s2
        )
        horizontal = particles.position_m[:, 0]
        vertical = particles.position_m[:, 1]
        exact = field.exact_acceleration_m_s2(
            horizontal,
            vertical,
            material=material,
            gravity_m_s2=gravity_m_s2,
        )
        inside = _interior(particles, size_m=field.size_m, margin_m=margin)
        if not bool(inside.any()):
            raise SolverInputError(
                f"no particle survives a {margin:.4g} m interior margin at "
                f"dx = {size!r}; widen the block or coarsen the study"
            )
        difference = discrete[inside] - exact[inside]
        count = int(inside.sum())
        first, second = field.hencky_strains(horizontal, vertical)
        levels.append(
            ManufacturedLevel(
                cell_size_m=size,
                error_rms_m_s2=math.sqrt(float((difference**2).sum()) / count),
                exact_rms_m_s2=math.sqrt(float((exact[inside] ** 2).sum()) / count),
                n_particles=particles.n_particles,
                n_interior=count,
                worst_yield_pa=float(
                    material.yield_value(np.stack([first, second], axis=1)).max()
                ),
                n_yielded=run.steps[-1].n_yielded,
            )
        )
    return ManufacturedSolutionStudy(
        levels=tuple(levels),
        observed_order=observed_order_from_errors(
            [level.cell_size_m for level in levels],
            [level.error_rms_m_s2 for level in levels],
        ),
        design_order=DESIGN_ORDER_SPATIAL,
        field=field,
    )


# --------------------------------------------------- temporal grid refinement


@dataclass(frozen=True, slots=True)
class TemporalLevel:
    """One timestep of a temporal refinement study.

    Attributes:
        time_step_s: The step.
        courant_number: The Courant number it came from.
        n_steps: Steps taken to cover the fixed window.
        value: The quantity of interest at the end of the window.
        n_yielded: Particles the return map moved on the final step.
    """

    time_step_s: float
    courant_number: float
    n_steps: int
    value: float
    n_yielded: int

    def summary(self) -> str:
        """A line fit for a verification report."""
        return (
            f"C={self.courant_number:g}, dt={self.time_step_s:.4e} s, "
            f"{self.n_steps} steps: {self.value:.8g}"
        )


@dataclass(frozen=True, slots=True)
class TemporalStudy:
    """A step refinement at fixed ``dx``, with its order and its GCI.

    The shipped grid study refines ``dt`` alongside ``dx`` -- it holds the
    Courant number fixed, which is the only refinement an explicit scheme
    can take to convergence -- so its GCI is a *space-time* band and not a
    spatial one.  This study holds ``dx`` fixed and refines only the step,
    which is what isolates the integrator.

    Attributes:
        cell_size_m: The grid every level was solved on.
        duration_s: The fixed physical window.
        transits: That window in elastic transits of the column.
        levels: One entry per step, coarsest first.
        difference_order: Order fitted to the successive differences,
            ``|phi(dt/2) - phi(dt)| ~ dt^p``. Independent of the Celik
            iteration and reported beside it.
        gci: The Celik result, from the repository's own implementation.
    """

    cell_size_m: float
    duration_s: float
    transits: float
    levels: tuple[TemporalLevel, ...]
    difference_order: ObservedOrder
    gci: GCIResult

    @property
    def converging(self) -> bool:
        """Whether the triplet supports a Richardson estimate at all."""
        return self.gci.convergence.supports_richardson

    @property
    def apparent_order(self) -> float:
        """Celik's ``p`` for the triplet."""
        return self.gci.apparent_order

    def summary(self) -> str:
        """A paragraph fit for a verification report."""
        verdict = (
            "converging"
            if self.converging
            else (
                "NOT converging: the differences grow as the step is refined, "
                "which is the particle-grid round trip rather than the "
                "integrator -- its cost is paid per step, so over a fixed "
                "window the total grows as 1/dt and eventually overtakes the "
                "integrator's own O(dt) error"
            )
        )
        return (
            f"F1 temporal refinement at fixed dx={self.cell_size_m * 1e3:.4g} mm "
            f"over {self.transits:g} elastic transits ({self.duration_s:.4e} s), "
            f"{len(self.levels)} steps, {verdict}.\n  "
            + "\n  ".join(level.summary() for level in self.levels)
            + "\n  "
            + self.gci.summary().replace("\n", "\n  ")
            + f"\n  difference fit: {self.difference_order.summary()}"
        )


def column_temporal_convergence(
    material: SandContinuum,
    *,
    cell_size_m: float = 0.003,
    courant_numbers: Sequence[float] = (0.4, 0.2, 0.1),
    transits: float = 1.0,
    width_m: float = 0.024,
    height_m: float = 0.048,
    gravity_m_s2: float = GRAVITY_M_S2,
) -> TemporalStudy:
    """Refine the step on a fixed grid and report the order and the GCI.

    The case is the elastic column of :func:`elastic_column_equilibrium`
    with the damping removed and the window cut short: an unstressed
    column released under gravity, marched for a fixed physical time, and
    read for its volume-weighted ``<sigma_zz>``.  The damping has to go,
    because ``damping_per_step`` is applied **once per step** -- halving
    the step doubles the number of times it acts, so a damped study
    measures the damping schedule and not the integrator.  Measured on
    this solver with ``damping_per_step = 0.01``: the passive limit load
    moved from 1.06 to 2.67 times its closed form, and refining ``dx``
    made it monotonically worse.

    Why the window is short
    -----------------------

    Over a *fixed* window the particle-grid round trip is taken ``T/dt``
    times, and what it costs does not shrink with the step, so its
    accumulated effect grows as the step is refined.  Inside about one
    elastic transit the integrator's own ``O(dt)`` error still dominates
    and the series converges; past that it does not.  Both are reported:
    ``transits=1`` gives a monotonic triplet, ``transits=4`` gives
    ``MONOTONIC_DIVERGENCE`` and :attr:`TemporalStudy.converging` is
    ``False`` rather than the study quietly extrapolating anyway.

    Args:
        material: The continuum.
        cell_size_m: Grid ``dx``, held fixed across the levels.
        courant_numbers: Courant numbers, coarsest first. Three at least,
            since Celik's apparent order needs three levels.
        transits: Window length in elastic transits of the column.
        width_m: Column width.
        height_m: Column height.
        gravity_m_s2: Gravitational acceleration.

    Returns:
        The study.

    Raises:
        SolverInputError: If fewer than three steps are supplied, if the
            window is not positive, or if any level yields -- which would
            make the difference between levels plastic dissipation rather
            than temporal truncation error.
    """
    numbers = [float(number) for number in courant_numbers]
    if len(numbers) < 3:
        raise SolverInputError(
            f"a temporal GCI needs at least three steps, got {len(numbers)}"
        )
    if not math.isfinite(transits) or transits <= 0.0:
        raise SolverInputError(f"transits must be positive, got {transits!r}")
    size = float(cell_size_m)
    duration_s = float(transits) * height_m / material.elastic_wave_speed_m_s
    grid_shape = (
        int(math.ceil(width_m / size)) + 3,
        int(math.ceil(height_m / size)) + 4,
    )
    levels: list[TemporalLevel] = []
    for number in numbers:
        particles = _particles(
            material,
            cell_size_m=size,
            width_m=width_m,
            height_m=height_m,
        )
        grid = PlaneStrainGrid((-size, -size), size, grid_shape)
        step_s = number * size / material.elastic_wave_speed_m_s
        n_steps = max(int(round(duration_s / step_s)), 2)
        solver = _free_solver(
            material,
            size,
            walls=_COLUMN_DOMAIN,
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
        if run.steps[-1].n_yielded:
            raise SolverInputError(
                f"{run.steps[-1].n_yielded} particle(s) yielded at C={number:g}, "
                "so the difference between levels would be plastic dissipation "
                "rather than the temporal truncation error the study fits"
            )
        levels.append(
            TemporalLevel(
                time_step_s=step_s,
                courant_number=number,
                n_steps=n_steps,
                value=mean_vertical_stress_pa(material, particles),
                n_yielded=run.steps[-1].n_yielded,
            )
        )
    ordered = sorted(levels, key=lambda level: level.time_step_s, reverse=True)
    differences = [
        abs(ordered[index + 1].value - ordered[index].value)
        for index in range(len(ordered) - 1)
    ]
    return TemporalStudy(
        cell_size_m=size,
        duration_s=duration_s,
        transits=float(transits),
        levels=tuple(ordered),
        difference_order=observed_order_from_errors(
            [level.time_step_s for level in ordered[:-1]], differences
        ),
        gci=grid_convergence_index(
            [
                GridSolution(
                    cell_size_m=level.time_step_s,
                    value=level.value,
                    label=f"dt={level.time_step_s:.3e} s",
                )
                for level in ordered
            ],
            quantity=(
                "F1 elastic column mean vertical stress, temporal band only "
                f"(dx fixed at {size * 1e3:.4g} mm)"
            ),
        ),
    )
