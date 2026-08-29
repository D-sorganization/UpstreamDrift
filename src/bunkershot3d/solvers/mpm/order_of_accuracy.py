"""Order of accuracy for F1: MMS in space, refinement in time (issue #8733).

Code verification.  **No experimental data appears in this module.**

Two studies and the identity that keeps them honest
---------------------------------------------------

**The patch test.**  A spatially *uniform* stress field must produce no
net internal force at any fully supported node, for any stress
whatsoever.  That is an identity of the scheme, so it is **round-off
class**: a fixed tolerance, no step size, and no order test.
:func:`uniform_stress_patch_residual` measures it, and it is what fixes
the floor the manufactured study is measured against -- if the transfer's
linear completeness were broken, the manufactured error would still fall
under refinement and would still fit a plausible order, and only this
residual would notice.

**MMS.**  :func:`manufactured_solution_convergence` manufactures a smooth
elastic stress field, derives the acceleration it must produce in closed
form, and asks the solver for exactly one step from rest -- which turns
the particle velocity into ``dt`` times the discrete P2G-solve-G2P
operator.  That is a **truncation-class** error and the *order* is its
test.  It covers the stress divergence and the transfer together, on the
elastic branch, and :meth:`ManufacturedSolutionStudy.summary` states the
half it does not cover rather than leaving a reader to assume a whole.

**Time.**  :func:`column_temporal_convergence` refines the step on a
*fixed* grid, which the shipped spatial study does not: that one holds
the Courant number fixed, so its GCI is a space-time band.  The temporal
band is reported separately and labelled as such, and where the series
stops converging that is reported too, with the mechanism named.

References
----------

* Roache, *Verification and Validation in Computational Science and
  Engineering* (1998) -- MMS.
* Celik et al., *J. Fluids Eng.* **130**(7):078001 (2008) -- the GCI,
  used here through :mod:`bunkershot3d.vandv.gci` rather than reproduced.
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
from ..envelope import GRAVITY_M_S2
from ..exceptions import SolverInputError
from .constitutive import PLANE_STRAIN_DIMENSION, SandContinuum
from .grid import PlaneStrainGrid
from .solver import MPMRun
from .state import ParticleState, settled_bed

# The shipped suite already owns the two wall configurations and the two
# set-up helpers these studies need, and a second copy of either would be a
# second place for the "wall bands land on the column" mistake to be made.
# The dependency runs one way only: verification.py does not import this
# module.
from .verification import (
    _COLUMN_DOMAIN,
    _free_solver,
    _OPEN_DOMAIN,
    _particles,
    mean_vertical_stress_pa,
)

__all__ = [
    "DEFAULT_MANUFACTURED_FIELD",
    "DESIGN_ORDER_SPATIAL",
    "ManufacturedField",
    "ManufacturedLevel",
    "ManufacturedSolutionStudy",
    "TemporalLevel",
    "TemporalStudy",
    "column_temporal_convergence",
    "manufactured_solution_convergence",
    "uniform_stress_patch_residual",
]


# ------------------------------------------- method of manufactured solutions


_TINY = float(np.finfo(np.float64).tiny)
"""Smallest positive normal double, used as a strictly-positive floor."""

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
