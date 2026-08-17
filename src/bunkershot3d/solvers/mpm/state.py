"""Particle state, bed initialisation and the domain walls.

The bed starts in its **geostatic** elastic state rather than unstressed.
An unstressed bed dropped under gravity rings for several elastic transit
times before it settles, and that ringing would be indistinguishable from
the club's own signal in a 2 ms engagement window.  Initialising at the
1-D consolidation solution

    ``sigma_zz = -rho g d``,  ``eps_xx = 0``,  ``eps_zz = sigma_zz / (lambda + 2 mu)``

removes the transient by construction, up to the discretisation error of
the stress divergence.

This is *not* the same statement as "the analytic verification case is
satisfied by construction".  The verification case in
:mod:`bunkershot3d.solvers.mpm.verification` starts from an **unstressed**
column and relaxes it, and then asks whether the solver found this state
on its own.  Seeding it here and finding it there are different claims,
and only the second one is evidence.

Admissibility of the seed is checked rather than assumed: a sand with a
low enough friction angle would yield under its own weight at ``K_0``
confinement, and seeding an inadmissible state would leave the first step
doing a large plastic correction that looks like club load.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from ..envelope import GRAVITY_M_S2
from ..exceptions import SolverInputError
from .constitutive import SandContinuum
from .grid import PlaneStrainGrid

__all__ = [
    "DomainWalls",
    "ParticleState",
    "SurfaceDepression",
    "WallCondition",
    "apply_wall_conditions",
    "settled_bed",
    "surface_depression",
    "surface_profile_m",
]

_DIMENSION = 2


class WallCondition(StrEnum):
    """What a domain wall does to the nodal velocity beside it."""

    STICKY = "sticky"
    """Both components zeroed. The bed floor."""

    SLIP = "slip"
    """Only the wall-normal component zeroed, in both directions.

    The lateral condition of a 1-D consolidation column: it enforces
    ``eps_xx = 0``, which is exactly the confinement the geostatic seed
    assumes."""

    SEPARATE = "separate"
    """The wall-normal component zeroed only when it points outward.

    Material may leave but not enter. The condition a free lateral
    boundary wants."""

    FREE = "free"
    """Nothing is done. The open top of the domain."""


@dataclass(frozen=True, slots=True)
class DomainWalls:
    """The four walls of the plane-strain domain.

    Attributes:
        lower_x: Condition on the ``-x`` face.
        upper_x: Condition on the ``+x`` face.
        lower_z: Condition on the ``-z`` face, the bed floor.
        upper_z: Condition on the ``+z`` face, normally open.
        thickness_cells: How many node layers each wall acts over. Two,
            by default, because a quadratic stencil is three nodes wide
            and a one-node wall lets a particle's stencil reach past it.
    """

    lower_x: WallCondition = WallCondition.SLIP
    upper_x: WallCondition = WallCondition.SLIP
    lower_z: WallCondition = WallCondition.STICKY
    upper_z: WallCondition = WallCondition.FREE
    thickness_cells: int = 2

    def __post_init__(self) -> None:
        if int(self.thickness_cells) < 1:
            raise SolverInputError(
                f"thickness_cells must be at least 1, got {self.thickness_cells!r}"
            )


def apply_wall_conditions(
    grid: PlaneStrainGrid,
    node_velocity_m_s: NDArray[np.float64],
    walls: DomainWalls,
) -> NDArray[np.float64]:
    """Impose the wall conditions on a nodal velocity field, in place.

    Args:
        grid: The background grid.
        node_velocity_m_s: ``(n_nodes, 2)`` nodal velocities, mutated.
        walls: The four conditions.

    Returns:
        The same array, for chaining.
    """
    count_x, count_z = grid.node_counts
    band = int(walls.thickness_cells)
    velocity = node_velocity_m_s.reshape(count_x, count_z, _DIMENSION)

    def _impose(
        view: NDArray[np.float64], condition: WallCondition, axis: int, outward: float
    ) -> None:
        if condition is WallCondition.FREE:
            return
        if condition is WallCondition.STICKY:
            view[...] = 0.0
            return
        component = view[..., axis]
        if condition is WallCondition.SLIP:
            component[...] = 0.0
            return
        # SEPARATE: stop only what is heading out of the domain.
        component[...] = np.where(component * outward > 0.0, 0.0, component)

    _impose(velocity[:band, :, :], walls.lower_x, 0, -1.0)
    _impose(velocity[count_x - band :, :, :], walls.upper_x, 0, +1.0)
    _impose(velocity[:, :band, :], walls.lower_z, 1, -1.0)
    _impose(velocity[:, count_z - band :, :], walls.upper_z, 1, +1.0)
    return node_velocity_m_s


@dataclass
class ParticleState:
    """The Lagrangian carriers of mass, momentum and deformation.

    Deliberately mutable: a step advances every array in place and the
    solver holds one instance for a whole run.  Everything is per unit
    out-of-plane width, so masses are kg/m and volumes m^2.

    Attributes:
        position_m: ``(n, 2)`` positions.
        velocity_m_s: ``(n, 2)`` velocities.
        affine: ``(n, 2, 2)`` APIC affine velocity matrices ``C_p``.
        deformation_gradient: ``(n, 2, 2)`` **elastic** deformation
            gradients; the plastic part is discarded by the return map.
        mass_kg: ``(n,)`` particle masses per unit width.
        initial_volume_m2: ``(n,)`` reference areas.
    """

    position_m: NDArray[np.float64]
    velocity_m_s: NDArray[np.float64]
    affine: NDArray[np.float64]
    deformation_gradient: NDArray[np.float64]
    mass_kg: NDArray[np.float64]
    initial_volume_m2: NDArray[np.float64]

    def __post_init__(self) -> None:
        count = self.position_m.shape[0]
        shapes = {
            "position_m": (count, _DIMENSION),
            "velocity_m_s": (count, _DIMENSION),
            "affine": (count, _DIMENSION, _DIMENSION),
            "deformation_gradient": (count, _DIMENSION, _DIMENSION),
            "mass_kg": (count,),
            "initial_volume_m2": (count,),
        }
        for name, expected in shapes.items():
            actual = getattr(self, name).shape
            if actual != expected:
                raise SolverInputError(
                    f"{name} must have shape {expected}, got {actual}"
                )
        if count == 0:
            raise SolverInputError(
                "a bed with no particles cannot be solved; an empty domain would "
                "return an identically zero wrench that reads as a result"
            )
        if np.any(self.mass_kg <= 0.0):
            raise SolverInputError("every particle must carry positive mass")

    @property
    def n_particles(self) -> int:
        """Number of particles."""
        return int(self.position_m.shape[0])

    @property
    def total_mass_kg(self) -> float:
        """Total bed mass per unit width. Invariant for the whole run."""
        return float(self.mass_kg.sum())

    def linear_momentum_kg_m_s(self) -> NDArray[np.float64]:
        """``(2,)`` total linear momentum per unit width."""
        return (self.mass_kg[:, None] * self.velocity_m_s).sum(axis=0)

    def kinetic_energy_j(self) -> float:
        """Translational kinetic energy per unit width.

        The APIC affine field carries kinetic energy too; it is excluded
        here and the exclusion is stated, because the energy residual it
        feeds is a truncation-class quantity whose *order* is the test,
        and a term that is itself O(dx^2) would blur that order.
        """
        speed_squared = np.einsum("ij,ij->i", self.velocity_m_s, self.velocity_m_s)
        return 0.5 * float((self.mass_kg * speed_squared).sum())

    def gravitational_energy_j(self, gravity_m_s2: float, datum_m: float) -> float:
        """Gravitational potential energy per unit width above ``datum_m``."""
        return float(
            (self.mass_kg * gravity_m_s2 * (self.position_m[:, 1] - datum_m)).sum()
        )

    def copy(self) -> ParticleState:
        """A deep copy, so a run can be restarted from a saved state."""
        return ParticleState(
            position_m=self.position_m.copy(),
            velocity_m_s=self.velocity_m_s.copy(),
            affine=self.affine.copy(),
            deformation_gradient=self.deformation_gradient.copy(),
            mass_kg=self.mass_kg.copy(),
            initial_volume_m2=self.initial_volume_m2.copy(),
        )


def settled_bed(
    material: SandContinuum,
    *,
    x_bounds_m: tuple[float, float],
    free_surface_height_m: float,
    depth_m: float,
    cell_size_m: float,
    particles_per_cell_axis: int = 2,
    gravity_m_s2: float = GRAVITY_M_S2,
    geostatic: bool = True,
) -> ParticleState:
    """Fill a rectangular bed with particles in geostatic equilibrium.

    Particles are placed on a regular sub-cell lattice, which is the
    standard MPM quadrature: ``particles_per_cell_axis ** 2`` per cell,
    each carrying ``dx^2 / ppc`` of reference area.  Two per axis (four
    per cell) is the usual choice and is enough for the quadratic basis.

    Args:
        material: The continuum, for density and stiffness.
        x_bounds_m: ``(lower, upper)`` horizontal extent of the bed.
        free_surface_height_m: World ``z`` of the undisturbed surface.
        depth_m: Bed depth below that surface.
        cell_size_m: Grid ``dx``; the particle lattice is derived from it.
        particles_per_cell_axis: Particles per cell per axis.
        gravity_m_s2: Gravitational acceleration.
        geostatic: Seed the 1-D consolidation stress. Setting this False
            gives an unstressed bed, which is what the analytic
            verification case needs so that finding the geostatic state
            is a result rather than an input.

    Returns:
        The bed.

    Raises:
        SolverInputError: If the bed is degenerate, or if the geostatic
            seed would be outside the yield surface -- which would make
            the first step a large plastic correction indistinguishable
            from club load.
    """
    lower_x, upper_x = (float(value) for value in x_bounds_m)
    depth = float(depth_m)
    size = float(cell_size_m)
    per_axis = int(particles_per_cell_axis)
    if not math.isfinite(lower_x) or not math.isfinite(upper_x) or upper_x <= lower_x:
        raise SolverInputError(
            f"x_bounds_m must be an increasing finite pair, got {x_bounds_m!r}"
        )
    if not math.isfinite(depth) or depth <= 0.0:
        raise SolverInputError(f"depth_m must be positive, got {depth_m!r}")
    if not math.isfinite(size) or size <= 0.0:
        raise SolverInputError(f"cell_size_m must be positive, got {cell_size_m!r}")
    if per_axis < 1:
        raise SolverInputError(
            f"particles_per_cell_axis must be at least 1, got "
            f"{particles_per_cell_axis!r}"
        )

    spacing = size / per_axis
    count_x = max(int(round((upper_x - lower_x) / spacing)), 1)
    count_z = max(int(round(depth / spacing)), 1)
    offsets_x = (np.arange(count_x) + 0.5) * spacing + lower_x
    offsets_z = (np.arange(count_z) + 0.5) * spacing + (
        float(free_surface_height_m) - depth
    )
    mesh_x, mesh_z = np.meshgrid(offsets_x, offsets_z, indexing="ij")
    position = np.stack([mesh_x.ravel(), mesh_z.ravel()], axis=1)

    volume = np.full(position.shape[0], spacing * spacing)
    mass = material.density_kg_m3 * volume

    gradient = np.tile(np.eye(_DIMENSION), (position.shape[0], 1, 1))
    if geostatic:
        below = np.maximum(float(free_surface_height_m) - position[:, 1], 0.0)
        vertical_stress = -material.density_kg_m3 * gravity_m_s2 * below
        vertical_strain = vertical_stress / material.p_wave_modulus_pa
        _require_admissible_seed(material, vertical_strain)
        gradient[:, 1, 1] = np.exp(vertical_strain)

    return ParticleState(
        position_m=position,
        velocity_m_s=np.zeros_like(position),
        affine=np.zeros((position.shape[0], _DIMENSION, _DIMENSION)),
        deformation_gradient=gradient,
        mass_kg=mass,
        initial_volume_m2=volume,
    )


def _require_admissible_seed(
    material: SandContinuum, vertical_strain: NDArray[np.float64]
) -> None:
    """Refuse a geostatic seed that is already yielding."""
    strain = np.stack([np.zeros_like(vertical_strain), vertical_strain], axis=1)
    worst = float(material.yield_value(strain).max())
    if worst > 0.0:
        raise SolverInputError(
            f"the geostatic seed is outside the yield surface (y = {worst:.4g} Pa) "
            f"for a friction angle of {material.friction_angle_deg:.3g} deg: this "
            "bed cannot stand up under its own weight at K_0 confinement, so "
            "seeding it would make the first step a large plastic correction that "
            "is indistinguishable from club load"
        )


def surface_profile_m(
    particles: ParticleState,
    *,
    x_bounds_m: tuple[float, float],
    n_bins: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Free-surface height per horizontal bin, from the topmost particle.

    The free surface in MPM is wherever the particles stop, so it is read
    off the particles rather than tracked.  Bins with no particle are
    reported as ``nan`` rather than interpolated, because an emptied bin
    is a real feature -- it is the divot -- and filling it in would erase
    the quantity being measured.

    Args:
        particles: The bed.
        x_bounds_m: Horizontal range to profile.
        n_bins: Number of bins across that range.

    Returns:
        ``(bin_centres, surface_height)``, both ``(n_bins,)``.

    Raises:
        SolverInputError: If ``n_bins`` is not positive.
    """
    if int(n_bins) < 1:
        raise SolverInputError(f"n_bins must be positive, got {n_bins!r}")
    lower, upper = (float(value) for value in x_bounds_m)
    if upper <= lower:
        raise SolverInputError(
            f"x_bounds_m must be an increasing pair, got {x_bounds_m!r}"
        )
    edges = np.linspace(lower, upper, int(n_bins) + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    index = np.digitize(particles.position_m[:, 0], edges) - 1
    # -inf while reducing, because np.maximum against nan propagates the nan
    # and would swallow every bin that does hold a particle.
    heights = np.full(int(n_bins), -np.inf)
    inside = (index >= 0) & (index < int(n_bins))
    if bool(inside.any()):
        np.maximum.at(heights, index[inside], particles.position_m[inside, 1])
    return centres, np.where(np.isfinite(heights), heights, np.nan)


@dataclass(frozen=True, slots=True)
class SurfaceDepression:
    """How far the free surface fell, and how much section that removed.

    :meth:`~bunkershot3d.solvers.mpm.solver.MPMRun.divot_depth_m` answers
    the first question alone.  The cross-tier comparison of issue #8713
    needs the second as well, because F0 reports its divot as an area and
    a mass (:class:`~bunkershot3d.metrics.divot.DivotMetrics`) and a depth
    cannot be compared against either.

    The empty-bin count is why this is a value object rather than a float.
    A bin the sand has left entirely has no surface height at all, so it
    can contribute no depth to the integral: skipping it under-reports the
    area and filling it in invents one.  The count therefore travels with
    the number, and :attr:`fully_resolved` is what a caller checks before
    quoting it.

    Attributes:
        section_area_m2: ``integral of (surface - height) dx`` over the
            populated bins [m^2]. Per unit out-of-plane width, like every
            other plane-strain quantity in this package.
        max_depth_m: Deepest single bin, non-negative.
        n_bins: Bins the profile was taken on.
        n_empty_bins: Of those, bins holding no particle.
        bed_width_m: Horizontal extent the profile spans [m].
    """

    section_area_m2: float
    max_depth_m: float
    n_bins: int
    n_empty_bins: int
    bed_width_m: float

    def __post_init__(self) -> None:
        """Validate the measurement.

        Raises:
            SolverInputError: If the area or depth is negative or not
                finite, if the bin counts are inconsistent, or if the bed
                width is not positive. A ``raise`` and not an ``assert``:
                ``python -O`` strips assertions, and a negative divot area
                would otherwise reach a comparison as a number.
        """
        for name, value in (
            ("section_area_m2", self.section_area_m2),
            ("max_depth_m", self.max_depth_m),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise SolverInputError(
                    f"{name} must be finite and non-negative, got {value!r}"
                )
        if int(self.n_bins) < 1:
            raise SolverInputError(f"n_bins must be positive, got {self.n_bins!r}")
        if not 0 <= int(self.n_empty_bins) <= int(self.n_bins):
            raise SolverInputError(
                f"n_empty_bins must lie in [0, {self.n_bins}], got "
                f"{self.n_empty_bins!r}"
            )
        if not math.isfinite(self.bed_width_m) or self.bed_width_m <= 0.0:
            raise SolverInputError(
                f"bed_width_m must be positive, got {self.bed_width_m!r}"
            )

    @property
    def fully_resolved(self) -> bool:
        """Whether every bin held sand, so the area integral is complete."""
        return self.n_empty_bins == 0

    def displaced_mass_kg(self, *, width_m: float, bulk_density_kg_m3: float) -> float:
        """Mass of sand this section corresponds to at a **declared** width.

        Plane strain has no out-of-plane extent, so there is no such thing
        as a mass here until a width is declared.  It is a required keyword
        with no default for the same reason
        :attr:`~bunkershot3d.solvers.mpm.solver.PlaneStrainMPMSolver.effective_width_m`
        is: the number is conditional on an assumption, and the assumption
        has to be stated by whoever quotes it.

        Args:
            width_m: Declared out-of-plane width [m].
            bulk_density_kg_m3: Sand bulk density [kg/m^3].

        Returns:
            The displaced mass [kg].

        Raises:
            SolverInputError: If either argument is not positive.
        """
        if not math.isfinite(width_m) or width_m <= 0.0:
            raise SolverInputError(f"width_m must be positive, got {width_m!r}")
        if not math.isfinite(bulk_density_kg_m3) or bulk_density_kg_m3 <= 0.0:
            raise SolverInputError(
                f"bulk_density_kg_m3 must be positive, got {bulk_density_kg_m3!r}"
            )
        return self.section_area_m2 * float(width_m) * float(bulk_density_kg_m3)

    def summary(self) -> str:
        """A line fit for a comparison report."""
        resolved = (
            "every bin held sand"
            if self.fully_resolved
            else f"{self.n_empty_bins} of {self.n_bins} bins held no sand, so the "
            "area is a lower bound"
        )
        return (
            f"section {self.section_area_m2 * 1e4:.4g} cm^2 over "
            f"{self.bed_width_m * 1e3:.4g} mm, deepest "
            f"{self.max_depth_m * 1e3:.4g} mm ({resolved})"
        )


def surface_depression(
    particles: ParticleState,
    *,
    free_surface_height_m: float,
    x_bounds_m: tuple[float, float],
    n_bins: int,
) -> SurfaceDepression:
    """Measure the divot the sand itself carries, as a depth **and** an area.

    Read off :func:`surface_profile_m`, so the free surface is wherever the
    particles stopped rather than a tracked interface.  Heaped shoulders --
    bins standing *above* the undisturbed level -- are clipped to zero
    rather than subtracted: the quantity asked for is how much section was
    removed from below the original surface, and letting a heap cancel a
    hole would report a shallow divot for a bed that had merely been
    rearranged.

    Args:
        particles: The bed, after the march.
        free_surface_height_m: The undisturbed surface [m].
        x_bounds_m: Horizontal range to profile.
        n_bins: Bins across that range.

    Returns:
        The measurement, carrying its own empty-bin count.

    Raises:
        SolverInputError: If ``n_bins`` is not positive, if the bounds do
            not increase, or if **no** bin holds a particle -- which is not
            a zero divot but an unmeasured one, and a zero would read as
            the first.
    """
    _centres, heights = surface_profile_m(
        particles, x_bounds_m=x_bounds_m, n_bins=n_bins
    )
    populated = np.isfinite(heights)
    if not bool(populated.any()):
        raise SolverInputError(
            "no particle lies inside the profiled range, so the free surface "
            "was not measured anywhere; a zero divot would read as a measured "
            "flat bed rather than as an empty window"
        )
    depth = np.zeros_like(heights)
    depth[populated] = np.maximum(
        float(free_surface_height_m) - heights[populated], 0.0
    )
    lower, upper = (float(value) for value in x_bounds_m)
    # Midpoint, not trapezoid. The profile is a binned maximum -- piecewise
    # constant over each bin -- rather than a sampled smooth curve, so the
    # midpoint rule is exact for it, while the trapezoid rule over the bin
    # *centres* would silently drop half a bin at each end of the bed.
    bin_width_m = (upper - lower) / float(int(n_bins))

    return SurfaceDepression(
        section_area_m2=float(depth[populated].sum() * bin_width_m),
        max_depth_m=float(depth[populated].max()),
        n_bins=int(n_bins),
        n_empty_bins=int((~populated).sum()),
        bed_width_m=upper - lower,
    )
