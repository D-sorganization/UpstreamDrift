"""The plastic-limit case for F1: a closed-form limit *load* (issue #8733).

Code verification.  **No experimental data appears in this module.**

Everything in :mod:`bunkershot3d.solvers.mpm.verification` is elastic, is
kinematic, or is a restatement of the yield function's own identity.
Issue #8733 section 4 records the gap that leaves: the Drucker-Prager
return map is never checked against an *answer* -- a load a rigid-plastic
limit analysis fixes in closed form and the solver has to find.

This module is that check.  A smooth rigid wall is pushed into a
cohesionless layer until every particle is at yield, and the reaction the
solver computes for itself -- the same traction integration a club head
is loaded through -- is compared with the Rankine passive thrust.

The closed form is the *model's own*
------------------------------------

F1's yield surface is written on the **two in-plane** principal Kirchhoff
stresses, so the plane-strain Coulomb criterion it enforces sits at an
equivalent friction angle ``phi* = asin(sqrt(2) alpha)``, which for every
sand preset in this package is 31.94 deg rather than the 34 deg passed to
:func:`~bunkershot3d.solvers.mpm.constitutive.drucker_prager_alpha`.  The
derivation is in :class:`RankineLimits`, and getting it wrong would show
up as a 4% "error" that is in fact a definition.

References
----------

* Rankine, "On the stability of loose earth", *Phil. Trans. R. Soc.
  Lond.* **147**:9-27 (1857) -- the limit state.
* Drucker & Prager, *Q. Appl. Math.* **10**:157-165 (1952) -- the cone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..elements import SurfaceElements
from ..envelope import GRAVITY_M_S2, RefusalPolicy
from ..exceptions import SolverInputError
from ..protocol import IntrusionState
from .constitutive import PLANE_STRAIN_DIMENSION, SandContinuum
from .grid import PlaneStrainGrid
from .solver import MPMRun, PlaneStrainMPMSolver
from .state import DomainWalls, ParticleState, WallCondition, settled_bed

if TYPE_CHECKING:  # pragma: no cover - only a checker needs the section type
    from .body import RigidSection

__all__ = [
    "PassiveWallLimit",
    "RankineLimits",
    "passive_earth_pressure_limit",
    "rankine_limits",
]


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

    The closed form and what it assumes
    -----------------------------------

    ``P_p = K_p rho g H^2 / 2 + (s T / (1 - s)) H`` per unit width is
    exact for this constitutive model under five conditions, each
    arranged by :func:`passive_earth_pressure_limit` and reported here
    rather than assumed:

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
    4. **Fully mobilised.**  Reported as :attr:`yielded_fraction`; a load
       read off a bed that is still elastic is a stiffness, not a limit.
    5. **No escape route.**  The plate runs past the toe and past the
       free surface.

    Passive rather than active, and why
    -----------------------------------

    The active limit needs the sand to *follow* a retreating wall, so
    what is measured is the wedge's ability to fall rather than its
    strength.  Measured: the active plateau reached 0.78 of the closed
    form on a frictionless base and 0.32 on a sticky one, while the
    passive case on the same grid and material reached 1.06.  The active
    number is available from
    :meth:`RankineLimits.active_thrust_n_per_m` but is not a verified one.

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


def _push_the_wall(
    material: SandContinuum,
    *,
    cell_size_m: float,
    height_m: float,
    length_m: float,
    wall_speed_m_s: float,
    travel_m: float,
    gravity_m_s2: float,
) -> tuple[ParticleState, MPMRun]:
    """Build the layer and the wall, and march the wall into it.

    The grid is padded on the left so the wall has somewhere to start,
    and its lower and right wall bands land on the layer's own floor and
    far edge.  That is the same trap the consolidation column records: a
    generously padded grid puts the "fixed base" two cells below the sand
    and the run then looks entirely plausible while reporting zero.

    Args:
        material: The continuum.
        cell_size_m: Grid ``dx``.
        height_m: Retained height ``H``.
        length_m: Layer length.
        wall_speed_m_s: Wall speed.
        travel_m: How far the wall travels.
        gravity_m_s2: Gravitational acceleration.

    Returns:
        ``(particles, run)`` at the end of the push.
    """
    solver = PlaneStrainMPMSolver(
        material=material,
        cell_size_m=cell_size_m,
        effective_width_m=1.0,
        bed_depth_m=height_m,
        gravity_m_s2=gravity_m_s2,
        contact_friction=0.0,
        walls=_SLIP_LAYER_DOMAIN,
        refusal_policy=RefusalPolicy.REPORT,
        max_steps=_PASSIVE_STEP_CAP,
    )
    particles = settled_bed(
        material,
        x_bounds_m=(0.0, length_m),
        free_surface_height_m=0.0,
        depth_m=height_m,
        cell_size_m=cell_size_m,
        particles_per_cell_axis=2,
        gravity_m_s2=gravity_m_s2,
        geostatic=True,
    )
    pad = _PASSIVE_PAD_CELLS
    grid = PlaneStrainGrid(
        (-(pad + 1) * cell_size_m, -height_m - cell_size_m),
        cell_size_m,
        (
            int(round(length_m / cell_size_m)) + pad + 3,
            int(round(height_m / cell_size_m)) + 4,
        ),
    )
    section = _rankine_wall_section(
        solver,
        height_m=height_m,
        cell_size_m=cell_size_m,
        speed_m_s=wall_speed_m_s,
    )
    step_s = 0.4 * cell_size_m / material.elastic_wave_speed_m_s
    n_steps = max(int(math.ceil(travel_m / (wall_speed_m_s * step_s))), 2)
    run = solver.march(
        particles,
        section,
        grid,
        n_steps=n_steps,
        time_step_s=step_s,
        free_surface_height_m=0.0,
        bed_x_bounds_m=(0.0, length_m),
    )
    return particles, run


def _require_static_limit_conditions(
    limits: RankineLimits,
    *,
    quasi_static_ratio: float,
    length_ratio: float,
    plateau_fraction: float,
) -> None:
    """Refuse a configuration whose answer would not be a static limit load.

    Args:
        limits: The closed-form limits, for the wedge extent in the
            message.
        quasi_static_ratio: ``v_wall / c``.
        length_ratio: Layer length in wall heights.
        plateau_fraction: Trailing share of the march to average.

    Raises:
        SolverInputError: If the wall is not quasi-static, if the layer
            is too short to hold the passive wedge, or if the plateau
            window is not a usable fraction. Each of those turns a limit
            load into something else while still returning a number.
    """
    if not 0.0 < plateau_fraction <= 1.0:
        raise SolverInputError(
            f"plateau_fraction must lie in (0, 1], got {plateau_fraction!r}"
        )
    if not math.isfinite(quasi_static_ratio) or (
        quasi_static_ratio > _MAX_QUASI_STATIC_RATIO
    ):
        raise SolverInputError(
            f"the wall runs at v/c = {quasi_static_ratio:.3g}, over the "
            f"{_MAX_QUASI_STATIC_RATIO:g} this case treats as quasi-static. "
            "Rankine is a *static* limit; at a finite fraction of the wave "
            "speed the measured thrust carries an inertial term the closed "
            "form does not have, and the comparison would be against the "
            "wrong problem"
        )
    if length_ratio < _MIN_PASSIVE_LENGTH_RATIO:
        wedge = math.tan(math.radians(45.0 + limits.phi_star_deg / 2.0))
        raise SolverInputError(
            f"a bed {length_ratio!r} wall-heights long cannot hold the passive "
            f"wedge, which reaches H tan(45 + phi*/2) = {wedge:.3g} H; use at "
            f"least {_MIN_PASSIVE_LENGTH_RATIO:g}"
        )


def _plateau_thrust_n_per_m(run: MPMRun, *, fraction: float) -> tuple[float, float]:
    """Mean and relative spread of the wall reaction over the plateau.

    The plateau is a plastic flow rather than a settled static answer, so
    it fluctuates.  Reporting the fluctuation beside the mean keeps a
    single averaged number from reading as a converged one.

    Args:
        run: The march.
        fraction: Trailing share of the march to average over.

    Returns:
        ``(mean, standard deviation / mean)`` in N/m and as a fraction.
    """
    reaction = np.abs(np.array([step.contact_force_n_per_m[0] for step in run.steps]))
    window = max(int(round(fraction * reaction.size)), 1)
    plateau = reaction[-window:]
    mean = float(plateau.mean())
    return mean, float(plateau.std()) / max(mean, _TINY)


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

    The plastic-limit case issue #8733 section 4 asks for.  It drives the
    whole layer to the Drucker-Prager limit and compares the wall
    reaction **the solver computes for itself** --
    ``StepDiagnostics.contact_force_n_per_m``, the same traction
    integration a club head is loaded through -- against
    ``P_p = K_p rho g H^2 / 2 + (s T / (1 - s)) H`` per unit width, whose
    coefficient and intercept come from :func:`rankine_limits`.
    :class:`PassiveWallLimit` states the five conditions that make that
    exact here, and why the case is passive rather than active.

    Args:
        material: The continuum. A cohesionless one puts all of the load
            on the friction cone; ``cohesive_share`` reports the split.
        cell_size_m: Grid ``dx``.
        height_m: Retained height ``H``.
        length_ratio: Bed length in multiples of ``H``.
        wall_speed_m_s: Wall speed.
        travel_ratio: Wall travel in multiples of ``H``.
        plateau_fraction: Trailing share of the march to average.
        gravity_m_s2: Gravitational acceleration.

    Returns:
        The measured limit load beside its closed form.

    Raises:
        SolverInputError: If the configuration cannot produce a static
            limit load (see :func:`_require_static_limit_conditions`), or
            if the bed never reached its limit -- either of which would
            return a number that is not one.
    """
    limits = rankine_limits(material)
    size = float(cell_size_m)
    height = float(height_m)
    length = float(length_ratio) * height
    speed = float(wall_speed_m_s)
    ratio = speed / material.elastic_wave_speed_m_s
    _require_static_limit_conditions(
        limits,
        quasi_static_ratio=ratio,
        length_ratio=float(length_ratio),
        plateau_fraction=float(plateau_fraction),
    )

    travel = float(travel_ratio) * height
    particles, run = _push_the_wall(
        material,
        cell_size_m=size,
        height_m=height,
        length_m=length,
        wall_speed_m_s=speed,
        travel_m=travel,
        gravity_m_s2=gravity_m_s2,
    )

    thrust, spread = _plateau_thrust_n_per_m(run, fraction=float(plateau_fraction))
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
        thrust_n_per_m=thrust,
        plateau_spread=spread,
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
