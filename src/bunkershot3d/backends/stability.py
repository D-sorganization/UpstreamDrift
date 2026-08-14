"""Timestep-stability and contact-resolution criteria for the DEM backends.

Issue #8612 (findings B12, B13, B30). Before this module there was no
stability criterion anywhere in ``bunkershot3d``: Chrono used
``dt = 1 / output_rate_hz`` as its *integrator* step (~11 900x the Rayleigh
limit for 0.4 mm quartz), LIGGGHTS hard-coded ``dt = 1e-5`` in two places, and
``docs/bunkershot3d/comparison.md`` advertised a 0.2-Rayleigh safety factor
that no backend implemented.

Three independent limits are enforced here:

Rayleigh
    ``t_R = pi R sqrt(rho / G) / (0.1631 nu + 0.8766)`` is the transit time of a
    Rayleigh surface wave across a grain. A soft-sphere DEM integrator must
    resolve it; the accepted practice is 0.1-0.2 t_R.

Courant / CFL
    A body must not traverse a grain within one step. ``dt <= C d / v`` with
    ``C ~ 0.1``. At 25 m/s a 1 ms step moves the clubhead 25 mm — 62 diameters
    of 0.4 mm sand — so contacts are simply never detected.

Hertzian overlap
    The peak overlap of two equal spheres in a binary impact,
    ``delta_max / d``, is *independent of grain size* (see
    :func:`hertz_overlap_ratio`), so coarse-graining cannot rescue an over-soft
    stiffness. At 25 m/s, ``E = 1e7 Pa`` — the soft-DEM folklore value the
    canonical config used to carry — gives 47 % interpenetration.

All three refusals ``raise`` rather than ``assert``: ``python -O`` strips
asserts, and a simulation that silently runs 10 000x over its stability limit
is exactly the failure this module exists to prevent.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

from src.shared.python.core.contracts import ensure, require

from ..exceptions import BunkerShot3DStateError, BunkerShot3DValueError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import BunkerShotConfig

#: Tour-professional clubhead speed at impact (m/s). ADR-0032.
REFERENCE_IMPACT_SPEED_MPS = 25.0

#: Fraction of the Rayleigh time used as the integration step.
RAYLEIGH_SAFETY_FACTOR = 0.2

#: Fraction of a grain diameter a body may traverse in one step.
COURANT_NUMBER = 0.1

#: Largest admissible peak Hertzian overlap, as a fraction of grain diameter.
MAX_HERTZ_OVERLAP_RATIO = 0.02

#: Default ceiling on integration steps for a single shot.
DEFAULT_MAX_STEPS = 200_000

#: Grains are placed +/- 3 sigma in log-space; the smallest grain governs t_R.
SIGMA_SPAN = 3.0


class TimestepStabilityError(BunkerShot3DValueError):
    """Raised when an integration timestep exceeds a stability limit."""


class ContactStiffnessError(BunkerShot3DValueError):
    """Raised when a contact stiffness cannot resolve the configured impact."""


class StepBudgetExceededError(BunkerShot3DStateError):
    """Raised when a stable run would need more steps than the caller allows."""


class StepPlan(NamedTuple):
    """A resolved integration schedule.

    Attributes:
        dt: Integration timestep (s). *Not* the output sampling interval.
        n_steps: Number of integration steps covering the requested duration.
        output_every: Integration steps between successive output samples.
        rayleigh_limit: The 0.2-Rayleigh limit used (s), or ``inf`` if unused.
        cfl_limit: The Courant limit used (s), or ``inf`` if unused.
    """

    dt: float
    n_steps: int
    output_every: int
    rayleigh_limit: float
    cfl_limit: float


def shear_modulus(youngs_modulus: float, poisson_ratio: float) -> float:
    """Isotropic shear modulus ``G = E / (2 (1 + nu))`` in Pa."""
    if youngs_modulus <= 0.0:
        raise ValueError(f"youngs_modulus must be positive, got {youngs_modulus}")
    if not -1.0 < poisson_ratio < 0.5:
        raise ValueError(f"poisson_ratio must be in (-1, 0.5), got {poisson_ratio}")
    return youngs_modulus / (2.0 * (1.0 + poisson_ratio))


def rayleigh_time(
    *,
    radius: float,
    density: float,
    youngs_modulus: float,
    poisson_ratio: float,
) -> float:
    """Rayleigh surface-wave transit time for one grain, in seconds.

    ``t_R = pi R sqrt(rho / G) / (0.1631 nu + 0.8766)``.

    Raises:
        ValueError: Any argument is non-positive (or ``nu`` out of range).
    """
    if radius <= 0.0:
        raise ValueError(f"radius must be positive, got {radius}")
    if density <= 0.0:
        raise ValueError(f"density must be positive, got {density}")
    modulus = shear_modulus(youngs_modulus, poisson_ratio)
    return (
        math.pi
        * radius
        * math.sqrt(density / modulus)
        / (0.1631 * poisson_ratio + 0.8766)
    )


def rayleigh_timestep(
    *,
    radius: float,
    density: float,
    youngs_modulus: float,
    poisson_ratio: float,
    safety_factor: float = RAYLEIGH_SAFETY_FACTOR,
) -> float:
    """The Rayleigh-limited integration timestep, ``safety_factor * t_R``."""
    if not 0.0 < safety_factor <= 1.0:
        raise ValueError(f"safety_factor must be in (0, 1], got {safety_factor}")
    return safety_factor * rayleigh_time(
        radius=radius,
        density=density,
        youngs_modulus=youngs_modulus,
        poisson_ratio=poisson_ratio,
    )


def cfl_timestep(
    *, diameter: float, max_speed: float, courant: float = COURANT_NUMBER
) -> float:
    """Largest step for which a body traverses at most ``courant`` diameters.

    Returns ``inf`` for a stationary problem, which leaves the Rayleigh limit
    as the only active constraint.
    """
    if diameter <= 0.0:
        raise ValueError(f"diameter must be positive, got {diameter}")
    if not 0.0 < courant <= 1.0:
        raise ValueError(f"courant must be in (0, 1], got {courant}")
    if max_speed <= 0.0:
        return math.inf
    return courant * diameter / max_speed


def hertz_overlap_ratio(
    *,
    impact_speed: float,
    density: float,
    youngs_modulus: float,
    poisson_ratio: float,
) -> float:
    """Peak Hertzian overlap of two equal grains, as a fraction of diameter.

    Energy balance for a binary impact of two identical spheres::

        (1/2) m* v^2 = (8/15) E* sqrt(R*) delta_max^(5/2)

    with ``m* = m/2``, ``R* = R/2`` and ``E* = E / (2 (1 - nu^2))``. Since
    ``m* / sqrt(R*)`` scales as ``R^(5/2)``, ``delta_max`` scales as ``R``:
    the *ratio* is scale-invariant. Coarse-graining therefore cannot reduce
    interpenetration — only a stiffer contact can.

    At 25 m/s in quartz sand this reproduces 47 % at 1e7 Pa, 19 % at 1e8 Pa,
    3.0 % at 1e10 Pa and 1.4 % at 7e10 Pa.
    """
    if impact_speed < 0.0:
        raise ValueError(f"impact_speed must be non-negative, got {impact_speed}")
    if density <= 0.0:
        raise ValueError(f"density must be positive, got {density}")
    if youngs_modulus <= 0.0:
        raise ValueError(f"youngs_modulus must be positive, got {youngs_modulus}")
    if impact_speed == 0.0:
        return 0.0
    e_star = youngs_modulus / (2.0 * (1.0 - poisson_ratio**2))
    radius = 1.0  # scale-invariant
    m_star = 0.5 * density * (4.0 / 3.0) * math.pi * radius**3
    r_star = radius / 2.0
    delta_max = (
        (15.0 / 16.0) * m_star * impact_speed**2 / (e_star * math.sqrt(r_star))
    ) ** 0.4
    return delta_max / (2.0 * radius)


def require_resolvable_contacts(
    *,
    impact_speed: float,
    density: float,
    youngs_modulus: float,
    poisson_ratio: float,
    max_overlap_ratio: float = MAX_HERTZ_OVERLAP_RATIO,
) -> float:
    """Refuse a contact stiffness that cannot resolve the configured impact.

    Returns:
        The peak overlap ratio, so callers can record it.

    Raises:
        ContactStiffnessError: ``delta_max / d`` exceeds ``max_overlap_ratio``.
    """
    ratio = hertz_overlap_ratio(
        impact_speed=impact_speed,
        density=density,
        youngs_modulus=youngs_modulus,
        poisson_ratio=poisson_ratio,
    )
    if ratio > max_overlap_ratio:
        raise ContactStiffnessError(
            f"contact stiffness E={youngs_modulus:.3g} Pa gives a peak Hertzian "
            f"overlap of {ratio:.1%} of a grain diameter at "
            f"{impact_speed:.3g} m/s, above the {max_overlap_ratio:.1%} limit. "
            "The overlap ratio is independent of grain size, so coarse-graining "
            "cannot fix it; raise youngs_modulus towards the quartz value "
            "(7e10 Pa) instead."
        )
    return ratio


def require_stable_timestep(
    dt: float,
    *,
    radius: float,
    density: float,
    youngs_modulus: float,
    poisson_ratio: float,
    max_speed: float,
    rayleigh_safety: float = RAYLEIGH_SAFETY_FACTOR,
    courant: float = COURANT_NUMBER,
    enforce_rayleigh: bool = True,
) -> None:
    """Refuse an integration timestep that exceeds a stability limit.

    Args:
        dt: Proposed integration timestep (s).
        enforce_rayleigh: Set ``False`` for solvers whose contacts are not
            soft-sphere Hertzian (MuJoCo resolves contacts implicitly at the
            velocity level, so the Rayleigh wave-speed limit does not govern;
            the Courant traversal limit still does).

    Raises:
        TimestepStabilityError: ``dt`` exceeds the Rayleigh or Courant limit.
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")

    courant_limit = cfl_timestep(
        diameter=2.0 * radius, max_speed=max_speed, courant=courant
    )
    if dt > courant_limit:
        raise TimestepStabilityError(
            f"timestep {dt:.3e} s violates the CFL/Courant limit "
            f"{courant_limit:.3e} s: at {max_speed:.3g} m/s the body traverses "
            f"{dt * max_speed / (2.0 * radius):.1f} grain diameters per step, "
            "so contacts are never detected."
        )

    if not enforce_rayleigh:
        return

    limit = rayleigh_timestep(
        radius=radius,
        density=density,
        youngs_modulus=youngs_modulus,
        poisson_ratio=poisson_ratio,
        safety_factor=rayleigh_safety,
    )
    if dt > limit:
        raise TimestepStabilityError(
            f"timestep {dt:.3e} s is {dt / limit:.0f}x the Rayleigh stability "
            f"limit {limit:.3e} s ({rayleigh_safety:g} t_R for a "
            f"{radius * 1e3:.3g} mm-radius grain). Refusing to integrate: the "
            "output sampling rate is not the integration timestep."
        )


def plan_steps(
    *,
    duration: float,
    dt: float,
    output_rate_hz: float,
    max_steps: int = DEFAULT_MAX_STEPS,
    extra_steps: int = 0,
) -> StepPlan:
    """Resolve a step schedule, separating integration from output sampling.

    Args:
        duration: Simulated time to cover (s).
        dt: Integration timestep (s), already checked for stability.
        output_rate_hz: Requested sample rate for the result file.
        max_steps: Ceiling on total integration steps (including
            ``extra_steps``) before the run is refused as intractable.
        extra_steps: Steps consumed outside the main loop, e.g. relaxation.

    Raises:
        StepBudgetExceededError: The stable schedule exceeds ``max_steps``.
    """
    require(duration > 0.0, "duration must be positive", duration)
    require(dt > 0.0, "dt must be positive", dt)
    require(output_rate_hz > 0.0, "output_rate_hz must be positive", output_rate_hz)
    if duration <= 0.0 or dt <= 0.0 or output_rate_hz <= 0.0:
        raise BunkerShot3DValueError(
            "duration, dt and output_rate_hz must all be positive"
        )

    n_steps = max(1, int(round(duration / dt)))
    output_every = max(1, int(round(1.0 / (output_rate_hz * dt))))
    ensure(output_every >= 1, "output_every must be at least one step", output_every)
    total = n_steps + max(0, extra_steps)
    if total > max_steps:
        raise StepBudgetExceededError(
            f"a stable run needs {total} integration steps "
            f"({duration:.3g} s at dt={dt:.3e} s) but the budget is "
            f"{max_steps} steps. This configuration is intractable for this "
            "backend at true grain scale (ADR-0032); reduce the duration, "
            "coarse-grain the grains, or raise max_steps deliberately."
        )
    return StepPlan(
        dt=dt,
        n_steps=n_steps,
        output_every=output_every,
        rayleigh_limit=math.inf,
        cfl_limit=math.inf,
    )


def smallest_grain_radius(config: BunkerShotConfig) -> float:
    """Smallest grain radius the configured population will produce (m).

    Diameters are log-normal about ``diameter_mean``; the drivers bound the
    population at +/- ``SIGMA_SPAN`` sigma in log-space, and the *smallest*
    grain sets the Rayleigh limit.
    """
    grains = config.to_grain_population()
    return grains.radius_mean_m * math.exp(-SIGMA_SPAN * grains.diameter_sigma_log)


def largest_grain_radius(config: BunkerShotConfig) -> float:
    """Largest grain radius the configured population will produce (m)."""
    grains = config.to_grain_population()
    return grains.radius_mean_m * math.exp(SIGMA_SPAN * grains.diameter_sigma_log)


def validate_contact_model(
    config: BunkerShotConfig,
    *,
    impact_speed: float = REFERENCE_IMPACT_SPEED_MPS,
) -> float:
    """Refuse a config whose contact stiffness cannot resolve the impact."""
    material = config.to_contact_material()
    return require_resolvable_contacts(
        impact_speed=impact_speed,
        density=config.to_grain_population().density_kg_m3,
        youngs_modulus=material.youngs_modulus_pa,
        poisson_ratio=material.poisson_ratio,
    )


def plan_from_config(
    config: BunkerShotConfig,
    *,
    max_speed: float,
    max_steps: int = DEFAULT_MAX_STEPS,
    extra_steps: int = 0,
    enforce_rayleigh: bool = True,
) -> StepPlan:
    """Derive a stable, budgeted step schedule for *config*.

    The integration timestep is the smaller of the Rayleigh and Courant limits
    evaluated for the *smallest* grain in the population; the output sampling
    interval comes separately from ``output.rate_hz``.

    Raises:
        ContactStiffnessError: The stiffness cannot resolve ``max_speed``.
        StepBudgetExceededError: The stable schedule is intractable.
    """
    material = config.to_contact_material()
    grains = config.to_grain_population()
    require_resolvable_contacts(
        impact_speed=max_speed,
        density=grains.density_kg_m3,
        youngs_modulus=material.youngs_modulus_pa,
        poisson_ratio=material.poisson_ratio,
    )

    radius = smallest_grain_radius(config)
    cfl_limit = cfl_timestep(diameter=2.0 * radius, max_speed=max_speed)
    if enforce_rayleigh:
        rayleigh_limit = rayleigh_timestep(
            radius=radius,
            density=grains.density_kg_m3,
            youngs_modulus=material.youngs_modulus_pa,
            poisson_ratio=material.poisson_ratio,
        )
    else:
        rayleigh_limit = math.inf

    dt = min(rayleigh_limit, cfl_limit)
    if not math.isfinite(dt):
        raise TimestepStabilityError(
            "neither the Rayleigh nor the Courant criterion is active; refusing "
            "to pick an integration timestep by guesswork"
        )
    # Stable by construction, but assert it explicitly so a future change to the
    # derivation above cannot quietly reintroduce B30.
    require_stable_timestep(
        dt,
        radius=radius,
        density=grains.density_kg_m3,
        youngs_modulus=material.youngs_modulus_pa,
        poisson_ratio=material.poisson_ratio,
        max_speed=max_speed,
        enforce_rayleigh=enforce_rayleigh,
    )

    settings = config.to_solver_settings()
    plan = plan_steps(
        duration=config.to_trajectory_source().duration_s,
        dt=dt,
        output_rate_hz=settings.output_rate_hz,
        max_steps=max_steps,
        extra_steps=extra_steps,
    )
    return plan._replace(rayleigh_limit=rayleigh_limit, cfl_limit=cfl_limit)
