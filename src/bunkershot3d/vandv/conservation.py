"""Conservation checks, split by class (issue #8616).

Code verification.  **No experimental data appears in this module.**

Two classes, two different tests
--------------------------------

The research digest is explicit that conservation checks fall into two
classes needing two different tests, and that running the wrong one is
worse than running none:

**Round-off class** -- mass, linear momentum, angular momentum.  These are
exact identities of the discrete scheme, so the residual is pure
floating-point noise.  They get a fixed absolute tolerance of about
``1e-12`` and **no order test**: refining the step will not shrink a
round-off residual, so an order test on one measures the floating-point
unit rather than the model.
:meth:`ConservationResidual.within_round_off` is available only on this
class, and
:func:`~bunkershot3d.vandv.convergence.observed_order_from_residuals`
refuses it.

**Truncation class** -- energy under a non-symplectic integrator.  Here
the residual *should* scale as ``dt^p``, so the order test **is** the
test.  A fixed tolerance on a truncation residual proves nothing: it can
be met by taking a smaller step, which is not the same as the scheme
being right.

Angular momentum is the test that finds these bugs
--------------------------------------------------

Momentum is conserved only if tangential forces are applied as an
equal-and-opposite pair **at the contact point**, with the matching
torque on both bodies.  The classic defect -- using the body centre
instead of the contact point, or applying a friction torque to one side
only -- **is invisible to a linear-momentum test**, because the resultant
force is unchanged.  Two residuals here look for exactly that:

* :func:`moment_transfer_residual` moves the reference point and checks
  ``tau(p2) = tau(p1) + (p1 - p2) x F``.  A torque built on the wrong
  lever arm fails this and passes every force test.
* :func:`element_moment_residual` rebuilds the torque with a naive
  per-element ``np.cross`` loop -- the deliberately slow oracle the
  digest calls for -- and compares it with the solver's hand-written
  cross product.  An axis swap or a sign flip in that hand-written form
  changes no force at all.

Both refuse to run on a configuration whose torque is too nearly
axis-aligned to detect a swap, so neither can pass vacuously.

The support is on the ledger too (issue #9544)
----------------------------------------------

A prescribed-rotation march holds the head's angular velocity fixed
while the sand applies a moment to it, so *something* -- the shaft, the
grip, the golfer -- is reacting that moment and doing work against it.
Leaving it implied is how an energy budget quietly fails to close.
:func:`support_angular_impulse` states the angular impulse the support
supplied (``-(integral of tau dt)``, about the moving body origin, world frame) and
:func:`prescribed_driver_work` the work the driver did holding the
delivered rotation (``-(integral of tau . omega dt)``).  Both are identities of
the prescribed scheme and both **refuse a coupled trace**: there the
coupling owns the head's inertia and the ledger is its to keep.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.shared.python.core.contracts import require

from ..solvers import DRFTSolver, IntrusionState, RotationMode, ShotResult
from .exceptions import ConservationClassError, VerificationError

__all__ = [
    "ROUND_OFF_TOLERANCE",
    "ConservationClass",
    "ConservationResidual",
    "PrescribedDriverWork",
    "SupportAngularImpulse",
    "element_moment_residual",
    "energy_work_residual",
    "linear_impulse_residual",
    "moment_transfer_residual",
    "prescribed_driver_work",
    "support_angular_impulse",
]

ROUND_OFF_TOLERANCE = 1e-12
"""Fixed relative tolerance for round-off-class conservation residuals."""

_MIN_COMPONENT_SHARE = 0.05
"""Smallest share of the torque magnitude every component must carry.

Below this a component is close enough to zero that swapping it with
another would not change the answer, and the test would pass whether or
not the cross product is right."""

_MIN_COMPONENT_SEPARATION = 0.01
"""Smallest relative difference between any two torque components.

Two equal components are interchangeable, so a configuration that
produces them cannot detect an axis swap either."""


class ConservationClass(StrEnum):
    """Which of the two conservation classes a residual belongs to."""

    ROUND_OFF = "round_off"
    """An exact identity of the scheme: mass, linear and angular momentum."""

    TRUNCATION = "truncation"
    """A quantity the scheme only conserves to ``O(dt^p)``: energy."""


@dataclass(frozen=True, slots=True)
class ConservationResidual:
    """One conservation residual, carrying the class that decides its test.

    Attributes:
        name: What was checked.
        conservation_class: Which test applies.
        residual: Absolute residual, in the quantity's own units.
        scale: The natural magnitude the residual is judged against.
        step_size_s: The timestep the residual was produced at. Required
            for the truncation class, since the order test needs it, and
            ``None`` for the round-off class, where it is meaningless.
    """

    name: str
    conservation_class: ConservationClass
    residual: float
    scale: float
    step_size_s: float | None = None

    def __post_init__(self) -> None:
        """Validate.

        Raises:
            ConservationClassError: If a truncation residual carries no
                step size, or a round-off residual carries one.
            VerificationError: If the scale is not usable.
        """
        if not math.isfinite(self.scale) or self.scale <= 0.0:
            raise VerificationError(
                f"residual {self.name!r} has a non-positive scale {self.scale!r}; "
                "a residual with nothing to be judged against is not a check"
            )
        truncation = self.conservation_class is ConservationClass.TRUNCATION
        if truncation and self.step_size_s is None:
            raise ConservationClassError(
                f"truncation-class residual {self.name!r} carries no step size, "
                "so the order test that *is* its test cannot be run"
            )
        if not truncation and self.step_size_s is not None:
            raise ConservationClassError(
                f"round-off-class residual {self.name!r} carries a step size. "
                "Round-off residuals do not scale with the step, so recording "
                "one invites an order test that would measure the "
                "floating-point unit rather than the model."
            )

    @property
    def relative(self) -> float:
        """Residual divided by its scale."""
        return self.residual / self.scale

    @property
    def within_round_off(self) -> bool:
        """Whether a round-off-class residual meets the fixed tolerance.

        Raises:
            ConservationClassError: If asked of a truncation-class
                residual, whose test is the order test, not a tolerance.
        """
        if self.conservation_class is not ConservationClass.ROUND_OFF:
            raise ConservationClassError(
                f"{self.name!r} is a truncation-class residual; it is judged by "
                "the observed order of its decay, not by a fixed tolerance. A "
                "truncation residual can always be made small by shrinking dt, "
                "which says nothing about the scheme."
            )
        return self.relative <= ROUND_OFF_TOLERANCE

    def summary(self) -> str:
        """A one-line statement fit for a report."""
        step = "" if self.step_size_s is None else f" at dt={self.step_size_s:.4g} s"
        return (
            f"{self.name} [{self.conservation_class.value}]: "
            f"residual {self.residual:.4g}, relative {self.relative:.3e}{step}"
        )


def _trace_step(trace: ShotResult, time_step_s: float) -> float:
    """Validate a trace and its step size.

    Raises:
        VerificationError: If the trace is too short or the step unusable.
    """
    if trace.n_steps < 2:
        raise VerificationError(
            f"a conservation check needs at least two samples, got "
            f"{trace.n_steps}; the shot never engaged the bed"
        )
    step = float(time_step_s)
    if not math.isfinite(step) or step <= 0.0:
        raise VerificationError(f"time_step_s must be positive, got {time_step_s!r}")
    return step


def linear_impulse_residual(
    trace: ShotResult,
    *,
    head_mass_kg: float,
    time_step_s: float,
    weight_n: ArrayLike | None = None,
) -> ConservationResidual:
    """``m (v_k - v_0) - dt sum_{j<k} F_j``, over every sample.

    This is an exact identity of the explicit update in
    :func:`~bunkershot3d.solvers.simulate_shot`, so it belongs to the
    round-off class: it detects a wrong mass, a wrong step, a dropped
    external force or a mis-ordered update, and it detects none of them
    approximately.

    Args:
        trace: The shot trace.
        head_mass_kg: Head mass used by the integration.
        time_step_s: The fixed step the trace was produced at.
        weight_n: External force applied alongside the sand wrench,
            ``None`` when the shot ran without gravity.

    Returns:
        The worst residual over the trace, round-off class.

    Raises:
        VerificationError: If the trace or the mass is unusable.
    """
    step = _trace_step(trace, time_step_s)
    mass = float(head_mass_kg)
    if not math.isfinite(mass) or mass <= 0.0:
        raise VerificationError(f"head_mass_kg must be positive, got {head_mass_kg!r}")
    external = (
        np.zeros(3, dtype=np.float64)
        if weight_n is None
        else np.asarray(weight_n, dtype=np.float64).reshape(3)
    )
    applied = trace.forces_n + external
    # Cumulative impulse *before* each sample: the update at step j uses the
    # force recorded at step j, so sample k has seen forces 0..k-1.
    impulse = step * np.cumsum(applied, axis=0)
    momentum = mass * (trace.velocities_m_s - trace.velocities_m_s[0])
    residual = np.abs(momentum[1:] - impulse[:-1]).max()
    scale = float(
        mass
        * np.sqrt(
            np.max(np.einsum("ij,ij->i", trace.velocities_m_s, trace.velocities_m_s))
        )  # noqa: E501 ⚡ Bolt: np.sqrt(np.max(np.einsum(...))) is ~2.7x faster than np.max(np.linalg.norm(..., axis=1))
        + step * np.abs(applied).sum()
    )
    return ConservationResidual(
        name="linear momentum (impulse identity)",
        conservation_class=ConservationClass.ROUND_OFF,
        residual=float(residual),
        scale=max(float(scale), float(np.finfo(np.float64).tiny)),
    )


def _require_detectable_torque(
    torque: NDArray[np.float64], scale: float, context: str
) -> None:
    """Refuse a torque too degenerate to expose an axis swap.

    Raises:
        VerificationError: If any component is negligible or two
            components are indistinguishable.
    """
    magnitude = float(np.linalg.norm(torque))
    require(
        magnitude > 0.0,
        f"{context}: the configuration produces no net torque at all, so the "
        "check would pass whatever the cross product did",
        value=magnitude,
    )
    shares = np.abs(torque) / magnitude
    if float(shares.min()) < _MIN_COMPONENT_SHARE:
        raise VerificationError(
            f"{context}: torque {torque!r} has a component carrying only "
            f"{shares.min():.1%} of the magnitude. A near-zero component is "
            "interchangeable with any other, so this configuration cannot "
            "detect the axis swap the test exists for. Use an asymmetric body "
            "and an offset reference point."
        )
    ordered = np.sort(np.abs(torque))
    separations = np.diff(ordered) / magnitude
    if float(separations.min()) < _MIN_COMPONENT_SEPARATION:
        raise VerificationError(
            f"{context}: torque {torque!r} has two components of nearly equal "
            "magnitude, which are interchangeable; swapping them would not "
            "change the residual and the test would pass vacuously."
        )
    if scale <= 0.0:
        raise VerificationError(f"{context}: the torque scale is not positive")


def moment_transfer_residual(
    solver: DRFTSolver, state: IntrusionState, *, reference_point_m: ArrayLike
) -> ConservationResidual:
    """Check ``tau(p2) = tau(p1) + (p1 - p2) x F`` between two reference points.

    The identity holds only if every element's force is applied at that
    element's own centroid.  A torque accumulated about the body origin,
    or about a fixed point regardless of the query's reference point --
    the "particle centre instead of contact point" defect -- breaks it
    while leaving the resultant force untouched.

    Args:
        solver: The solver under test.
        state: The query, which must have zero angular velocity so that
            moving the reference point does not change the physics.
        reference_point_m: The second reference point.

    Returns:
        The residual, round-off class.

    Raises:
        VerificationError: If the state rotates, or the torque is too
            degenerate to expose an axis swap.
    """
    if bool(np.asarray(state.angular_velocity_rad_s).any()):
        raise VerificationError(
            "moment transfer is only a pure bookkeeping identity for a "
            "non-rotating body: with a non-zero angular velocity the element "
            "velocities depend on the reference point and the two solves are "
            "different physical problems"
        )
    second_point = np.asarray(reference_point_m, dtype=np.float64).reshape(3)
    first = solver.solve(state)
    moved = IntrusionState(
        state.elements,
        state.velocity_m_s,
        angular_velocity_rad_s=state.angular_velocity_rad_s,
        reference_point_m=second_point,
        free_surface_height_m=state.free_surface_height_m,
    )
    second = solver.solve(moved)
    shifted = first.wrench.about(second_point)
    residual = float(np.abs(second.wrench.torque_n_m - shifted.torque_n_m).max())
    scale = float(
        np.abs(second.wrench.torque_n_m).max()
        + np.linalg.norm(state.reference_point_m - second_point)
        * first.force_magnitude_n
    )
    _require_detectable_torque(
        second.wrench.torque_n_m, scale, "moment transfer residual"
    )
    return ConservationResidual(
        name="angular momentum (moment transfer)",
        conservation_class=ConservationClass.ROUND_OFF,
        residual=residual,
        scale=scale,
    )


def element_moment_residual(
    solver: DRFTSolver, state: IntrusionState
) -> ConservationResidual:
    """Rebuild the torque with a naive oracle loop and compare.

    The solver replaces ``np.cross`` with a hand-written component form
    because ``np.cross`` shows up in a profile of the shot loop.  That is
    a sound optimisation and also exactly the place an index or sign error
    survives review, because **no force test can see it**: the resultant
    force does not involve the cross product at all.

    The oracle is a deliberately slow per-element ``np.cross`` loop, kept
    for the same reason the digest keeps a brute-force neighbour search:
    duplication of implementation is safe when the knowledge still has one
    authoritative representation.

    Args:
        solver: The solver under test.
        state: The query. Its body must be asymmetric enough that all
            three torque components are distinguishable.

    Returns:
        The residual, round-off class.

    Raises:
        VerificationError: If the query engages no elements, or the torque
            is too degenerate to expose an axis swap.
    """
    result = solver.solve(state)
    response = solver.element_response(state)
    if response.index.size == 0:
        raise VerificationError(
            "no element is both submerged and a leading edge, so there is no "
            "torque to check"
        )
    elements = state.elements
    areas = elements.areas_m2[response.index]
    tractions = response.depth_traction_pa + response.inertial_traction_pa
    centroids = elements.centroids_m[response.index]

    oracle = np.zeros(3, dtype=np.float64)
    lever_scale = 0.0
    for centroid, traction, area in zip(centroids, tractions, areas, strict=True):
        lever = centroid - state.reference_point_m
        force = traction * area
        oracle = oracle + np.cross(lever, force)
        lever_scale += float(np.linalg.norm(lever) * np.linalg.norm(force))

    residual = float(np.abs(result.wrench.torque_n_m - oracle).max())
    scale = max(lever_scale, float(np.abs(oracle).max()))
    _require_detectable_torque(oracle, scale, "element moment residual")
    return ConservationResidual(
        name="angular momentum (element moment oracle)",
        conservation_class=ConservationClass.ROUND_OFF,
        residual=residual,
        scale=scale,
    )


def energy_work_residual(
    trace: ShotResult, *, head_mass_kg: float, time_step_s: float
) -> ConservationResidual:
    """``|dKE - W|`` along the trace: the truncation-class residual.

    The shot integrator is semi-implicit Euler, which is not energy
    conserving, so this residual is **not** expected to vanish.  It is
    expected to *shrink linearly with the step*: the per-step defect is
    ``-(dt^2/2m)|F|^2`` and there are ``T/dt`` steps, so the total is
    ``O(dt)``.  Feed a series of these to
    :func:`~bunkershot3d.vandv.convergence.observed_order_from_residuals`
    and the observed order is the test.

    Args:
        trace: The shot trace, over a window in which the head stays
            engaged so every level covers the same interval.
        head_mass_kg: Head mass.
        time_step_s: The fixed step the trace was produced at.

    Returns:
        The residual, truncation class, carrying its step size.

    Raises:
        VerificationError: If the trace or the mass is unusable.
    """
    step = _trace_step(trace, time_step_s)
    mass = float(head_mass_kg)
    if not math.isfinite(mass) or mass <= 0.0:
        raise VerificationError(f"head_mass_kg must be positive, got {head_mass_kg!r}")
    velocities = trace.velocities_m_s
    kinetic = (
        0.5
        * mass
        * (
            float(velocities[-1] @ velocities[-1])
            - float(velocities[0] @ velocities[0])
        )
    )
    displacements = np.diff(trace.positions_m, axis=0)
    work = float(np.einsum("ij,ij->i", trace.forces_n[:-1], displacements).sum())
    magnitude = float(
        np.abs(np.einsum("ij,ij->i", trace.forces_n[:-1], displacements)).sum()
    )
    return ConservationResidual(
        name="energy (work-energy theorem)",
        conservation_class=ConservationClass.TRUNCATION,
        residual=abs(kinetic - work),
        scale=max(
            float(abs(kinetic)), float(magnitude), float(np.finfo(np.float64).tiny)
        ),
        step_size_s=step,
    )


def _require_prescribed(trace: ShotResult, ledger: str) -> None:
    """Refuse a coupled trace, whose rotational ledger the coupling keeps.

    Raises:
        VerificationError: If the trace was not marched with prescribed
            rotation, or is too short to integrate.
    """
    if trace.rotation_mode is not RotationMode.PRESCRIBED:
        raise VerificationError(
            f"{ledger} is an identity of the prescribed-rotation march, but "
            f"this trace was produced in {trace.rotation_mode.value} mode; a "
            "coupled head's angular momentum lives in the coupling's own "
            "inertia model, which this ledger does not have (issue #9544)"
        )
    if trace.n_steps < 2:
        raise VerificationError(
            f"{ledger} needs at least two samples to integrate, got {trace.n_steps}"
        )


@dataclass(frozen=True, slots=True)
class SupportAngularImpulse:
    """What the support supplied to hold a prescribed rotation.

    Both vectors are taken about the body-frame origin -- the point
    ``ShotResult.positions_m[k]``, which moves with the head -- in the
    world frame, the same reference every ``torques_n_m[k]`` uses.

    The gyroscopic term ``omega x I omega`` of a non-spherical head is
    **not** included: the F0 march carries no inertia tensor, so the
    ledger states only the part the sand moment fixes.  For a constant
    ``omega`` that term is a constant the sand does not touch.

    Attributes:
        sand_n_m_s: ``integral of tau dt``, trapezoidal, the angular impulse the
            sand put into the head.
        support_n_m_s: ``-sand_n_m_s``, the angular impulse the support
            had to supply so the angular velocity did not change.
        residual: ``|sand + support|``, identically zero: this is a
            round-off-class statement and it is reported as one so the
            ledger cannot be mistaken for a measurement.
        conservation_class: Always ``ROUND_OFF``.
    """

    sand_n_m_s: NDArray[np.float64]
    support_n_m_s: NDArray[np.float64]
    residual: float
    conservation_class: ConservationClass = ConservationClass.ROUND_OFF


def support_angular_impulse(trace: ShotResult) -> SupportAngularImpulse:
    """The angular impulse a prescribed march's support must have supplied.

    Translation is free, so the support's *linear* impulse is zero by the
    same idealisation; only the angular side needs stating.

    Args:
        trace: A prescribed-rotation shot trace.

    Returns:
        The sand and support angular impulses about the moving body origin.

    Raises:
        VerificationError: If the trace is coupled or too short.
    """
    _require_prescribed(trace, "the support angular-impulse ledger")
    sand = np.asarray(
        np.trapezoid(trace.torques_n_m, x=trace.times_s, axis=0), dtype=np.float64
    )
    support = -sand
    return SupportAngularImpulse(
        sand_n_m_s=sand,
        support_n_m_s=support,
        residual=float(np.abs(sand + support).max()),
    )


@dataclass(frozen=True, slots=True)
class PrescribedDriverWork:
    """The energy the driver spent holding the delivered rotation.

    With ``omega`` fixed the head's rotational kinetic energy cannot
    change, so the sand's torque work and the driver's work sum to zero
    identically: ``0 = W_sand_torque + W_driver``.  The translational
    side, ``dKE = W_force``, is :func:`energy_work_residual`'s.

    Attributes:
        sand_torque_work_j: ``integral of tau . omega dt``, trapezoidal.
        driver_work_j: ``-sand_torque_work_j``: positive when the driver
            had to push the head round against the sand.
    """

    sand_torque_work_j: float
    driver_work_j: float


def prescribed_driver_work(trace: ShotResult) -> PrescribedDriverWork:
    """The work done holding a prescribed rotation against the sand moment.

    Args:
        trace: A prescribed-rotation shot trace.

    Returns:
        The sand's torque work and the driver's equal-and-opposite work.

    Raises:
        VerificationError: If the trace is coupled or too short.
    """
    _require_prescribed(trace, "the prescribed-driver work ledger")
    power = np.einsum("ij,ij->i", trace.torques_n_m, trace.angular_velocities_rad_s)
    work = float(np.trapezoid(power, x=trace.times_s))
    return PrescribedDriverWork(sand_torque_work_j=work, driver_work_j=-work)


def inertial_power_is_dissipative(
    solver: DRFTSolver, state: IntrusionState
) -> tuple[float, bool]:
    """Return the inertial term's power and whether it removes energy.

    The DRFT inertial traction is ``-n lambda rho v_n^2`` with
    ``v_n = max(v . n, 0)``, so its power density is
    ``-lambda rho v_n^3 <= 0`` on every active element.  It is therefore
    *provably* dissipative, and this is an analytic limit check rather
    than a numerical tolerance.

    The depth term carries no such guarantee, because the tangential part
    surviving the surface-friction cutoff can point anywhere in the
    tangent plane.  This function deliberately reports only the term the
    algebra covers.

    Args:
        solver: The solver under test.
        state: The query.

    Returns:
        ``(power_w, is_dissipative)``.
    """
    response = solver.element_response(state)
    if response.index.size == 0:
        return (0.0, True)
    velocities = state.element_velocities_m_s()[response.index]
    areas = state.elements.areas_m2[response.index]
    power = float(
        np.einsum("ij,ij->i", response.inertial_traction_pa, velocities) @ areas
    )
    return (power, power <= 0.0)


def residual_series(
    residuals: Sequence[ConservationResidual],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Split a residual series into ``(step sizes, residuals)``.

    Args:
        residuals: Truncation-class residuals from a step refinement.

    Returns:
        The step sizes and the residuals, in the order supplied.

    Raises:
        ConservationClassError: If any residual is round-off class, whose
            decay is round-off noise rather than truncation error.
        VerificationError: If fewer than two levels are supplied.
    """
    if len(residuals) < 2:
        raise VerificationError(
            f"an order test needs at least two refinement levels, got {len(residuals)}"
        )
    steps: list[float] = []
    values: list[float] = []
    for item in residuals:
        if item.conservation_class is not ConservationClass.TRUNCATION:
            raise ConservationClassError(
                f"{item.name!r} is a round-off-class residual. The digest is "
                "explicit that round-off quantities get a fixed ~1e-12 "
                "tolerance and NO order test: refining the step does not "
                "shrink floating-point noise, so an order fitted to it "
                "describes the hardware, not the scheme."
            )
        if item.step_size_s is None:  # pragma: no cover - forbidden by __post_init__
            raise ConservationClassError(f"{item.name!r} carries no step size")
        steps.append(item.step_size_s)
        values.append(item.residual)
    return (tuple(steps), tuple(values))
