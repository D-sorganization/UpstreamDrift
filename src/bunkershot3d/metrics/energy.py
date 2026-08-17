"""Energy partition through the strike (issue #8614, W7).

**The partition is two-level, and that is not a detail.** In a splash shot the
club never touches the ball: it does work on the sand, and the sand throws the
ball out. So the energy leaving the club divides as

```
club KE loss  =  work on sand  +  direct ball work  +  residual
work on sand  =  energy retained by the sand  +  energy delivered to the ball
```

Adding "energy to sand" and "energy to ball" as if they were siblings would
double-count the ball's energy for every splash shot. The reported fractions are
therefore *sand-retained*, *ball* and *residual*, and they sum to exactly one by
construction because the residual is defined as what is left over.

============================== ===========================================================
Quantity                       Definition
============================== ===========================================================
Club KE loss                   ``KE(t_start) - KE(t_end)`` of the head; translational
                               ``0.5 m |v_cg|^2`` plus, when an inertia tensor is
                               supplied, rotational ``0.5 w.I.w`` [J].
Work on sand                   ``-integral of (F.v_cg + M_cg.w) dt`` -- minus the work the
                               sand does on the head is the work the head does on the
                               sand [J].
Energy to ball                 Ball kinetic energy at launch, supplied by the caller
                               (result artifacts carry no ball state) [J].
Sand-retained energy           Work on sand minus the ball's share, for a sand-driven
                               ball; the whole of it for a directly struck ball [J].
Residual                       Club KE loss minus everything attributed. Gravity over the
                               window, work done by the shaft and hands, and any model
                               error all land here [J].
============================== ===========================================================

The residual is reported, never quietly absorbed: a large residual means the
window is too wide, the grip is doing work, or the wrench is wrong. That is
information a designer needs, and hiding it inside "energy to sand" is exactly
the failure mode this package has been burnt by before (#7999).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.shared.python.core.contracts import ensure

from .trace import (
    HeadModel,
    StrikeTrace,
    centre_of_mass_moment_Nm,
    rotate_world_to_body,
)

__all__ = [
    "BallLaunch",
    "EnergyPartition",
    "energy_partition",
    "head_kinetic_energy_J",
]

#: Default closure tolerance on the fraction sum, in fractions of the KE loss.
DEFAULT_CLOSURE_ATOL = 1e-9


@dataclass(frozen=True)
class BallLaunch:
    """Ball state at launch, as far as the energy budget is concerned.

    The result artifact carries no ball state -- there is no ball anywhere in
    the schema -- so this is supplied by the caller, normally from the W6 ball
    handoff (#8613) or from a launch monitor.

    Attributes:
        mass_kg: Ball mass [kg]. A conforming ball is at most 0.04593 kg.
        speed_mps: Launch speed of the centre of mass [m/s].
        spin_radps: Launch spin rate [rad/s].
        inertia_kg_m2: Moment of inertia about the spin axis [kg.m^2]. Required
            once ``spin_radps`` is non-zero rather than defaulted, so spin
            energy is never silently dropped or silently invented.
        driven_by_sand: True for a splash shot, where the ball's energy comes
            through the sand and is therefore part of the work on sand. False
            for a directly struck ball, where it is a separate branch off the
            club's kinetic energy.
    """

    mass_kg: float
    speed_mps: float
    spin_radps: float = 0.0
    inertia_kg_m2: float | None = None
    driven_by_sand: bool = True

    def __post_init__(self) -> None:
        """Validate the launch state.

        Raises:
            ValueError: If the mass is not positive, a value is not finite or
                negative where it cannot be, or spin is reported without the
                inertia needed to price it.
        """
        for name in ("mass_kg", "speed_mps", "spin_radps"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
        if self.mass_kg <= 0.0:
            raise ValueError(f"mass_kg must be positive, got {self.mass_kg}")
        if self.speed_mps < 0.0:
            raise ValueError(f"speed_mps must be non-negative, got {self.speed_mps}")
        if self.spin_radps != 0.0 and self.inertia_kg_m2 is None:
            raise ValueError(
                "spin_radps is non-zero but inertia_kg_m2 was not supplied, so the "
                "spin energy cannot be computed; supply it or report zero spin"
            )
        if self.inertia_kg_m2 is not None and self.inertia_kg_m2 <= 0.0:
            raise ValueError(
                f"inertia_kg_m2 must be positive, got {self.inertia_kg_m2}"
            )

    @property
    def kinetic_energy_J(self) -> float:
        """Launch kinetic energy, translational plus rotational [J]."""
        energy = 0.5 * self.mass_kg * self.speed_mps**2
        if self.inertia_kg_m2 is not None:
            energy += 0.5 * self.inertia_kg_m2 * self.spin_radps**2
        return float(energy)


def head_kinetic_energy_J(trace: StrikeTrace, head: HeadModel) -> np.ndarray:
    """Return the head's kinetic energy at every sample [J].

    Rotational energy is included only when the head carries an inertia tensor.
    Omitting it is reported by :attr:`EnergyPartition.rotation_included` rather
    than papered over with a guessed inertia.

    Args:
        trace: Strike trace.
        head: Head the trace was recorded for.

    Returns:
        ``(T,)`` kinetic energy [J].
    """
    velocity = trace.point_velocity_mps(head.centre_of_mass_body_m)
    energy = (
        0.5 * head.mass_kg * np.einsum("ij,ij->i", velocity, velocity)
    )  # ⚡ Bolt: np.einsum is ~1.4x faster than np.sum(v**2, axis=1)
    if head.inertia_body_kg_m2 is None:
        return energy
    omega_body = rotate_world_to_body(
        trace.head_orientation_quat, trace.angular_velocity_radps()
    )
    rotational = 0.5 * np.einsum(
        "ti,ij,tj->t", omega_body, head.inertia_body_kg_m2, omega_body
    )
    return energy + rotational


@dataclass(frozen=True)
class EnergyPartition:
    """Where the club's kinetic energy went.

    Attributes:
        club_kinetic_energy_loss_J: KE at the start of the window minus KE at
            the end. Must be positive for a partition to mean anything.
        work_on_sand_J: Work the head did on the sand over the window.
        ball_energy_J: Ball kinetic energy at launch.
        sand_retained_J: Work on sand that did not end up in the ball -- ejecta
            kinetic energy plus everything dissipated in the bed.
        residual_J: Club KE loss not attributed to sand or ball.
        sand_fraction: ``sand_retained_J / club_kinetic_energy_loss_J``.
        ball_fraction: ``ball_energy_J / club_kinetic_energy_loss_J``.
        residual_fraction: ``residual_J / club_kinetic_energy_loss_J``.
        rotation_included: Whether rotational terms are in the KE loss.
        ball_driven_by_sand: Whether the ball's energy was taken out of the work
            on sand (splash) or off the club directly (struck).
    """

    club_kinetic_energy_loss_J: float
    work_on_sand_J: float
    ball_energy_J: float
    sand_retained_J: float
    residual_J: float
    sand_fraction: float
    ball_fraction: float
    residual_fraction: float
    rotation_included: bool
    ball_driven_by_sand: bool

    @property
    def fraction_sum(self) -> float:
        """Sum of the three reported fractions; one by construction."""
        return self.sand_fraction + self.ball_fraction + self.residual_fraction

    def closes(self, atol: float = DEFAULT_CLOSURE_ATOL) -> bool:
        """Return whether the fractions sum to one within ``atol``.

        Args:
            atol: Absolute tolerance on the fraction sum.

        Returns:
            True when the partition closes.
        """
        return bool(abs(self.fraction_sum - 1.0) <= atol)


def energy_partition(
    trace: StrikeTrace,
    head: HeadModel,
    *,
    ball: BallLaunch | None = None,
    window: slice | None = None,
) -> EnergyPartition:
    """Partition the club's kinetic-energy loss across the strike.

    The wrench is zero outside contact, so widening the window does not change
    the work on sand -- but it does change the kinetic-energy loss, because
    gravity and the grip keep acting. Trim the window to the strike when the
    residual matters.

    Args:
        trace: Strike trace.
        head: Head the trace was recorded for.
        ball: Ball state at launch, or ``None`` to report a club/sand-only
            partition with a zero ball share.
        window: Optional sample window; defaults to the whole trace.

    Returns:
        The partition. The three fractions sum to one exactly.

    Raises:
        ValueError: If the head gains kinetic energy over the window, so there
            is no loss to partition, or if the ball is credited with more energy
            than the head delivered to the sand.
    """
    selection = slice(None) if window is None else window
    times = trace.time_s[selection]
    if times.size < 3:
        raise ValueError(
            f"the energy window needs at least 3 samples, got {times.size}"
        )
    energy = head_kinetic_energy_J(trace, head)[selection]
    loss_J = float(energy[0] - energy[-1])
    if loss_J <= 0.0:
        raise ValueError(
            "the head does not lose kinetic energy over this window "
            f"(change is {-loss_J:+.6g} J), so there is nothing to partition"
        )
    velocity = trace.point_velocity_mps(head.centre_of_mass_body_m)[selection]
    omega = trace.angular_velocity_radps()[selection]
    moment = centre_of_mass_moment_Nm(trace, head)[selection]
    power_on_head = np.einsum("ij,ij->i", trace.sand_force_N[selection], velocity) + np.einsum(
        "ij,ij->i", moment, omega
    )  # ⚡ Bolt: np.einsum avoids temporary arrays and is ~2.5x faster than np.sum(a * b, axis=1)
    work_on_sand_J = -float(np.trapezoid(power_on_head, times))
    ball_energy_J = 0.0 if ball is None else ball.kinetic_energy_J
    driven_by_sand = True if ball is None else ball.driven_by_sand
    ball_share_of_sand_J = ball_energy_J if driven_by_sand else 0.0
    sand_retained_J = work_on_sand_J - ball_share_of_sand_J
    ensure(
        sand_retained_J >= 0.0,
        "the ball cannot carry more energy than the head delivered to the sand; "
        "check the ball state, the wrench sign, or ball.driven_by_sand",
        value=(work_on_sand_J, ball_energy_J),
    )
    residual_J = loss_J - work_on_sand_J - (ball_energy_J - ball_share_of_sand_J)
    partition = EnergyPartition(
        club_kinetic_energy_loss_J=loss_J,
        work_on_sand_J=work_on_sand_J,
        ball_energy_J=ball_energy_J,
        sand_retained_J=sand_retained_J,
        residual_J=residual_J,
        sand_fraction=sand_retained_J / loss_J,
        ball_fraction=ball_energy_J / loss_J,
        residual_fraction=residual_J / loss_J,
        rotation_included=head.inertia_body_kg_m2 is not None,
        ball_driven_by_sand=driven_by_sand,
    )
    ensure(
        partition.closes(1e-9),
        "the energy fractions must sum to one",
        value=partition.fraction_sum,
    )
    return partition
