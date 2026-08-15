"""Sand-mediated momentum transfer for splash shots (issues #8613, #8657).

In a splash shot the club enters the sand behind the ball and never touches
it. Momentum reaches the ball through the sand the club threw.

What this module used to do, and why it changed
-----------------------------------------------

The first cut of this model estimated the displaced sand as a box -- sole
length times entry depth times a hard-coded 12 cm of sand contact -- and put a
flat efficiency on the result. Ball speed came out **linear in entry depth**
and blind to everything else. That was reasonable before the F0 solver
existed. It is not reasonable now, because:

* the solver returns the impulse it actually put into the bed, which is the
  physically meaningful driver, and
* :mod:`bunkershot3d.metrics.divot` measures the divot's mass from the sole
  path, which is the displaced sand rather than a guess at it.

Estimating with a magic number a quantity that is measured two modules away is
a DRY violation whose duplicated copy is the less accurate one, and carry
drives ``playability_window_area``, the primary scalar objective of the tool
(issue #8614). So the box is gone.

The partition
-------------

The moving sand is treated as a slug of mass ``m_divot`` carrying momentum
``J`` -- so its mean speed ``J / m_divot`` is *derived*, not a fraction of club
speed. Only the share ``f`` of that slug on a path that meets the ball can
reach it, and that share collides with the ball partially inelastically::

    p_int  = f * J                              momentum on a path to the ball
    m_int  = f * m_divot                        mass of that share
    p_ball = eta * m_b / (m_int + m_b) * p_int

Both dependencies are the right way round. At fixed sand mass the ball
responds linearly to the delivered impulse; at fixed impulse, moving more sand
means a slower slug and *less* ball speed, never more. And ``p_ball`` cannot
exceed ``J``, because ``eta <= 1`` and ``m_b / (m_int + m_b) <= 1``; the
partition is checked against that bound with a plain ``raise``.

What is still not grounded
--------------------------

``eta`` is uncalibrated. So is the launch **direction**, which is taken from
the effective loft: the momentum the head puts into the bed points forward and
*down*, and it is the free surface -- not modelled here -- that turns the
ejecta up and out. Per issue #8616 **no published measurement of ball speed,
launch angle or spin out of sand exists anywhere**, so none of these can be
calibrated by reading harder. Every result therefore carries a
:class:`~bunkershot3d.solvers.envelope.ValidityVerdict` floored at
``BEYOND_VALIDATION`` and a
:class:`~bunkershot3d.sand.provenance.SandProvenance` record naming the basis
of every parameter, in the same shapes the rest of the package uses.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field

from src.shared.python.core.contracts import require

from ..sand.provenance import PropertyProvenance, ProvenanceBasis, SandProvenance
from ..solvers.envelope import EnvelopeStatus, ValidityVerdict, worst_of
from .lie import BallLie, BallProperties, compute_exposed_cap_fraction

__all__ = [
    "BALL_LAUNCH_MEASUREMENT_GAP",
    "BALL_LAUNCH_UNCALIBRATED_REASON",
    "BALL_MOMENTUM_TRANSFER_EFFICIENCY",
    "DEFAULT_MOMENTUM_TRANSFER",
    "SAND_BALL_FRICTION",
    "SPIN_LEVER_ARM_FRACTION",
    "SUPERSONIC_EJECTA_REASON",
    "BallLaunchResult",
    "ContactType",
    "MomentumTransfer",
    "SandDelivery",
    "SplashTransferResult",
    "compute_ball_launch_from_splash",
    "compute_sand_ejecta_velocity",
    "compute_splash_impulse",
    "momentum_transfer_provenance",
]

BALL_MOMENTUM_TRANSFER_EFFICIENCY: float = 0.5
"""Share of the intercepted sand momentum the ball ends up with.

**Uncalibrated.** A partially inelastic collision through a granular stream
loses momentum to grains that glance off, to grains that arrive after the ball
has left, and to the sand-on-sand contacts inside the slug. None of that has
been measured for a bunker shot, so this is a stated placeholder that scales
the answer linearly and must be reported as such."""

SAND_BALL_FRICTION: float = 0.5
"""Tangential share of the ball impulse available to spin it up. Uncalibrated."""

SPIN_LEVER_ARM_FRACTION: float = 0.7
"""Where the sand stream meets the ball, as a fraction of the radius below the
centre. A modelling convention: it sets the spin lever arm and no measurement
of the contact patch exists."""

BALL_LAUNCH_MEASUREMENT_GAP = (
    "No published value exists anywhere for ball speed, launch angle or spin "
    "out of a bunker. An exhaustive enumeration of the sports-engineering "
    "literature found no bunker, sand or wedge-interaction paper at all "
    "(issue #8616), so this parameter is uncalibrated and cannot be calibrated "
    "by reading harder."
)

BALL_LAUNCH_UNCALIBRATED_REASON = (
    "ball launch is partitioned out of the delivered sand impulse through an "
    "uncalibrated transfer efficiency, and its direction is taken from the "
    "effective loft by convention; no published measurement of ball speed, "
    "launch angle or spin out of sand exists to calibrate either against "
    "(issue #8616)"
)
"""Why a carry number is beyond validation however good the shot behind it was."""

SUPERSONIC_EJECTA_REASON = (
    "the derived mean ejecta speed of {ejecta:.3g} m/s exceeds the head's "
    "entry speed of {entry:.3g} m/s, which sand thrown by that head cannot "
    "do. The prismatic divot model counts only the sand under the sole path "
    "and a real divot has sloped walls, so the mass that shared the delivered "
    "momentum is under-counted here and ball speed is over-predicted. The "
    "direction of the bias is known; its size is not."
)
"""Template for the diagnostic that fires when the divot mass under-counts.

Not a clamp. Capping the ejecta speed would make ball speed stop responding to
the delivered impulse in exactly the regime issue #8657 exists to fix, so the
inconsistency is reported instead of hidden."""


class ContactType(enum.Enum):
    """Type of club-ball-sand interaction."""

    SPLASH = "splash"  # Club never touches ball
    THIN = "thin"  # Club strikes ball directly (blade/thin shot)
    MIXED = "mixed"  # Both sand and direct contact


def _refuse(name: str, value: float, requirement: str) -> None:
    """Raise on an unusable measurement.

    A plain ``raise`` rather than a contract: ``python -O`` strips assertions
    and ``DBC_LEVEL=off`` disables contracts, and a momentum budget that
    evaporates under an optimisation flag is worse than none.

    Args:
        name: Field name, quoted in the message.
        value: The offending value.
        requirement: What the field must satisfy.

    Raises:
        ValueError: Always.
    """
    raise ValueError(f"{name} must be {requirement}, got {value!r}")


@dataclass(frozen=True, slots=True)
class MomentumTransfer:
    """The **uncalibrated** parameters of the sand-to-ball partition.

    Named and passed rather than inlined, so a reader can see how much of the
    answer rests on numbers nobody has measured, and a caller can sweep them.

    Attributes:
        efficiency: ``eta``, the share of intercepted sand momentum the ball
            keeps. Scales ball speed linearly.
        sand_ball_friction: Tangential share of the ball impulse that spins it.
        spin_lever_arm_fraction: Height below the ball centre at which the sand
            stream is taken to act, as a fraction of the radius.
    """

    efficiency: float = BALL_MOMENTUM_TRANSFER_EFFICIENCY
    sand_ball_friction: float = SAND_BALL_FRICTION
    spin_lever_arm_fraction: float = SPIN_LEVER_ARM_FRACTION

    def __post_init__(self) -> None:
        """Bound every parameter to ``[0, 1]``.

        The bound on :attr:`efficiency` is what makes the momentum budget
        provable, so it is a ``raise`` and not a contract.

        Raises:
            ValueError: If any parameter falls outside ``[0, 1]``.
        """
        for name in ("efficiency", "sand_ball_friction", "spin_lever_arm_fraction"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                _refuse(name, value, "a finite fraction in [0, 1]")


DEFAULT_MOMENTUM_TRANSFER = MomentumTransfer()
"""The shipped placeholder values. Nothing here is a calibration."""


def momentum_transfer_provenance(transfer: MomentumTransfer) -> SandProvenance:
    """Return the provenance record for one set of partition parameters.

    Args:
        transfer: The parameters the launch was computed with.

    Returns:
        A record naming the basis of every parameter, in the shape the sand
        presets and the RFT coefficients already use.
    """
    placeholder = f"chosen placeholder, {transfer.efficiency:.3g}; not a calibration"
    return SandProvenance(
        entries={
            "transfer_efficiency": PropertyProvenance(
                basis=ProvenanceBasis.ESTIMATED,
                source=placeholder,
                note="uncalibrated. " + BALL_LAUNCH_MEASUREMENT_GAP,
            ),
            "sand_ball_friction": PropertyProvenance(
                basis=ProvenanceBasis.ESTIMATED,
                source=f"chosen placeholder, {transfer.sand_ball_friction:.3g}",
                note="uncalibrated. " + BALL_LAUNCH_MEASUREMENT_GAP,
            ),
            "spin_lever_arm": PropertyProvenance(
                basis=ProvenanceBasis.CONVENTION,
                source=(
                    f"{transfer.spin_lever_arm_fraction:.3g} of the ball radius "
                    "below its centre"
                ),
                note=(
                    "a modelling convention for where the sand stream acts; no "
                    "measurement of the contact patch exists."
                ),
            ),
            "intercepted_fraction": PropertyProvenance(
                basis=ProvenanceBasis.CONVENTION,
                source="linear exposed-cap taper, bunkershot3d.ball.lie",
                note=(
                    "the share of the moving sand taken to meet the ball is the "
                    "share of its upper hemisphere a splash can still reach. The "
                    "taper is continuous at both ends and strictly decreasing "
                    "between them, and it is a convention, not cap geometry."
                ),
            ),
            "launch_direction": PropertyProvenance(
                basis=ProvenanceBasis.CONVENTION,
                source="effective loft of the delivered face",
                note=(
                    "the momentum the head puts into the bed points forward and "
                    "down; the free surface that turns the ejecta up is not "
                    "modelled, so the launch direction is taken from the loft. "
                    + BALL_LAUNCH_MEASUREMENT_GAP
                ),
            ),
        }
    )


@dataclass(frozen=True, slots=True)
class SandDelivery:
    """What the solver and the metrics layer measured about one strike.

    Every field is computed elsewhere: the impulse and the speeds by
    :func:`bunkershot3d.solvers.simulate_shot`, the displaced mass by
    :func:`bunkershot3d.metrics.divot_metrics`. Nothing in it is fitted here,
    which is the whole point of the object.

    Attributes:
        impulse_n_s: Magnitude of the sand impulse exchanged with the head
            [N.s]. By Newton's third law this is also the momentum the head
            delivered to the bed.
        displaced_mass_kg: Divot mass [kg] -- the sand actually moved.
        contact_duration_s: Time the sole spent engaged with the bed [s].
        entry_speed_m_s: Head speed at the first sample [m/s].
        exit_speed_m_s: Head speed at the last sample [m/s].
        verdict: The solver's validity statement for the strike, which the
            launch verdict is combined with so carry can never read better
            than the shot behind it.
    """

    impulse_n_s: float
    displaced_mass_kg: float
    contact_duration_s: float
    entry_speed_m_s: float
    exit_speed_m_s: float
    verdict: ValidityVerdict

    def __post_init__(self) -> None:
        """Refuse a strike that was not measured.

        Raises:
            ValueError: If any measurement is unusable, or the verdict is
                missing. A launch computed from a defaulted impulse would be
                exactly the invented number this rewrite removed.
        """
        if not isinstance(self.verdict, ValidityVerdict):
            raise ValueError(
                "a sand delivery must carry the solver's verdict for the strike; "
                f"got {type(self.verdict).__name__}. A carry number without one "
                "cannot be reported (issue #8657)"
            )
        for name in ("impulse_n_s", "contact_duration_s", "entry_speed_m_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                _refuse(name, value, "finite and non-negative")
        if not math.isfinite(self.exit_speed_m_s) or self.exit_speed_m_s < 0.0:
            _refuse("exit_speed_m_s", self.exit_speed_m_s, "finite and non-negative")
        if not math.isfinite(self.displaced_mass_kg) or self.displaced_mass_kg <= 0.0:
            _refuse(
                "displaced_mass_kg",
                self.displaced_mass_kg,
                "finite and positive -- a strike that moved no sand has no ejecta",
            )

    @property
    def mean_ejecta_speed_m_s(self) -> float:
        """``J / m_divot``: the mean speed of the sand that was moved [m/s]."""
        return self.impulse_n_s / self.displaced_mass_kg


@dataclass(frozen=True, slots=True)
class SplashTransferResult:
    """How the delivered momentum was partitioned between sand and ball.

    Attributes:
        impulse_x_ns: Ball impulse in x (forward) [N.s].
        impulse_y_ns: Ball impulse in y (lateral) [N.s].
        impulse_z_ns: Ball impulse in z (up) [N.s].
        ball_impulse_n_s: Magnitude of the ball impulse [N.s]. Never exceeds
            :attr:`delivered_impulse_n_s`.
        delivered_impulse_n_s: Momentum the head put into the bed [N.s], from
            the solver.
        angular_impulse_x_ns: Angular impulse about x [N.m.s].
        angular_impulse_y_ns: Angular impulse about y [N.m.s]; negative is
            backspin.
        angular_impulse_z_ns: Angular impulse about z [N.m.s].
        intercepted_fraction: Share of the moving sand on a path to the ball.
        intercepted_mass_kg: ``intercepted_fraction * divot mass`` [kg].
        ejecta_speed_m_s: Mean speed of the moving sand [m/s], derived.
        contact_duration_s: Measured engagement time [s].
    """

    impulse_x_ns: float
    impulse_y_ns: float
    impulse_z_ns: float
    ball_impulse_n_s: float
    delivered_impulse_n_s: float
    angular_impulse_x_ns: float
    angular_impulse_y_ns: float
    angular_impulse_z_ns: float
    intercepted_fraction: float
    intercepted_mass_kg: float
    ejecta_speed_m_s: float
    contact_duration_s: float


@dataclass(frozen=True, slots=True)
class BallLaunchResult:
    """Ball launch conditions, with the statement they may be quoted under.

    Attributes:
        ball_speed_m_s: Ball launch speed [m/s].
        launch_angle_rad: Launch angle from horizontal [rad].
        azimuth_rad: Azimuth angle [rad], 0 = forward.
        spin_rate_rpm: Spin rate [RPM].
        spin_axis: Spin axis unit vector.
        ball_velocity: Ball velocity vector [m/s].
        ball_angular_velocity: Ball angular velocity [rad/s].
        contact_type: Type of contact.
        energy_transfer_fraction: Share of head kinetic energy the ball holds.
        delivered_impulse_n_s: Momentum the head put into the bed [N.s].
        ball_impulse_n_s: Momentum the ball received [N.s].
        verdict: The solver's verdict combined with the launch model's own,
            floored at ``BEYOND_VALIDATION``.
        provenance: Basis of every parameter of the partition.
    """

    ball_speed_m_s: float
    launch_angle_rad: float
    azimuth_rad: float
    spin_rate_rpm: float
    spin_axis: tuple[float, float, float]
    ball_velocity: tuple[float, float, float]
    ball_angular_velocity: tuple[float, float, float]
    contact_type: ContactType
    energy_transfer_fraction: float
    delivered_impulse_n_s: float
    ball_impulse_n_s: float
    verdict: ValidityVerdict
    provenance: SandProvenance = field(default_factory=SandProvenance)

    def measured_constants(self) -> tuple[str, ...]:
        """Names of every constant measured on a real bunker shot.

        Empty, and required to stay empty: mirrors
        :meth:`bunkershot3d.solvers.MaterialResponse.measured_constants`.
        """
        return self.provenance.measured_properties()


def compute_sand_ejecta_velocity(delivery: SandDelivery) -> float:
    """Return the mean speed of the sand the club threw.

    Derived from the two measured quantities rather than fitted: the momentum
    the solver delivered, divided by the mass the metrics layer says was moved.

    Args:
        delivery: The measured strike.

    Returns:
        Mean ejecta speed [m/s].
    """
    return delivery.mean_ejecta_speed_m_s


def compute_splash_impulse(
    *,
    lie: BallLie,
    ball: BallProperties,
    delivery: SandDelivery,
    club_loft_rad: float,
    transfer: MomentumTransfer = DEFAULT_MOMENTUM_TRANSFER,
) -> SplashTransferResult:
    """Partition the delivered momentum between the bed and the ball.

    Args:
        lie: Ball position and burial in the sand.
        ball: Ball properties.
        delivery: The measured strike.
        club_loft_rad: Effective loft at delivery [rad], which sets the launch
            direction by convention.
        transfer: The uncalibrated partition parameters.

    Returns:
        The partition, carrying both sides of the momentum budget.

    Raises:
        ValueError: If the partition would give the ball more momentum than
            the head gave the bed. A plain ``raise``: the budget must survive
            ``DBC_LEVEL=off``.
    """
    require(
        0.0 < club_loft_rad < math.pi / 2,
        "loft must be in (0, pi/2)",
        club_loft_rad,
    )
    intercepted = compute_exposed_cap_fraction(lie, ball)
    intercepted_mass_kg = intercepted * delivery.displaced_mass_kg
    ball_impulse = (
        transfer.efficiency
        * (ball.mass_kg / (intercepted_mass_kg + ball.mass_kg))
        * (intercepted * delivery.impulse_n_s)
    )
    if ball_impulse > delivery.impulse_n_s:
        raise ValueError(
            f"the partition gave the ball {ball_impulse:.6g} N.s out of a "
            f"delivered {delivery.impulse_n_s:.6g} N.s; a splash cannot hand the "
            "ball more momentum than the head handed the bed"
        )
    lever_arm_m = ball.radius_m * transfer.spin_lever_arm_fraction
    return SplashTransferResult(
        impulse_x_ns=ball_impulse * math.cos(club_loft_rad),
        impulse_y_ns=0.0,
        impulse_z_ns=ball_impulse * math.sin(club_loft_rad),
        ball_impulse_n_s=ball_impulse,
        delivered_impulse_n_s=delivery.impulse_n_s,
        angular_impulse_x_ns=0.0,
        angular_impulse_y_ns=-lever_arm_m * ball_impulse * transfer.sand_ball_friction,
        angular_impulse_z_ns=0.0,
        intercepted_fraction=intercepted,
        intercepted_mass_kg=intercepted_mass_kg,
        ejecta_speed_m_s=delivery.mean_ejecta_speed_m_s,
        contact_duration_s=delivery.contact_duration_s,
    )


def _launch_reasons(delivery: SandDelivery) -> tuple[str, ...]:
    """Return the findings the launch model raises about one strike.

    Args:
        delivery: The measured strike.

    Returns:
        The uncalibrated-transfer statement, plus the under-counted-divot
        diagnostic when the two measured quantities disagree with each other.
    """
    ejecta = delivery.mean_ejecta_speed_m_s
    entry = delivery.entry_speed_m_s
    if entry > 0.0 and ejecta > entry:
        return (
            BALL_LAUNCH_UNCALIBRATED_REASON,
            SUPERSONIC_EJECTA_REASON.format(ejecta=ejecta, entry=entry),
        )
    return (BALL_LAUNCH_UNCALIBRATED_REASON,)


def launch_verdict(delivery: SandDelivery) -> ValidityVerdict:
    """Combine the solver's verdict with the launch model's own.

    The launch model's own verdict is ``BEYOND_VALIDATION`` unconditionally,
    because per issue #8616 there is no published ball speed, launch angle or
    spin to compare against; combining it with the solver's means a carry
    number can never read better than the shot it came from, and never reads
    as though it were measured.

    Args:
        delivery: The measured strike, carrying the solver's verdict.

    Returns:
        The combined verdict, on the solver's feature scales, with the mean
        ejecta speed carried in its details so a manifest can show what the
        two measured quantities implied.
    """
    solver = delivery.verdict
    details = dict(solver.details)
    details["mean_ejecta_speed_m_s"] = delivery.mean_ejecta_speed_m_s
    # The launch verdict is listed first so that it wins a tie in ``worst_of``
    # -- both are normally BEYOND_VALIDATION -- and its merged details, which
    # already include the solver's, survive onto the combined verdict.
    return worst_of(
        (
            ValidityVerdict(
                status=EnvelopeStatus.BEYOND_VALIDATION,
                groups=solver.groups,
                governing_index=solver.governing_index,
                reasons=_launch_reasons(delivery),
                details=details,
            ),
            solver,
        )
    )


def _angular_velocity(
    splash: SplashTransferResult, ball: BallProperties
) -> tuple[float, float, float]:
    """Return the ball's angular velocity from the angular impulse.

    Args:
        splash: The partition.
        ball: Ball properties, supplying the moment of inertia.

    Returns:
        ``(omega_x, omega_y, omega_z)`` [rad/s].
    """
    moi = ball.moi_kg_m2
    return (
        splash.angular_impulse_x_ns / moi,
        splash.angular_impulse_y_ns / moi,
        splash.angular_impulse_z_ns / moi,
    )


def compute_ball_launch_from_splash(
    *,
    lie: BallLie,
    ball: BallProperties,
    delivery: SandDelivery,
    club_loft_rad: float,
    club_mass_kg: float = 0.30,
    transfer: MomentumTransfer = DEFAULT_MOMENTUM_TRANSFER,
) -> BallLaunchResult:
    """Compute ball launch conditions from a measured strike.

    Args:
        lie: Ball position and burial in the sand.
        ball: Ball properties.
        delivery: The measured strike: the solver's impulse and speeds and the
            metrics layer's divot mass.
        club_loft_rad: Effective loft at delivery [rad].
        club_mass_kg: Head mass [kg], for the energy share only.
        transfer: The uncalibrated partition parameters.

    Returns:
        The launch, its validity verdict and the provenance of every parameter
        the partition used.

    Raises:
        ValueError: If the momentum budget does not close.
    """
    splash = compute_splash_impulse(
        lie=lie,
        ball=ball,
        delivery=delivery,
        club_loft_rad=club_loft_rad,
        transfer=transfer,
    )
    velocity = (
        splash.impulse_x_ns / ball.mass_kg,
        splash.impulse_y_ns / ball.mass_kg,
        splash.impulse_z_ns / ball.mass_kg,
    )
    ball_speed = math.hypot(*velocity)
    horizontal = math.hypot(velocity[0], velocity[1])
    launch_angle = (
        math.atan2(velocity[2], horizontal) if horizontal > 1e-10 else club_loft_rad
    )
    azimuth = math.atan2(velocity[1], velocity[0]) if horizontal > 1e-10 else 0.0

    omega = _angular_velocity(splash, ball)
    spin_rate_rad_s = math.hypot(*omega)
    spin_axis: tuple[float, float, float] = (
        (
            omega[0] / spin_rate_rad_s,
            omega[1] / spin_rate_rad_s,
            omega[2] / spin_rate_rad_s,
        )
        if spin_rate_rad_s > 1e-10
        else (0.0, -1.0, 0.0)  # pure backspin, the splash-shot default
    )

    head_ke = 0.5 * club_mass_kg * delivery.entry_speed_m_s**2
    ball_ke = 0.5 * ball.mass_kg * ball_speed**2
    ball_rot_ke = 0.5 * ball.moi_kg_m2 * spin_rate_rad_s**2
    return BallLaunchResult(
        ball_speed_m_s=ball_speed,
        launch_angle_rad=launch_angle,
        azimuth_rad=azimuth,
        spin_rate_rpm=spin_rate_rad_s * 60.0 / (2.0 * math.pi),
        spin_axis=spin_axis,
        ball_velocity=velocity,
        ball_angular_velocity=omega,
        contact_type=ContactType.SPLASH,
        energy_transfer_fraction=(
            (ball_ke + ball_rot_ke) / head_ke if head_ke > 0.0 else 0.0
        ),
        delivered_impulse_n_s=splash.delivered_impulse_n_s,
        ball_impulse_n_s=splash.ball_impulse_n_s,
        verdict=launch_verdict(delivery),
        provenance=momentum_transfer_provenance(transfer),
    )
