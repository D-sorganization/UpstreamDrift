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

``eta`` depends on the lie (issue #8704)
----------------------------------------

It did not, and the model was backwards because of it: holding geometry and
delivery fixed and sweeping the bed gave firm 12.13 m/s, fluffy 12.55 m/s and
plugged 12.60 m/s -- a plugged lie launching the ball *fastest*.

The sand model was not at fault. The delivered impulse barely moves across the
four conditions (2.848-2.855 N.s, 0.2 %), because the head gives up almost the
same momentum whatever it decelerates against: head speed loss moved 1.1 %
while peak sand force moved 22 %. What did move was the divot **mass**, which
scales with bulk density -- 65.4 g in firm sand against 61.2 g in plugged -- so
the added-mass term ``m_b / (m_int + m_b)`` *penalised* the firm bed, and a
lie-independent ``eta`` left nothing to pay it back.

A transfer model whose efficiency does not depend on the lie cannot represent a
lie-dependent splash, so ``eta`` is now a function of the bed's packing state::

    eta(D_r) = efficiency * (1 - packing_sensitivity * (1 - D_r))

increasing in relative density, equal to the stated dense-bed value at
``D_r = 1`` and never above it, so the bound above still holds. The direction
comes from critical-state soil mechanics -- loose sand contracts under shear
and spends momentum rearranging grains, dense sand carries it out through force
chains -- and the **magnitude is assumed, not measured**. See
:data:`BED_PACKING_TRANSFER_SENSITIVITY`.

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
    "BED_PACKING_DEPENDENCE_REASON",
    "BED_PACKING_TRANSFER_SENSITIVITY",
    "DEFAULT_MOMENTUM_TRANSFER",
    "INADMISSIBLE_EJECTA_REFUSAL",
    "MASS_INTERVAL_FLOOR_REASON",
    "SAND_BALL_FRICTION",
    "SPIN_LEVER_ARM_FRACTION",
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
"""Share of the intercepted sand momentum a ball struck out of a **fully dense**
bed ends up with.

**Uncalibrated.** A partially inelastic collision through a granular stream
loses momentum to grains that glance off, to grains that arrive after the ball
has left, and to the sand-on-sand contacts inside the slug. None of that has
been measured for a bunker shot, so this is a stated placeholder that scales
the answer linearly and must be reported as such.

Since issue #8704 this is the value at relative density 1 rather than a flat
constant; see :data:`BED_PACKING_TRANSFER_SENSITIVITY`."""

BED_PACKING_TRANSFER_SENSITIVITY: float = 0.5
"""Share of the transfer efficiency a **fully loose** bed costs, so that

    eta(D_r) = efficiency * (1 - sensitivity * (1 - D_r))

**The sign is physical; the size is assumed.** Critical-state soil mechanics
has it that a dense sand *dilates* under shear while a loose one *contracts*:
in a loose bed the grains still have voids to rearrange into, so a larger share
of the delivered momentum is spent on compaction and on grain-on-grain
collisions and never leaves as a directed stream. A dense bed is already near
jamming and carries momentum out through force chains. That is the mechanism
by which a fluffy or plugged lie plays dead, and it is the direction issue
#8704 found the model getting backwards.

The **magnitude** -- a fully loose bed delivering half the share a fully dense
one does -- is a stated placeholder, not a calibration. Per issue #8616 no
published measurement of ball speed out of sand exists to fit it to, and the
provenance record says so.

Bounding it at or below 1 is what keeps ``eta <= efficiency <= 1``, and so
keeps the momentum-budget proof of issue #8657 intact."""

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

BED_PACKING_DEPENDENCE_REASON = (
    "the share of the intercepted sand momentum the ball keeps is taken to "
    "fall with the bed's relative density, because a loose bed contracts under "
    "shear and spends momentum rearranging grains that a dense one carries out "
    "through force chains. The direction is the one issue #8704 found the "
    "model getting backwards -- softer lies were launching the ball faster -- "
    "and it is physically motivated, but its magnitude is assumed and not "
    "measured (issue #8616)"
)
"""Why the lie-dependence exists, and what about it is still an assumption."""

BALL_LAUNCH_UNCALIBRATED_REASON = (
    "ball launch is partitioned out of the delivered sand impulse through an "
    "uncalibrated transfer efficiency, and its direction is taken from the "
    "effective loft by convention; no published measurement of ball speed, "
    "launch angle or spin out of sand exists to calibrate either against "
    "(issue #8616)"
)
"""Why a carry number is beyond validation however good the shot behind it was."""

INADMISSIBLE_EJECTA_REFUSAL = (
    "the sand this strike moved would have to leave at {ejecta:.4g} m/s to "
    "carry the {impulse:.4g} N.s the solver delivered, from a head that "
    "arrived at {entry:.4g} m/s. Sand cannot leave faster than the thing that "
    "threw it, so this pair of numbers does not describe a strike and no ball "
    "launch is derivable from it. The mass is the suspect quantity: see "
    "bunkershot3d.metrics.divot.AcceleratedSandMass, which is what a shipped "
    "caller passes here (issue #8659)"
)
"""Message of the refusal that replaces #8657's supersonic-ejecta diagnostic.

A refusal and **not** a clamp. Capping the ejecta speed would make ball speed
stop responding to the delivered impulse in exactly the regime issue #8657
exists to fix; refusing says the inputs are inconsistent without inventing a
consistent pair, which is the one thing a caller can act on."""

MASS_INTERVAL_FLOOR_REASON = (
    "the momentum budget excludes the lower part of the accelerated-mass "
    "interval on its own: the delivered {impulse:.4g} N.s over the interval's "
    "lower edge of {lower:.4g} kg needs {implied:.4g} m/s of ejecta against a "
    "{entry:.4g} m/s head, so the mass is at least {floor:.4g} kg however the "
    "interval was formed. The number reported here is not clamped to that "
    "floor -- it is the interval's own central value, and the floor is quoted "
    "so the asymmetry of the remaining uncertainty is visible (issue #8659)"
)
"""Template for the diagnostic that fires when only the band's lower edge is
inadmissible. This is information about the *width* of the interval, not a
statement that the reported mass is wrong, so it is reported and not raised."""


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
        efficiency: ``eta`` at relative density 1, the share of intercepted
            sand momentum a ball struck out of a fully dense bed keeps. Scales
            ball speed linearly.
        sand_ball_friction: Tangential share of the ball impulse that spins it.
        spin_lever_arm_fraction: Height below the ball centre at which the sand
            stream is taken to act, as a fraction of the radius.
        packing_sensitivity: Share of :attr:`efficiency` a fully loose bed
            costs. Zero recovers the lie-independent model issue #8704 was
            filed against.
    """

    efficiency: float = BALL_MOMENTUM_TRANSFER_EFFICIENCY
    sand_ball_friction: float = SAND_BALL_FRICTION
    spin_lever_arm_fraction: float = SPIN_LEVER_ARM_FRACTION
    packing_sensitivity: float = BED_PACKING_TRANSFER_SENSITIVITY

    def __post_init__(self) -> None:
        """Bound every parameter to ``[0, 1]``.

        The bounds on :attr:`efficiency` and :attr:`packing_sensitivity` are
        what make the momentum budget provable -- together they cap the
        effective efficiency at :attr:`efficiency` -- so they are a ``raise``
        and not a contract.

        Raises:
            ValueError: If any parameter falls outside ``[0, 1]``.
        """
        for name in (
            "efficiency",
            "sand_ball_friction",
            "spin_lever_arm_fraction",
            "packing_sensitivity",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                _refuse(name, value, "a finite fraction in [0, 1]")

    def efficiency_for(self, bed_relative_density: float) -> float:
        """Return the share of intercepted momentum the ball keeps in one bed.

        ``eta(D_r) = efficiency * (1 - packing_sensitivity * (1 - D_r))``:
        increasing in relative density, equal to :attr:`efficiency` in a fully
        dense bed and never above it. See
        :data:`BED_PACKING_TRANSFER_SENSITIVITY` for the mechanism assumed and
        for what about it is not measured.

        Args:
            bed_relative_density: ``D_r`` of the bed the strike was made in,
                0 at the loosest packing and 1 at the densest.

        Returns:
            The effective transfer efficiency, in ``[0, efficiency]``.

        Raises:
            ValueError: If ``bed_relative_density`` is not a finite fraction.
                A plain ``raise``: the momentum budget rests on the bound and
                must survive ``python -O``.
        """
        density = float(bed_relative_density)
        if not math.isfinite(density) or not 0.0 <= density <= 1.0:
            _refuse("bed relative density", density, "a finite fraction in [0, 1]")
        efficiency = self.efficiency * (
            1.0 - self.packing_sensitivity * (1.0 - density)
        )
        if efficiency > self.efficiency:
            raise ValueError(
                f"the bed-dependent efficiency {efficiency:.6g} exceeds the "
                f"dense-bed value {self.efficiency:.6g}; the momentum budget "
                "of issue #8657 rests on it not doing so"
            )
        return efficiency


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
    placeholder = (
        f"chosen placeholder, {transfer.efficiency:.3g} in a fully dense bed; "
        "not a calibration"
    )
    return SandProvenance(
        entries={
            "transfer_efficiency": PropertyProvenance(
                basis=ProvenanceBasis.ESTIMATED,
                source=placeholder,
                note="uncalibrated. " + BALL_LAUNCH_MEASUREMENT_GAP,
            ),
            "bed_packing_dependence": PropertyProvenance(
                basis=ProvenanceBasis.ESTIMATED,
                source=(
                    f"a fully loose bed costs "
                    f"{transfer.packing_sensitivity:.3g} of the dense-bed "
                    "transfer efficiency, linear in relative density"
                ),
                note=(
                    "the direction is physically motivated -- a loose bed "
                    "contracts under shear and spends momentum rearranging "
                    "grains, which is why a plugged lie plays dead -- but the "
                    "magnitude is assumed, not measured. " + BALL_LAUNCH_MEASUREMENT_GAP
                ),
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
        displaced_mass_kg: The mass of sand the strike **accelerated** [kg],
            which is not the swept divot prism. Since issue #8659 a shipped
            caller passes
            :attr:`bunkershot3d.metrics.divot.AcceleratedSandMass.central_kg`;
            the prism counted only the sand under the sole path and dividing
            the delivered impulse by it implied ejecta leaving faster than the
            head that threw it.
        displaced_mass_bounds_kg: ``(lower, upper)`` of the interval
            ``displaced_mass_kg`` was drawn from [kg], or ``None`` when the
            caller has a point estimate and nothing else. Carried rather than
            recomputed so a launch can report how wide the mass it divided by
            actually was.
        contact_duration_s: Time the sole spent engaged with the bed [s].
        entry_speed_m_s: Head speed at the first sample [m/s].
        exit_speed_m_s: Head speed at the last sample [m/s].
        bed_relative_density: ``D_r`` of the bed the strike was made in, from
            :attr:`bunkershot3d.sand.SandState.relative_density`. Required
            rather than defaulted: with a lie-independent efficiency the model
            launches the ball *faster* out of a plugged lie than a firm one
            (issue #8704), so there is no safe default to fall back on.
        verdict: The solver's validity statement for the strike, which the
            launch verdict is combined with so carry can never read better
            than the shot behind it.
    """

    impulse_n_s: float
    displaced_mass_kg: float
    contact_duration_s: float
    entry_speed_m_s: float
    exit_speed_m_s: float
    bed_relative_density: float
    verdict: ValidityVerdict
    displaced_mass_bounds_kg: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        """Refuse a strike that was not measured, or that cannot have happened.

        Raises:
            ValueError: If any measurement is unusable, if the verdict is
                missing, or if the impulse and the mass together imply sand
                leaving faster than the head arrived. A launch computed from a
                defaulted impulse would be exactly the invented number this
                rewrite removed; a launch computed from an inadmissible pair
                would be the contradiction of issue #8659, shipped again.
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
        if (
            not math.isfinite(self.bed_relative_density)
            or not 0.0 <= self.bed_relative_density <= 1.0
        ):
            _refuse(
                "bed_relative_density",
                self.bed_relative_density,
                "a finite fraction in [0, 1] -- the lie sets how much of the "
                "delivered momentum reaches the ball (issue #8704)",
            )
        self._require_bounds()
        self._require_admissible_ejecta()

    def _require_bounds(self) -> None:
        """Refuse an interval that does not bracket the value drawn from it.

        Raises:
            ValueError: If the bounds are not an ordered, finite, positive
                pair containing :attr:`displaced_mass_kg`.
        """
        if self.displaced_mass_bounds_kg is None:
            return
        try:
            lower, upper = (float(value) for value in self.displaced_mass_bounds_kg)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "displaced_mass_bounds_kg must be a (lower, upper) pair, got "
                f"{self.displaced_mass_bounds_kg!r}"
            ) from error
        if not math.isfinite(lower) or not math.isfinite(upper) or lower <= 0.0:
            _refuse(
                "displaced_mass_bounds_kg",
                self.displaced_mass_bounds_kg,
                "a finite pair of positive masses",
            )
        if not lower <= self.displaced_mass_kg <= upper:
            raise ValueError(
                f"the reported mass {self.displaced_mass_kg:.6g} kg lies outside "
                f"the interval [{lower:.6g}, {upper:.6g}] kg it is said to have "
                "come from; a point estimate outside its own band is not a "
                "narrower claim, it is a different one"
            )

    def _require_admissible_ejecta(self) -> None:
        """Refuse an impulse and a mass that cannot both describe one strike.

        The one relation in this model that is not a convention: momentum is
        carried by mass moving, the sand was set moving by the head, and
        nothing the head threw can outrun it. Both quantities meet here for
        the first time -- the impulse from the solver, the mass from the
        metrics layer -- so this is the only place the pair can be checked
        before something divides one by the other.

        A plain ``raise`` and never an ``assert``: ``python -O`` strips
        assertions and ``DBC_LEVEL=off`` disables contracts, and a physical
        impossibility that evaporates under an optimisation flag is worse than
        no check at all. Not a clamp either -- see
        :data:`INADMISSIBLE_EJECTA_REFUSAL`.

        Raises:
            ValueError: If the implied mean ejecta speed exceeds the head's
                entry speed.
        """
        entry = float(self.entry_speed_m_s)
        if entry <= 0.0 or self.impulse_n_s <= 0.0:
            return
        ejecta = self.mean_ejecta_speed_m_s
        if ejecta > entry:
            raise ValueError(
                INADMISSIBLE_EJECTA_REFUSAL.format(
                    ejecta=ejecta, impulse=self.impulse_n_s, entry=entry
                )
            )

    @property
    def mean_ejecta_speed_m_s(self) -> float:
        """``J / m``: the mean speed of the sand that was accelerated [m/s].

        Never above :attr:`entry_speed_m_s` for a delivery that could be
        built: :meth:`_require_admissible_ejecta` refuses the pair otherwise.
        """
        return self.impulse_n_s / self.displaced_mass_kg

    @property
    def admissible_mass_floor_kg(self) -> float:
        """Smallest mass the delivered impulse can be carried by [kg].

        ``J / v_entry``. Zero when there is no entry speed to divide by, which
        is the same condition :meth:`_require_admissible_ejecta` declines to
        judge.
        """
        if self.entry_speed_m_s <= 0.0:
            return 0.0
        return self.impulse_n_s / float(self.entry_speed_m_s)


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
        transfer_efficiency: The bed-dependent ``eta`` the partition actually
            used, so the number is visible rather than implied (issue #8704).
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
    transfer_efficiency: float


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
        transfer_efficiency: The bed-dependent ``eta`` the partition used, so a
            caller can see how much of the answer the lie decided (#8704).
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
    transfer_efficiency: float
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
    efficiency = transfer.efficiency_for(delivery.bed_relative_density)
    ball_impulse = (
        efficiency
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
        transfer_efficiency=efficiency,
    )


def _launch_reasons(delivery: SandDelivery) -> tuple[str, ...]:
    """Return the findings the launch model raises about one strike.

    The supersonic-ejecta diagnostic issue #8657 added is gone from here,
    because the condition it reported is now refused outright in
    :meth:`SandDelivery._require_admissible_ejecta` rather than carried on a
    result. What is left is the *interval* diagnostic: a delivery whose mass
    band reaches below the momentum floor is still admissible at the value it
    reports, but half its band is not, and that asymmetry is worth saying.

    Args:
        delivery: The measured strike.

    Returns:
        The uncalibrated-transfer statements, plus the interval-floor
        diagnostic when the mass band's lower edge is inadmissible.
    """
    reasons = [BALL_LAUNCH_UNCALIBRATED_REASON, BED_PACKING_DEPENDENCE_REASON]
    bounds = delivery.displaced_mass_bounds_kg
    floor = delivery.admissible_mass_floor_kg
    if bounds is not None and floor > 0.0 and float(bounds[0]) < floor:
        lower = float(bounds[0])
        reasons.append(
            MASS_INTERVAL_FLOOR_REASON.format(
                impulse=delivery.impulse_n_s,
                lower=lower,
                implied=delivery.impulse_n_s / lower,
                entry=delivery.entry_speed_m_s,
                floor=floor,
            )
        )
    return tuple(reasons)


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
        transfer_efficiency=splash.transfer_efficiency,
        delivered_impulse_n_s=splash.delivered_impulse_n_s,
        ball_impulse_n_s=splash.ball_impulse_n_s,
        verdict=launch_verdict(delivery),
        provenance=momentum_transfer_provenance(transfer),
    )
