"""Divot geometry and the dig-versus-skid discriminator (issue #8614, W7).

Definitions, all measured on the **sole reference point** of
:class:`~bunkershot3d.metrics.trace.HeadModel` and all in SI:

============================ ============================================================
Quantity                     Definition
============================ ============================================================
``depth_m(s)``               Sole depth below the undisturbed sand surface, positive
                             downward, as a function of along-track travel ``s``.
Entry point                  The first downward crossing ``depth = 0``, linearly
                             interpolated between the bracketing samples.
Entry distance behind ball   ``-s_entry`` with ``s`` measured from the ball along the
                             travel axis. Wivou et al. (2016) report 25-150 mm.
Maximum depth                ``max(depth)`` over the submerged window. Resolved to the
                             sample spacing; no sub-sample peak fit is applied.
Exit point                   The first upward crossing ``depth = 0`` after entry.
Divot length                 ``s_exit - s_entry`` [m].
Divot section area           ``integral of depth ds`` between entry and exit [m^2].
Divot volume                 ``section area * width`` [m^3] -- a prismatic model.
Divot mass                   ``volume * sand bulk density`` [kg]. The sand under
                             the sole path, and **not** the sand the strike
                             accelerated -- see the section below.
Accelerated sand mass        The mass that shared the delivered momentum [kg],
                             reported as an interval by
                             :class:`AcceleratedSandMass`.
Entry descent speed          ``-v_z`` of the sole at the first submerged sample;
                             positive downward [m/s].
Exit climb speed             ``+v_z`` of the sole at the last submerged sample;
                             positive upward [m/s].
Descent-return ratio         exit climb speed / entry descent speed.
Incoming path slope          Chord slope ``-dz/d(travel)`` across the last free-flight
                             step; equals ``tan|attack angle|`` at delivery. Reported
                             as context; **nothing is divided by it**.
============================ ============================================================

**The discriminator, and the observable that expresses it.** Dig and skid are
claims about what the sand does with the head's descent. A digging head hands
its descent to the sand and stays down: it is still being carried deeper when
the sand finally lets go, and it crawls out. A skidding head bottoms out early
and is handed its descent back: the sole planes, and the sand throws it out
about as fast as it came down. The observable is therefore the **vertical
restitution of the strike** -- the sole's upward speed as it leaves the sand
over its downward speed as it entered, both read off the same sole reference
point the divot is measured on, at the two ``depth = 0`` crossings that bound
the divot. Near 0 the sand kept the descent (DIG); near 1 it returned it
(SKID).

Two structural properties follow, and both are what the metric this replaced
lacked. The ratio has **no window parameter** -- its window is the divot, which
the trace defines -- so no threshold on it is a statement about where a window
was placed. And its scale is **absolute**: 0 and 1 mean something before any
calibration, where a slope ratio's numbers meant nothing without one.

**What this replaced, and why (issue #8703)**
---------------------------------------------

The previous discriminator was the *entry slope ratio*: the penetration slope
over the first 10 mm of travel after entry, divided by the delivered path
slope. Across the demo's 77-point sweep it returned ``MARGINAL`` at every
point, with ratios spanning only 0.9987 to 1.0000. The head really can deflect
-- ``simulate_shot`` integrates translation under the sand wrench and
prescribes only the rotation -- but at a 25 m/s delivery 10 mm of travel is
about **0.4 ms**, and a 0.3 kg head under an order-5 N.s total impulse cannot
bend measurably in 0.4 ms.

Resizing that window was measured rather than argued, over 48 design points and
six window sizes, and **refuted**: the spread opens (0.0015 at 10 mm to 0.28 at
half the divot) but the correlation with maximum sole depth is negative at
every informative window (-0.50 to -0.68), so the deepest-cutting designs came
out nearest the *skid* threshold. Normalising by the delivered slope divides
out the attack angle, which is the variable that decides the outcome, and a
resized window would have shipped a confidently **inverted** verdict. The
quantity was also pinned at 1 as the window vanished and at 0 at the exit
crossing, for every design.

**What the replacement measures, over the same design space**
-------------------------------------------------------------

Swept through the shipped ``WorkbenchModel`` at its default settings over 384
points -- marketed bounce 8/14/20/26 deg x sole width 16/20/24 mm x attack
-2/-6/-10/-14 deg x firm/fluffy/wet/plugged x 20 and 25 m/s:

=================================  ====================  =====================
quantity                           span over the sweep   r with max sole depth
=================================  ====================  =====================
entry slope ratio (removed)        0.0012                -0.64  (inverted)
descent-return ratio               0.6162                -0.97
=================================  ====================  =====================

The sign is the point: deeper is more dig. Holding the delivery fixed, the
design axes alone still move the ratio by 0.11 (at -2 deg of attack) to 0.18
(at -14 deg), against 0.0012 for the whole of the old signal.

**One caveat the verdict carries rather than hides.** Over that same workbench
sweep the ratio falls from 0.424 to 0.339 as marketed bounce rises from 8 to
26 deg at -14 deg of attack and a 24 mm sole -- more bounce reading as *more*
dig, which is the opposite of fitting practice. That is the model, not the
metric: maximum sole depth over the same span rises from 18.0 to 21.5 mm, so
the F0 solver itself puts the higher-bounce sole deeper in this shallow,
non-burying regime. Where the head does bury -- three shipped presets at
5.0/8.0/14.42 deg of marketed bounce, run through ``simulate_shot`` directly --
the ordering is the expected one: at -10 deg of attack the 5 deg sole buries
75 mm and returns 0.35 of its descent while the 14.42 deg sole buries 13 mm and
returns 0.51. Both measurements are reported;
:data:`DIG_SKID_BOUNCE_ORDERING_REASON` travels with every verdict so that
nobody reads one as a bounce recommendation.

**Still uncalibrated.** Separating is not the same as being measured. Nobody
has published a vertical restitution for a wedge sole leaving bunker sand, so
the DIG and SKID thresholds remain conventions, every
:class:`DigSkidResult` carries a :class:`DigSkidCalibration` saying so in the
same shape the solver's ``ValidityVerdict`` and the ball model's provenance
record use, and :meth:`DigSkidCalibration.require_calibrated` refuses.

**The force side of the same question.** Through the submerged window the
head's vertical momentum change must equal the sum of the sand impulse, the
gravity impulse and whatever the shaft and hands supply. All four are reported,
so the residual is visible rather than absorbed -- and they are reported over
the same sampled window the two speeds are read from, so the restitution and
the balance describe one strike. The flat-surface scene is the remaining
approximation here, and it is stated.

**Two masses, and why (issue #8659)**
--------------------------------------

The prismatic divot mass was an honest approximation for as long as it was a
*reported metric*. Issue #8657 made it a **denominator** -- ball launch became
the delivered impulse divided by it -- and the arithmetic immediately produced
something impossible. At the workbench's nominal greenside shot the F0 solver
integrates 2.917 N.s out of a head arriving at 25.0 m/s, and the prism says
63.7 g of sand carried it: a mean ejecta speed of **45.8 m/s**. Sand cannot
leave faster than the thing that threw it.

The impulse is an integral of a force computed element by element. The prism
is a constant-width channel under the sole path, and a real splash throws
material the sole never passed over -- the bow wave ahead of the leading edge,
the heave above the original surface, and the divot's own sloping walls. So
the mass was the wrong quantity, and this module now reports both:

* :attr:`DivotMetrics.mass_kg` is the prism, unchanged, with the same
  provenance and the same warning it always carried; and
* :attr:`DivotMetrics.accelerated_mass` is the mass that shared the delivered
  momentum, as an **interval** -- because the in-plane part of the correction
  was measured against the F1 MPM tier and the out-of-plane part cannot be,
  plane strain having no out-of-plane extent.

Nothing here is a clamp. The ejecta speed is not capped; the mass is bigger
because a different, better-founded mass is being reported. And nothing here
is a validation: two uncalibrated tiers agreeing is two uncalibrated tiers
agreeing, and :data:`ACCELERATED_MASS_CONSISTENCY_REASON` says so in the words
a verdict carries. What the comparison did do is falsify -- the prism was
inadmissible against the head's own entry speed, and the interval is not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.shared.python.core.contracts import ensure, require

from ..exceptions import BunkerShot3DValueError
from .enums import DigSkidVerdict
from .trace import STANDARD_GRAVITY_MPS2, HeadModel, StrikeScene, StrikeTrace

__all__ = [
    "ACCELERATED_MASS_CONSISTENCY_REASON",
    "ACCELERATED_MASS_LATERAL_REASON",
    "DEFAULT_DIG_DESCENT_RETURN",
    "DEFAULT_SKID_DESCENT_RETURN",
    "DIG_SKID_BOUNCE_ORDERING_REASON",
    "DIG_SKID_COARSE_WINDOW_REASON",
    "DIG_SKID_UNCALIBRATED_REASON",
    "F1_ENTRAINMENT_FACTOR_BOUNDS",
    "MIN_RESOLVED_SUBMERGED_SAMPLES",
    "MIN_SUBMERGED_SAMPLES",
    "STANDARD_GRAVITY_MPS2",
    "AcceleratedSandMass",
    "DigSkidCalibration",
    "DigSkidResult",
    "DivotMetrics",
    "SoleDepthProfile",
    "StrikeInterval",
    "dig_vs_skid",
    "divot_metrics",
    "sole_depth_profile",
    "submerged_interval",
]

#: Descent-return ratio at or below which the strike is called a dig: the sand
#: kept at least half of the descent the sole arrived with, so the head is left
#: lower than it was handed back. Half is a round number, not a measurement.
DEFAULT_DIG_DESCENT_RETURN = 0.50

#: Descent-return ratio at or above which the strike is called a skid: the sand
#: returned at least four fifths of the descent, so the sole planed and was
#: thrown out about as fast as it came down. A convention, not a measurement.
DEFAULT_SKID_DESCENT_RETURN = 0.80

#: Submerged samples below which the entry and exit speeds are not two separate
#: measurements at all: both are centred differences, so with fewer than three
#: samples they overlap and the ratio is an arithmetic identity. Refused.
MIN_SUBMERGED_SAMPLES = 3

#: Submerged samples below which the two speeds are answerable but resolution
#: limited -- each is a centred difference over a window this short. Reported,
#: not refused.
MIN_RESOLVED_SUBMERGED_SAMPLES = 8

#: Share of the sole's speed below which a measured descent is differencing
#: noise rather than a delivered blow. A numerical floor, not a physical
#: threshold: at 25 m/s it is 25 micrometres per second.
_MIN_ENTRY_DESCENT_SPEED_FRACTION = 1e-6

DIG_SKID_UNCALIBRATED_REASON = (
    "the dig-versus-skid verdict is a convention on an uncalibrated ratio and "
    "must not be quoted as a finding. The descent-return ratio separates the "
    "design space -- it spans 0.34-0.95 over the workbench's 384-point sweep "
    "and ranks against maximum sole depth at -0.97 -- but no vertical "
    "restitution has been published for a wedge sole leaving bunker sand, so "
    "the DIG and SKID thresholds it is compared against are round numbers "
    "rather than measurements (issue #8703)."
)
"""Why a dig-versus-skid verdict is never a finding, however clean the trace."""

DIG_SKID_BOUNCE_ORDERING_REASON = (
    "the F0 model's response to marketed bounce disagrees with fitting "
    "practice in the shallow, non-burying regime: over the workbench sweep the "
    "ratio falls from 0.424 to 0.339 as marketed bounce rises from 8 to 26 deg "
    "at -14 deg of attack, and maximum sole depth rises from 18.0 to 21.5 mm "
    "over the same span, so the solver -- not this metric -- puts more bounce "
    "deeper there. Where the head buries the ordering is the expected one. "
    "Read the verdict as what the model did, never as a bounce recommendation "
    "(issue #8703)."
)
"""The one ordering a designer must not take at face value from this verdict."""

DIG_SKID_COARSE_WINDOW_REASON = (
    "the entry and exit speeds are centred differences over a submerged window "
    "of only {samples} samples, below the {minimum} this metric treats as "
    "resolved, so the ratio carries the sample spacing as well as the strike"
)
"""Template for the diagnostic that fires on a barely-resolved divot."""

F1_ENTRAINMENT_FACTOR_BOUNDS: tuple[float, float] = (2.84, 3.90)
"""How much more sand is moving than the swept prism accounts for, in plane.

Read off the **F1 plane-strain MPM tier**, not off a bunker. Ten whole-shot F1
marches (:func:`bunkershot3d.solvers.mpm.wholeshot.simulate_f1_shot`,
``dx = 4 mm``, 12 ms, firm bed 80 mm deep) were run at the workbench's own
designs and deliveries -- attack -4/-8/-12 deg, marketed bounce 8/20 deg, sole
16/20/24 mm, firm/fluffy/plugged beds, 20 and 25 m/s -- and each march was
reduced **twice from its own record**:

* by the prismatic rule :func:`divot_metrics` applies -- the same
  ``integral of depth ds`` along its own sole path, at the same declared width
  and bulk density; and
* by asking the particles what mass was actually in motion.

The second reading is momentum-and-energy consistent and threshold-free: a bed
carrying momentum ``P`` and kinetic energy ``T`` has exactly one mass
``P^2 / (2 T)`` that would carry both while moving as a single lump, and by
Cauchy-Schwarz that mass is a **lower bound** on the mass with any motion at
all. Counting particles above a speed threshold instead gives 8-12x the prism
rather than 3x, so this choice is the conservative one by a wide margin.

Both readings come from one march, so the constitutive difference between the
tiers divides out and what is left is what the prism misses: the bow wave
ahead of the leading edge and the heave above the original surface.

**Where the ratio is evaluated matters, and it is stated rather than chosen
quietly.** F1's prism keeps growing along the sole path while the moving mass
saturates, so the ratio *falls* through a march -- above 10 early, near 2 at
the end of a truncated 12 ms window. Each ratio above is therefore taken at
**matched prism**: the instant F1's own accumulated prism equals the prism F0
reported for the same design, so both tiers are asked about the same amount of
swept sand. Over the nine designs whose marches reached that point the ratio
runs 2.845 (plugged bed) to 3.898 (16 mm sole), geometric mean 3.30 -- and the
bounds above are that min and max. The one design that did not reach matched
prism in 12 ms (-4 deg of attack, whose divot is 308 mm long) is excluded
rather than extrapolated.

The spread is real and it is mostly one effect: the designs with the smallest
prisms sit higher on the declining ratio curve. Holding that aside, the sand
condition moves it (firm 3.54, fluffy 2.92, plugged 2.85) and the delivery
speed barely does at all (3.54 at 25 m/s against 3.51 at 20 m/s).

Plane strain has no out-of-plane extent, so this factor is **blind** to the
divot's sloping walls; :class:`AcceleratedSandMass` carries those separately,
and that blindness is why the correction is an interval."""

ACCELERATED_MASS_CONSISTENCY_REASON = (
    "the accelerated sand mass is a **consistency** correction between two "
    "uncalibrated models and not a measurement. Its in-plane factor of "
    f"{F1_ENTRAINMENT_FACTOR_BOUNDS[0]:.3g}-"
    f"{F1_ENTRAINMENT_FACTOR_BOUNDS[1]:.3g} was read off the "
    "F1 MPM tier, which is BEYOND_VALIDATION, whose published-speed ceiling is "
    "1.44 m/s against a 25 m/s greenside delivery and whose NASA-STD-7009B "
    "validation is 0 of 4; no ejecta mass has ever been measured on a real "
    "bunker shot (issue #8616). Two uncalibrated tiers agreeing is two "
    "uncalibrated tiers agreeing. What the comparison can do is falsify, and "
    "it did: the prismatic mass was inadmissible against the head's own entry "
    "speed and the corrected interval is not (issue #8659)"
)
"""Why an accelerated mass is never quotable as a measured ejecta mass."""

ACCELERATED_MASS_LATERAL_REASON = (
    "the upper edge of the accelerated mass widens the divot's walls to the "
    "bed's own friction angle, on the argument that a trench cut in a "
    "cohesionless sand cannot stand steeper than the material it is cut in. "
    "The angle is the bed's, so nothing new is fitted, but the shape is a "
    "model: F1 is plane strain and cannot see out of the plane, so no tier in "
    "this package has measured what the walls actually do (issue #8659)"
)
"""Why the out-of-plane half of the interval is modelled rather than measured."""


@dataclass(frozen=True)
class SoleDepthProfile:
    """The sole's depth-versus-travel trace -- the raw curve the metrics reduce.

    Attributes:
        travel_m: ``(T,)`` signed along-track distance from the ball [m];
            negative behind it.
        depth_m: ``(T,)`` depth below the undisturbed surface [m], positive
            downward.
        position_m: ``(T, 3)`` world path of the sole reference point [m].
        velocity_mps: ``(T, 3)`` world velocity of the sole reference point.
    """

    travel_m: np.ndarray
    depth_m: np.ndarray
    position_m: np.ndarray
    velocity_mps: np.ndarray

    def depth_at_travel_m(self, travel_m: float) -> float:
        """Return the depth at an along-track station by linear interpolation.

        Args:
            travel_m: Along-track station [m], on the same origin as
                :attr:`travel_m`.

        Returns:
            Depth [m].

        Raises:
            ValueError: If the travel coordinate ever decreases, so the depth at
                a station would be ambiguous.
        """
        if np.any(np.diff(self.travel_m) < 0.0):
            raise ValueError(
                "depth_at_travel_m needs a non-decreasing travel coordinate; "
                "the head does not move forward monotonically in this trace"
            )
        return float(np.interp(travel_m, self.travel_m, self.depth_m))


@dataclass(frozen=True)
class StrikeInterval:
    """The submerged window, with sub-sample entry and exit crossings.

    Attributes:
        entry_index: First sample with the sole below the surface.
        exit_index: Last sample with the sole below the surface.
        entry_time_s: Interpolated time of the downward ``depth = 0`` crossing.
        exit_time_s: Interpolated time of the upward crossing.
        entry_point_m: ``(3,)`` interpolated world entry point.
        exit_point_m: ``(3,)`` interpolated world exit point.
        entry_travel_m: Along-track station of the entry point [m].
        exit_travel_m: Along-track station of the exit point [m].
    """

    entry_index: int
    exit_index: int
    entry_time_s: float
    exit_time_s: float
    entry_point_m: np.ndarray
    exit_point_m: np.ndarray
    entry_travel_m: float
    exit_travel_m: float

    @property
    def duration_s(self) -> float:
        """Time the sole spent below the undisturbed surface [s]."""
        return self.exit_time_s - self.entry_time_s

    @property
    def sample_slice(self) -> slice:
        """Slice selecting the submerged samples."""
        return slice(self.entry_index, self.exit_index + 1)


def sole_depth_profile(
    trace: StrikeTrace, head: HeadModel, scene: StrikeScene
) -> SoleDepthProfile:
    """Return the sole's depth-versus-travel trace.

    Args:
        trace: Strike trace.
        head: Head the trace was recorded for.
        scene: Sand surface, ball and travel axis.

    Returns:
        The profile, one entry per trace sample.
    """
    position = trace.point_path_m(head.sole_reference_body_m)
    velocity = trace.point_velocity_mps(head.sole_reference_body_m)
    return SoleDepthProfile(
        travel_m=scene.along_travel_m(position),
        depth_m=scene.depth_m(position),
        position_m=position,
        velocity_mps=velocity,
    )


def _crossing_fraction(before: float, after: float) -> float:
    """Return the interpolation fraction where a depth trace crosses zero.

    Args:
        before: Depth at the earlier sample [m].
        after: Depth at the later sample [m].

    Returns:
        Fraction in ``[0, 1]`` from the earlier sample to the crossing.
    """
    span = after - before
    if span == 0.0:
        return 0.0
    return float(np.clip(-before / span, 0.0, 1.0))


def _lerp(values: np.ndarray, index: int, fraction: float) -> np.ndarray:
    """Linearly interpolate between rows ``index`` and ``index + 1``."""
    return values[index] + fraction * (values[index + 1] - values[index])


def submerged_interval(
    trace: StrikeTrace, head: HeadModel, scene: StrikeScene
) -> StrikeInterval:
    """Return the window over which the sole is below the undisturbed surface.

    Args:
        trace: Strike trace.
        head: Head the trace was recorded for.
        scene: Sand surface, ball and travel axis.

    Returns:
        The interval, with entry and exit interpolated to the ``depth = 0``
        crossings rather than snapped to samples.

    Raises:
        ValueError: If the sole never goes below the surface (there is no
            divot to measure), if the trace starts already submerged (the entry
            happened before the record began), or if it ends still submerged
            (the exit is not in the record). Guessing any of those would invent
            a divot boundary.
    """
    return _interval_from_profile(sole_depth_profile(trace, head, scene), trace, scene)


def _interval_from_profile(
    profile: SoleDepthProfile, trace: StrikeTrace, scene: StrikeScene
) -> StrikeInterval:
    """Return the submerged window of an already-computed depth profile.

    Args:
        profile: Sole depth profile.
        trace: Strike trace the profile came from, for its sample times.
        scene: Scene the profile was measured against.

    Returns:
        The interval.

    Raises:
        ValueError: See :func:`submerged_interval`.
    """
    depth = profile.depth_m
    submerged = depth > 0.0
    if not np.any(submerged):
        raise ValueError(
            "the sole never goes below the sand surface in this trace, so there "
            "is no divot; check the scene's sand_surface_height_m"
        )
    first = int(np.argmax(submerged))
    last = int(len(submerged) - 1 - np.argmax(submerged[::-1]))
    if first == 0:
        raise ValueError(
            "the trace starts with the sole already below the surface; the entry "
            "point is outside the record, so entry distance cannot be measured"
        )
    if last == len(submerged) - 1:
        raise ValueError(
            "the trace ends with the sole still below the surface; the exit point "
            "is outside the record, so divot length cannot be measured"
        )
    entry_fraction = _crossing_fraction(depth[first - 1], depth[first])
    exit_fraction = _crossing_fraction(depth[last], depth[last + 1])
    entry_point = _lerp(profile.position_m, first - 1, entry_fraction)
    exit_point = _lerp(profile.position_m, last, exit_fraction)
    return StrikeInterval(
        entry_index=first,
        exit_index=last,
        entry_time_s=float(_lerp(trace.time_s, first - 1, entry_fraction)),
        exit_time_s=float(_lerp(trace.time_s, last, exit_fraction)),
        entry_point_m=entry_point,
        exit_point_m=exit_point,
        entry_travel_m=float(scene.along_travel_m(entry_point)),
        exit_travel_m=float(scene.along_travel_m(exit_point)),
    )


@dataclass(frozen=True, slots=True)
class AcceleratedSandMass:
    """The sand one strike set in motion, as an **interval** and not a point.

    Why an interval. The swept prism is not the mass that shared the delivered
    momentum, and issue #8659 is the arithmetic that proves it: at the nominal
    greenside shot the solver's 2.917 N.s over the prism's 63.7 g implies sand
    leaving at 45.8 m/s from a 25.0 m/s head. Something has to be bigger, and
    it is the mass.

    Two separate things are missing from the prism and only one of them has
    been looked at by a solver. **In plane**, the head throws a bow wave ahead
    of its leading edge and heaves material above the original surface, and
    the F1 MPM tier resolves that -- see
    :data:`F1_ENTRAINMENT_FACTOR_BOUNDS` for what it measured and under what
    licence. **Out of plane**, the divot's walls slope away from the sole and
    a plane-strain tier cannot see them at all, so that half is a stated model
    on the bed's own friction angle rather than a measurement.

    The interval is built so that each edge says which of those it rests on:

    * :attr:`lower_kg` is the prism scaled by the *smallest* in-plane factor
      the F1 sweep produced and no lateral spread whatsoever -- the
      plane-strain reading, taken literally.
    * :attr:`upper_kg` is the prism scaled by the largest in-plane factor and
      by the lateral widening a wall at the bed's friction angle implies.
    * :attr:`central_kg` is their geometric mean, because the two edges are
      multiplicative factors rather than additive offsets. A convention, and
      the only number here that is neither measured nor derived.

    None of this is calibrated against a bunker; nothing in this package is.
    :data:`ACCELERATED_MASS_CONSISTENCY_REASON` says so in the words a
    verdict carries.

    Attributes:
        prismatic_kg: The swept-prism mass this was formed from [kg], which
            is :attr:`DivotMetrics.mass_kg` unchanged.
        entrainment_lower: Smallest in-plane factor of the F1 sweep.
        entrainment_upper: Largest in-plane factor of the F1 sweep.
        lateral_factor: Out-of-plane widening, ``>= 1``, from the wall model.
        wall_angle_deg: The friction angle the walls were laid back at.
    """

    prismatic_kg: float
    entrainment_lower: float
    entrainment_upper: float
    lateral_factor: float
    wall_angle_deg: float

    def __post_init__(self) -> None:
        """Refuse an interval that is not one.

        Raises:
            ValueError: If the prism is not positive, if either factor is not
                at least one, if the factors are out of order, or if the
                lateral widening is below one. A plain ``raise`` and not a
                contract: ball launch divides by the number this object
                produces, and ``python -O`` strips assertions.
        """
        if not np.isfinite(self.prismatic_kg) or self.prismatic_kg <= 0.0:
            raise ValueError(
                "the prismatic divot mass must be positive and finite, got "
                f"{self.prismatic_kg!r}"
            )
        for name in ("entrainment_lower", "entrainment_upper", "lateral_factor"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 1.0:
                raise ValueError(
                    f"{name} must be a finite factor of at least 1 -- the sand "
                    "outside the swept prism can only add mass, never remove "
                    f"it -- got {value!r}"
                )
        if self.entrainment_upper < self.entrainment_lower:
            raise ValueError(
                "the entrainment bounds are out of order: "
                f"{self.entrainment_lower!r} above {self.entrainment_upper!r}"
            )
        if not np.isfinite(self.wall_angle_deg) or not 0.0 < self.wall_angle_deg < 90.0:
            raise ValueError(
                "the wall angle must lie in (0, 90) degrees, got "
                f"{self.wall_angle_deg!r}"
            )

    @property
    def lower_kg(self) -> float:
        """Smallest admissible accelerated mass [kg]: F1 in plane, no walls."""
        return self.prismatic_kg * self.entrainment_lower

    @property
    def upper_kg(self) -> float:
        """Largest [kg]: the F1 sweep's widest factor, walls laid back."""
        return self.prismatic_kg * self.entrainment_upper * self.lateral_factor

    @property
    def central_kg(self) -> float:
        """Geometric mean of the two edges [kg]. A convention, not a value."""
        return float(np.sqrt(self.lower_kg * self.upper_kg))

    @property
    def bounds_kg(self) -> tuple[float, float]:
        """The interval, ``(lower, upper)`` [kg]."""
        return (self.lower_kg, self.upper_kg)

    def summary(self) -> str:
        """A line fit for a report, carrying the interval and not the point."""
        return (
            f"{self.central_kg * 1e3:.4g} g "
            f"[{self.lower_kg * 1e3:.4g}-{self.upper_kg * 1e3:.4g} g], "
            f"{self.central_kg / self.prismatic_kg:.3g}x the swept prism "
            f"({self.prismatic_kg * 1e3:.4g} g)"
        )


def lateral_spread_factor(
    section_area_m2: float,
    depth_squared_integral_m3: float,
    *,
    width_m: float,
    wall_angle_deg: float,
) -> float:
    """How much the divot's sloping walls widen the swept prism.

    A trench of bottom width ``w`` and depth ``d(s)`` whose walls lie back at
    the bed's friction angle ``phi`` measured from the horizontal has section
    ``w d + d^2 cot(phi)``, so the whole volume is
    ``w * integral d ds + cot(phi) * integral d^2 ds`` and the widening is the
    ratio of that to the prism. The argument for the angle is that a trench
    cut in a cohesionless sand cannot stand steeper than the material it is
    cut in; the argument for the *shape* is only that it is the simplest one
    that has the right limits, and :data:`ACCELERATED_MASS_LATERAL_REASON`
    says so.

    Args:
        section_area_m2: ``integral of depth ds`` over the divot [m^2].
        depth_squared_integral_m3: ``integral of depth^2 ds`` over the same
            window [m^3].
        width_m: Sole width in contact [m].
        wall_angle_deg: Friction angle of the bed [deg], from the sand state.

    Returns:
        The widening factor, at least 1.

    Raises:
        ValueError: If any argument is out of range. A ``raise`` for the same
            reason :class:`AcceleratedSandMass` uses one.
    """
    if not np.isfinite(width_m) or width_m <= 0.0:
        raise ValueError(f"width_m must be positive and finite, got {width_m}")
    if not np.isfinite(wall_angle_deg) or not 0.0 < wall_angle_deg < 90.0:
        raise ValueError(
            f"wall_angle_deg must lie in (0, 90) degrees, got {wall_angle_deg}"
        )
    for name, value in (
        ("section_area_m2", section_area_m2),
        ("depth_squared_integral_m3", depth_squared_integral_m3),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative, got {value}")
    if section_area_m2 <= 0.0:
        return 1.0
    cotangent = 1.0 / np.tan(np.radians(float(wall_angle_deg)))
    return 1.0 + float(cotangent) * float(depth_squared_integral_m3) / (
        float(width_m) * float(section_area_m2)
    )


@dataclass(frozen=True)
class DivotMetrics:
    """Divot geometry, measured from the sole path.

    Attributes:
        entry_point_m: ``(3,)`` world point where the sole crosses into the sand.
        entry_distance_behind_ball_m: Positive when the sole enters behind the
            ball, which is the reporting convention of the delivery data.
        max_depth_m: Deepest point of the sole below the undisturbed surface.
        max_depth_travel_m: Along-track station of the deepest point [m].
        max_depth_behind_ball_m: Same station expressed as a distance behind the
            ball; negative once the deepest point is past the ball.
        exit_point_m: ``(3,)`` world point where the sole leaves the sand.
        exit_distance_past_ball_m: Positive when the sole exits past the ball.
        length_m: Along-track distance from entry to exit.
        section_area_m2: ``integral of depth ds`` over the divot.
        depth_squared_integral_m3: ``integral of depth^2 ds`` over the same
            window. Carried because the divot's *walls* scale with it, not
            with the section: see :func:`lateral_spread_factor`.
        width_m: Effective cutting width used for the prismatic volume.
        volume_m3: ``section_area_m2 * width_m``.
        mass_kg: ``volume_m3 * bulk_density_kg_m3`` -- the **swept prism**,
            unchanged and still the sand directly under the sole path. It is
            reported for what it is and is no longer what ball launch divides
            by; :attr:`accelerated_mass` is (issue #8659).
        accelerated_mass: The mass that shared the delivered momentum, as an
            interval. ``None`` only when the caller declined to state a bed
            friction angle, which no shipped path does.
        bulk_density_kg_m3: Sand bulk density used for the mass.
        submerged_duration_s: Time between the entry and exit crossings.
    """

    entry_point_m: np.ndarray
    entry_distance_behind_ball_m: float
    max_depth_m: float
    max_depth_travel_m: float
    max_depth_behind_ball_m: float
    exit_point_m: np.ndarray
    exit_distance_past_ball_m: float
    length_m: float
    section_area_m2: float
    depth_squared_integral_m3: float
    width_m: float
    volume_m3: float
    mass_kg: float
    bulk_density_kg_m3: float
    submerged_duration_s: float
    accelerated_mass: AcceleratedSandMass


def divot_metrics(
    trace: StrikeTrace,
    head: HeadModel,
    scene: StrikeScene,
    *,
    width_m: float,
    bulk_density_kg_m3: float,
    friction_angle_deg: float,
) -> DivotMetrics:
    """Measure the divot the strike cuts, and the mass the strike accelerated.

    Two masses come back and they are different quantities. The **prismatic**
    one, :attr:`DivotMetrics.mass_kg`, is unchanged: a constant-width channel
    whose depth follows the sole, counting only the sand directly beneath the
    sole path. That was always an approximation and was always stated as one.

    It stopped being an adequate approximation when it became a *denominator*.
    Issue #8657 made ball launch divide the delivered impulse by it, and issue
    #8659 is the contradiction that produced: 2.917 N.s over 63.7 g is sand
    leaving at 45.8 m/s from a head that arrived at 25.0 m/s. The prism does
    not count the bow wave, the heave, or the divot's sloping walls, so
    :attr:`DivotMetrics.accelerated_mass` is what ball launch divides by now,
    and it is an interval rather than a point. See
    :class:`AcceleratedSandMass` for how the interval is built and
    :data:`ACCELERATED_MASS_CONSISTENCY_REASON` for what may be said about it.

    Args:
        trace: Strike trace.
        head: Head the trace was recorded for.
        scene: Sand surface, ball and travel axis.
        width_m: Effective cutting width [m], normally the sole width in
            contact. Must be positive.
        bulk_density_kg_m3: Sand bulk density [kg/m^3]. The measured Covia
            Signature 500 bunker sand is 1550 kg/m^3 (research addendum).
        friction_angle_deg: Internal friction angle of the bed [deg], from
            :attr:`bunkershot3d.sand.SandState.friction_angle_deg`. Required
            rather than defaulted: it is the angle the divot's walls are laid
            back at, so a default would be an invented divot shape, and the
            bed already knows the number.

    Returns:
        The divot metrics.

    Raises:
        ValueError: If ``width_m``, ``bulk_density_kg_m3`` or
            ``friction_angle_deg`` is out of range, if the sole never enters or
            never leaves the sand, or if the head does not travel forward
            monotonically through the strike (the section integral would double
            back on itself).
    """
    if not np.isfinite(width_m) or width_m <= 0.0:
        raise ValueError(f"width_m must be positive and finite, got {width_m}")
    if not np.isfinite(bulk_density_kg_m3) or bulk_density_kg_m3 <= 0.0:
        raise ValueError(
            f"bulk_density_kg_m3 must be positive and finite, got {bulk_density_kg_m3}"
        )
    if not np.isfinite(friction_angle_deg) or not 0.0 < friction_angle_deg < 90.0:
        raise ValueError(
            f"friction_angle_deg must lie in (0, 90) degrees, got {friction_angle_deg}"
        )
    profile = sole_depth_profile(trace, head, scene)
    interval = _interval_from_profile(profile, trace, scene)
    window = interval.sample_slice
    travel = np.concatenate(
        ([interval.entry_travel_m], profile.travel_m[window], [interval.exit_travel_m])
    )
    depth = np.concatenate(([0.0], profile.depth_m[window], [0.0]))
    # Reversal is refused; a repeated station is not. The interpolated entry or
    # exit crossing can land exactly on the first or last submerged sample, and
    # the resulting zero-width interval contributes nothing to the integral.
    if np.any(np.diff(travel) < 0.0):
        raise ValueError(
            "the head must travel forward monotonically through the strike for a "
            "divot section area to be well defined; this trace reverses"
        )
    section_area_m2 = float(np.trapezoid(depth, travel))
    ensure(
        section_area_m2 >= 0.0,
        "divot section area must be non-negative -- depth is positive downward",
        value=section_area_m2,
    )
    depth_squared_integral_m3 = float(np.trapezoid(np.square(depth), travel))
    peak = int(np.argmax(profile.depth_m[window])) + interval.entry_index
    volume_m3 = section_area_m2 * width_m
    lower, upper = F1_ENTRAINMENT_FACTOR_BOUNDS
    accelerated = AcceleratedSandMass(
        prismatic_kg=volume_m3 * float(bulk_density_kg_m3),
        entrainment_lower=lower,
        entrainment_upper=upper,
        lateral_factor=lateral_spread_factor(
            section_area_m2,
            depth_squared_integral_m3,
            width_m=float(width_m),
            wall_angle_deg=float(friction_angle_deg),
        ),
        wall_angle_deg=float(friction_angle_deg),
    )
    return DivotMetrics(
        entry_point_m=interval.entry_point_m,
        entry_distance_behind_ball_m=-interval.entry_travel_m,
        max_depth_m=float(profile.depth_m[peak]),
        max_depth_travel_m=float(profile.travel_m[peak]),
        max_depth_behind_ball_m=-float(profile.travel_m[peak]),
        exit_point_m=interval.exit_point_m,
        exit_distance_past_ball_m=interval.exit_travel_m,
        length_m=interval.exit_travel_m - interval.entry_travel_m,
        section_area_m2=section_area_m2,
        depth_squared_integral_m3=depth_squared_integral_m3,
        width_m=float(width_m),
        volume_m3=volume_m3,
        mass_kg=volume_m3 * float(bulk_density_kg_m3),
        bulk_density_kg_m3=float(bulk_density_kg_m3),
        submerged_duration_s=interval.duration_s,
        accelerated_mass=accelerated,
    )


@dataclass(frozen=True)
class DigSkidCalibration:
    """Whether a dig-versus-skid verdict may be quoted, and why not.

    Carried on every :class:`DigSkidResult` so the statement travels with the
    number rather than living in report prose, in the same shape as
    :class:`bunkershot3d.solvers.envelope.ValidityVerdict` and
    :class:`bunkershot3d.sand.provenance.SandProvenance`.

    Attributes:
        calibrated: ``False``, unconditionally, until somebody publishes a
            vertical restitution for a wedge sole leaving bunker sand for the
            thresholds to be measured against (issue #8703).
        dig_descent_return: The DIG threshold the verdict was formed against.
        skid_descent_return: The SKID threshold the verdict was formed against.
        submerged_samples: Samples the sole spent below the surface, which is
            the resolution both speeds were differenced at. Below
            :data:`MIN_RESOLVED_SUBMERGED_SAMPLES` the ratio carries the
            sample spacing as well as the strike.
        reasons: Everything wrong with quoting this verdict, most general
            first.
    """

    calibrated: bool
    dig_descent_return: float
    skid_descent_return: float
    submerged_samples: int
    reasons: tuple[str, ...]

    def measured_constants(self) -> tuple[str, ...]:
        """Names of every threshold measured on a real bunker shot.

        Empty, and required to stay empty: mirrors
        :meth:`bunkershot3d.solvers.MaterialResponse.measured_constants` and
        :meth:`bunkershot3d.ball.splash.BallLaunchResult.measured_constants`.

        Returns:
            An empty tuple.
        """
        return ()

    def require_calibrated(self) -> None:
        """Refuse to let a caller treat the verdict as a finding.

        A plain ``raise`` rather than a contract: ``python -O`` strips
        assertions and ``DBC_LEVEL=off`` disables contracts, and an honesty
        guard that evaporates under an optimisation flag is not a guard.

        Raises:
            BunkerShot3DValueError: Always, while :attr:`calibrated` is False.
        """
        if not self.calibrated:
            raise BunkerShot3DValueError(
                "the dig-versus-skid verdict is not calibrated and may not be "
                "quoted as a finding: " + " ".join(self.reasons)
            )

    def summary(self) -> str:
        """Return the statement a report shows beside the verdict.

        Returns:
            One line naming the calibration state, then one line per reason.
        """
        state = "calibrated" if self.calibrated else "UNCALIBRATED"
        head = (
            f"dig-versus-skid: {state} "
            f"(dig at or below {self.dig_descent_return:.2f}, skid at or above "
            f"{self.skid_descent_return:.2f}, measured over "
            f"{self.submerged_samples} submerged samples)"
        )
        return "\n".join([head, *(f"  - {reason}" for reason in self.reasons)])


@dataclass(frozen=True)
class DigSkidResult:
    """The dig-versus-skid discriminator and the vertical impulse balance.

    Attributes:
        verdict: DIG, SKID or MARGINAL. **Uncalibrated**: read
            :attr:`calibration` before quoting it (issue #8703).
        calibration: Whether the verdict may be quoted, and why not.
        entry_descent_speed_mps: How fast the sole was going *down* at the
            first submerged sample; positive downward.
        exit_climb_speed_mps: How fast it is going *up* at the last submerged
            sample; positive upward, and negative if it is still descending
            there.
        descent_return_ratio: ``exit_climb_speed_mps /
            entry_descent_speed_mps`` -- the share of its descent the sand
            handed back. Near 0 the sand kept it (dig); near 1 it returned it
            (skid).
        incoming_path_slope: ``tan|attack angle|`` from the chord across the
            last free-flight step; dimensionless. Context only: the ratio is
            **not** referenced to it, because normalising by the delivered
            slope is what made the metric this replaced saturate.
        entry_attack_angle_rad: ``-arctan(incoming_path_slope)``; negative for a
            descending blow, matching the -2 to -12 deg delivery convention.
        vertical_sand_impulse_Ns: ``integral of F_z dt`` over the submerged
            window; positive is upward, i.e. the sand holding the sole up.
        gravity_impulse_Ns: ``-m g * submerged duration``; always negative.
        measured_vertical_momentum_change_Ns: ``m * (v_z_exit - v_z_entry)`` of
            the head centre of mass.
        constraint_vertical_impulse_Ns: The residual -- what the shaft, hands
            and any unmodelled contact supplied. Reported rather than absorbed.
            It, and the gravity term, are the two things besides the sand that
            moved the head between the two speeds the ratio is built from, so
            they are reported beside it rather than corrected out of it.
    """

    verdict: DigSkidVerdict
    calibration: DigSkidCalibration
    entry_descent_speed_mps: float
    exit_climb_speed_mps: float
    descent_return_ratio: float
    incoming_path_slope: float
    entry_attack_angle_rad: float
    vertical_sand_impulse_Ns: float
    gravity_impulse_Ns: float
    measured_vertical_momentum_change_Ns: float
    constraint_vertical_impulse_Ns: float


def _incoming_path_slope(
    profile: SoleDepthProfile, scene: StrikeScene, entry_index: int
) -> float:
    """Return the chord slope of the last free-flight step before entry.

    Reported as context beside the verdict -- it is the delivered attack angle
    the shot was set up with -- and **nothing is divided by it**. Normalising
    the discriminant by this slope is what made the metric of issue #8703
    saturate, so it is kept as a reading rather than as a denominator.

    A **backward** difference across the two samples preceding entry, not a
    centred one at the last free-flight sample: a centred difference there
    straddles the entry and would average the delivered slope with the first
    submerged one, so it would report the strike rather than the delivery.

    Args:
        profile: Sole depth profile.
        scene: Scene supplying the travel axis.
        entry_index: First submerged sample.

    Returns:
        The dimensionless descent slope, ``tan|attack angle|``.

    Raises:
        ValueError: If fewer than two free-flight samples precede entry, or the
            sole is not travelling forward across that step.
    """
    if entry_index < 2:
        raise ValueError(
            "the delivered path slope is measured across the two samples before "
            f"entry, and only {entry_index} precede it; record more free flight"
        )
    step = profile.position_m[entry_index - 1] - profile.position_m[entry_index - 2]
    along = float(step @ scene.travel_axis)
    if along <= 0.0:
        raise ValueError(
            "the sole must be travelling forward into the sand for a penetration "
            f"slope to be defined; the along-track step is {along:.6g} m"
        )
    return -float(step[2]) / along


def _descent_and_climb(
    profile: SoleDepthProfile, interval: StrikeInterval
) -> tuple[float, float]:
    """Return the sole's descent speed at entry and climb speed at exit.

    Both are read at the bounding **samples** of the submerged window, the same
    two the impulse balance integrates between, so the restitution and the
    balance describe one window rather than two.

    Args:
        profile: Sole depth profile.
        interval: Submerged window.

    Returns:
        ``(entry descent speed, exit climb speed)`` [m/s], the first positive
        downward and the second positive upward.

    Raises:
        BunkerShot3DValueError: If the sole is not measurably descending at the
            first submerged sample. Without a descent there is nothing for the
            sand to return, and the ratio would have no denominator.
    """
    entry_index = interval.entry_index
    vertical = profile.velocity_mps[:, 2]
    entry_descent = -float(vertical[entry_index])
    exit_climb = float(vertical[interval.exit_index])
    speed = float(np.linalg.norm(profile.velocity_mps[entry_index]))
    floor = _MIN_ENTRY_DESCENT_SPEED_FRACTION * speed
    if entry_descent <= floor:
        raise BunkerShot3DValueError(
            "the sole is not descending at the first submerged sample "
            f"({entry_descent:+.6g} m/s downward against a {speed:.4g} m/s "
            "sole speed), so there is no delivered descent for the sand to "
            "return and the descent-return ratio has no denominator. This is "
            "a trace whose entry crossing is a sampling artefact -- a scuff, "
            "or a dip resolved by one sample -- rather than a strike "
            "(issue #8703)"
        )
    return entry_descent, exit_climb


def _dig_skid_calibration(
    *,
    submerged_samples: int,
    dig_descent_return: float,
    skid_descent_return: float,
) -> DigSkidCalibration:
    """Build the calibration record that travels with a verdict.

    Args:
        submerged_samples: Samples the sole spent below the surface.
        dig_descent_return: The DIG threshold used.
        skid_descent_return: The SKID threshold used.

    Returns:
        The record, never reporting itself calibrated.
    """
    reasons = [DIG_SKID_UNCALIBRATED_REASON, DIG_SKID_BOUNCE_ORDERING_REASON]
    if submerged_samples < MIN_RESOLVED_SUBMERGED_SAMPLES:
        reasons.append(
            DIG_SKID_COARSE_WINDOW_REASON.format(
                samples=submerged_samples, minimum=MIN_RESOLVED_SUBMERGED_SAMPLES
            )
        )
    return DigSkidCalibration(
        calibrated=False,
        dig_descent_return=float(dig_descent_return),
        skid_descent_return=float(skid_descent_return),
        submerged_samples=submerged_samples,
        reasons=tuple(reasons),
    )


def dig_vs_skid(
    trace: StrikeTrace,
    head: HeadModel,
    scene: StrikeScene,
    *,
    dig_descent_return: float = DEFAULT_DIG_DESCENT_RETURN,
    skid_descent_return: float = DEFAULT_SKID_DESCENT_RETURN,
    gravity_mps2: float = STANDARD_GRAVITY_MPS2,
) -> DigSkidResult:
    """Classify the strike as digging or skidding, and balance vertical impulse.

    The discriminant is the **descent-return ratio**: how much of the descent
    the sole arrived with the sand hands back by the time it leaves. A digging
    head gives its descent away and crawls out; a skidding head bottoms out and
    is thrown out about as fast as it came down. See the module docstring for
    what this replaced, and for the sweep that showed it separates.

    Args:
        trace: Strike trace.
        head: Head the trace was recorded for.
        scene: Sand surface, ball and travel axis.
        dig_descent_return: Ratio at or **below** which the verdict is DIG.
            A convention, not a calibration; carried on the result.
        skid_descent_return: Ratio at or **above** which the verdict is SKID.
            A convention, not a calibration; carried on the result.
        gravity_mps2: Gravitational acceleration used for the impulse balance.

    Returns:
        The discriminator and the impulse balance. The verdict is
        **uncalibrated**; :attr:`DigSkidResult.calibration` says so and says
        why, and :meth:`DigSkidCalibration.require_calibrated` refuses.

    Raises:
        ValueError: If the thresholds are not ordered, if the sole is submerged
            for fewer than :data:`MIN_SUBMERGED_SAMPLES` samples so the two
            centred differences overlap, if the sole is not descending at the
            first submerged sample, or if the sole path does not support the
            delivered-slope measurement reported beside the ratio.
    """
    require(
        dig_descent_return < skid_descent_return,
        "dig_descent_return must be below skid_descent_return, leaving a marginal band",
        value=(dig_descent_return, skid_descent_return),
    )
    profile = sole_depth_profile(trace, head, scene)
    interval = _interval_from_profile(profile, trace, scene)
    submerged_samples = interval.exit_index - interval.entry_index + 1
    if submerged_samples < MIN_SUBMERGED_SAMPLES:
        raise BunkerShot3DValueError(
            f"the sole is below the surface for only {submerged_samples} "
            f"submerged samples, below the {MIN_SUBMERGED_SAMPLES} this metric "
            "needs. Both speeds are centred differences, so at this spacing "
            "they are the same measurement and the ratio would be an "
            "arithmetic identity rather than a strike. Record the strike more "
            "finely (issue #8703)"
        )
    entry_descent, exit_climb = _descent_and_climb(profile, interval)
    ratio = exit_climb / entry_descent
    if ratio <= dig_descent_return:
        verdict = DigSkidVerdict.DIG
    elif ratio >= skid_descent_return:
        verdict = DigSkidVerdict.SKID
    else:
        verdict = DigSkidVerdict.MARGINAL
    incoming_slope = _incoming_path_slope(profile, scene, interval.entry_index)
    return DigSkidResult(
        verdict=verdict,
        calibration=_dig_skid_calibration(
            submerged_samples=submerged_samples,
            dig_descent_return=dig_descent_return,
            skid_descent_return=skid_descent_return,
        ),
        entry_descent_speed_mps=entry_descent,
        exit_climb_speed_mps=exit_climb,
        descent_return_ratio=ratio,
        incoming_path_slope=incoming_slope,
        entry_attack_angle_rad=-float(np.arctan(incoming_slope)),
        **_vertical_balance(trace, head, interval, gravity_mps2),
    )


def _vertical_balance(
    trace: StrikeTrace,
    head: HeadModel,
    interval: StrikeInterval,
    gravity_mps2: float,
) -> dict[str, float]:
    """Return the vertical impulse balance over the submerged window.

    Args:
        trace: Strike trace.
        head: Head the trace was recorded for.
        interval: Submerged window.
        gravity_mps2: Gravitational acceleration [m/s^2].

    Returns:
        Mapping of the four :class:`DigSkidResult` impulse fields.

    Note:
        All three terms use the **sampled** window ``[entry_index, exit_index]``
        rather than the interpolated crossing times, so the balance closes on one
        window. The sand force is ~0 at the crossings, so the difference is
        negligible, but mixing the two windows would leave a spurious residual.
    """
    window = interval.sample_slice
    times = trace.time_s[window]
    sampled_duration_s = float(times[-1] - times[0])
    sand = float(np.trapezoid(trace.sand_force_N[window, 2], times))
    weight = -head.mass_kg * gravity_mps2 * sampled_duration_s
    velocity_z = trace.point_velocity_mps(head.centre_of_mass_body_m)[:, 2]
    measured = head.mass_kg * float(
        velocity_z[interval.exit_index] - velocity_z[interval.entry_index]
    )
    return {
        "vertical_sand_impulse_Ns": sand,
        "gravity_impulse_Ns": weight,
        "measured_vertical_momentum_change_Ns": measured,
        "constraint_vertical_impulse_Ns": measured - sand - weight,
    }
