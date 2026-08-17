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
Divot mass                   ``volume * sand bulk density`` [kg].
Entry penetration slope      ``d(depth)/ds`` evaluated over the first ``entry_window_m``
                             of travel after entry; dimensionless.
Incoming path slope          Chord slope ``-dz/d(travel)`` across the last free-flight
                             step; equals ``tan|attack angle|`` at delivery.
Slope ratio                  entry penetration slope / incoming path slope.
============================ ============================================================

**The discriminator.** A sole that keeps descending as steeply as it was
delivered is digging -- the bounce is not deflecting it. A sole that flattens is
skidding, i.e. planing on the sand. The reported quantity is the dimensionless
*ratio*, referenced to the delivered attack angle rather than to an absolute
angle, because the same geometry digs at -12 deg and skids at -2 deg. The
DIG/SKID/MARGINAL thresholds on that ratio are conventions and are arguments.

**The verdict is not calibrated, and at the shipped window it saturates**
------------------------------------------------------------------------

Issue #8703. Across the demo's full 77-point sweep -- geometric bounce
14-26 deg, sole width 16-24 mm, attack -2 to -14 deg, four sand conditions --
``dig_vs_skid`` returned ``MARGINAL`` at every point, with slope ratios
spanning only 0.9987 to 1.0000. The head really can deflect: ``simulate_shot``
integrates translation under the sand wrench and prescribes only the rotation.
The problem is *scale*. At a 25 m/s delivery, the 10 mm ``entry_window_m``
below is about **0.4 ms** of travel, and a 0.3 kg head under an order-5 N.s
total impulse cannot bend measurably in 0.4 ms. The signal is 0.13 % wide and
design-independent, so no threshold on it separates anything.

**Resizing the window does not rescue it, and was measured rather than
assumed.** Sweeping 48 design points (bounce 14/20/26 deg x sole 16/24 mm x
attack -2/-6/-10/-14 deg x firm/fluffy) over windows from 10 mm to the whole
divot gives:

===================  ==========  ==================================
entry window         ratio span  correlation with max sole depth
===================  ==========  ==================================
10 mm (shipped)      0.0015      -0.57
40 mm                0.0175      -0.54
80 mm                0.1602      -0.50
0.25 x divot length  0.0446      -0.68
0.50 x divot length  0.2800      -0.60
1.00 x divot length  0.0000      n/a
===================  ==========  ==================================

Widening the window does open the spread -- and opens it **the wrong way
round**. The correlation with sole depth is negative at every informative
window, so the designs that cut deepest (28-30 mm at 26 deg bounce, 24 mm sole,
-14 deg attack) come out *nearest the skid threshold*, while the shallowest
(9 mm, at -2 deg attack) sit nearest ``DIG``. Normalising by the delivered
slope divides out the attack angle, which is the variable that decides the
outcome. A resized window would therefore ship a **confidently inverted**
verdict, which is worse than a saturated one.

The last row is the reason no window is a free parameter: the entry chord
starts at the entry crossing and the depth returns to zero at the exit
crossing, so the ratio is pinned at 1 as the window shrinks and at 0 when it
reaches the whole divot, **for every design**. Any threshold on it is a
statement about where on that fixed arc the window was placed.

So the verdict is left as it is and marked, rather than retuned. Three things
are enforced here rather than described:

* a window shorter than one sample of travel is **refused**: the entry chord is
  then interpolated inside the single still-straight step the delivered chord
  was measured across, and the answer is the input divided by itself;
* a window that reaches the whole divot is **refused**, because the chord
  through the exit crossing is identically zero whatever the design did; and
* every :class:`DigSkidResult` carries a :class:`DigSkidCalibration` declaring
  the verdict uncalibrated, in the same shape the solver's ``ValidityVerdict``
  and the ball model's provenance record use, so a caller cannot read
  ``MARGINAL`` as a finding. A verdict that separates needs a quantity that
  varies with the design -- sole depth already does, cleanly -- and an absolute
  threshold on it that nobody has published yet.

**The force side of the same question.** Through the submerged window the
head's vertical momentum change must equal the sum of the sand impulse, the
gravity impulse and whatever the shaft and hands supply. All four are reported,
so the residual is visible rather than absorbed. The prismatic divot model and
the flat-surface scene are the two approximations here, and both are stated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.shared.python.core.contracts import ensure, require

from ..exceptions import BunkerShot3DValueError
from .enums import DigSkidVerdict
from .trace import STANDARD_GRAVITY_MPS2, HeadModel, StrikeScene, StrikeTrace

__all__ = [
    "DEFAULT_DIG_SLOPE_RATIO",
    "DEFAULT_ENTRY_WINDOW_M",
    "DEFAULT_SKID_SLOPE_RATIO",
    "DIG_SKID_INVERTED_SPREAD_REASON",
    "DIG_SKID_THIN_WINDOW_REASON",
    "DIG_SKID_UNCALIBRATED_REASON",
    "DIG_SKID_UNDEFLECTED_ENTRY_REASON",
    "MIN_INFORMATIVE_ENTRY_WINDOW_SAMPLES",
    "STANDARD_GRAVITY_MPS2",
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

#: Travel window after entry over which the penetration slope is evaluated [m].
#: 10 mm is short compared with a 25-150 mm entry distance, so it measures the
#: entry rather than the whole strike.
DEFAULT_ENTRY_WINDOW_M = 0.010

#: Slope ratio at or above which the strike is called a dig: the sole is still
#: descending at least as steeply as it was delivered.
DEFAULT_DIG_SLOPE_RATIO = 1.0

#: Slope ratio at or below which the strike is called a skid: the sole has lost
#: at least half of its delivered descent rate within the entry window.
DEFAULT_SKID_SLOPE_RATIO = 0.5

#: Entry-window length, in samples of along-track travel, below which the ratio
#: is dominated by the still-straight entry rather than by the design. Two
#: samples is the smallest window that contains a change of slope at all.
MIN_INFORMATIVE_ENTRY_WINDOW_SAMPLES = 2.0

#: Departure from 1.0 below which the entry chord is the delivered chord to
#: within floating point, so nothing about the design entered the ratio.
_UNDEFLECTED_RATIO_ATOL = 1e-9

DIG_SKID_UNCALIBRATED_REASON = (
    "the dig-versus-skid verdict is a convention on an uncalibrated ratio and "
    "must not be quoted as a finding. At the shipped 10 mm entry window the "
    "ratio spanned 0.9987-1.0000 over the whole 77-point demo design space and "
    "every point returned MARGINAL: 10 mm is 0.4 ms at greenside speed, and a "
    "0.3 kg head cannot deflect measurably in 0.4 ms. Widening the window does "
    "spread the ratio but correlates it negatively with sole depth, so a "
    "resized window would invert the verdict rather than calibrate it. A "
    "separating verdict needs a quantity that varies with the design and an "
    "absolute threshold nobody has published (issue #8703)."
)
"""Why a dig-versus-skid verdict is never a finding, however clean the trace."""

DIG_SKID_UNDEFLECTED_ENTRY_REASON = (
    "the entry chord equals the delivered chord to within floating point, so "
    "the ratio is 1 by arithmetic rather than by physics: over this window the "
    "sole is still travelling on the straight line it was delivered on"
)
"""Fires when the sand has not bent the path at all inside the entry window."""

DIG_SKID_THIN_WINDOW_REASON = (
    "the entry window spans only {samples:.2f} samples of along-track travel, "
    "below the {minimum:.0f} samples needed to contain a change of slope, so "
    "the ratio is bounded near 1 by the sampling as well as by the physics"
)
"""Template for the diagnostic that fires on a window of one or two samples."""

DIG_SKID_INVERTED_SPREAD_REASON = (
    "the entry window spans {fraction:.0%} of the divot; the ratio is pinned "
    "at 1 for a vanishing window and at 0 at the exit crossing, so a wide "
    "window reports where on that fixed arc it was placed rather than what the "
    "sole did, and it orders the designs opposite to sole depth (issue #8703)"
)
"""Template for the diagnostic that fires once the window is a wide chord."""

#: Fraction of the divot beyond which the entry chord is mostly the pinned
#: arc rather than the entry. Reported, not refused: only the full divot, where
#: the chord is identically zero, is refused.
_WIDE_CHORD_DIVOT_FRACTION = 0.25


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
        width_m: Effective cutting width used for the prismatic volume.
        volume_m3: ``section_area_m2 * width_m``.
        mass_kg: ``volume_m3 * bulk_density_kg_m3``.
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
    width_m: float
    volume_m3: float
    mass_kg: float
    bulk_density_kg_m3: float
    submerged_duration_s: float


def divot_metrics(
    trace: StrikeTrace,
    head: HeadModel,
    scene: StrikeScene,
    *,
    width_m: float,
    bulk_density_kg_m3: float,
) -> DivotMetrics:
    """Measure the divot the strike cuts.

    The volume model is prismatic: a constant-width channel whose depth follows
    the sole. It is an approximation -- a real divot has sloped walls and the
    ejected sand does not all come from below the sole path -- and it is stated
    here rather than buried, because divot mass feeds the momentum budget.

    Args:
        trace: Strike trace.
        head: Head the trace was recorded for.
        scene: Sand surface, ball and travel axis.
        width_m: Effective cutting width [m], normally the sole width in
            contact. Must be positive.
        bulk_density_kg_m3: Sand bulk density [kg/m^3]. The measured Covia
            Signature 500 bunker sand is 1550 kg/m^3 (research addendum).

    Returns:
        The divot metrics.

    Raises:
        ValueError: If ``width_m`` or ``bulk_density_kg_m3`` is not positive, if
            the sole never enters or never leaves the sand, or if the head does
            not travel forward monotonically through the strike (the section
            integral would double back on itself).
    """
    if not np.isfinite(width_m) or width_m <= 0.0:
        raise ValueError(f"width_m must be positive and finite, got {width_m}")
    if not np.isfinite(bulk_density_kg_m3) or bulk_density_kg_m3 <= 0.0:
        raise ValueError(
            f"bulk_density_kg_m3 must be positive and finite, got {bulk_density_kg_m3}"
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
    peak = int(np.argmax(profile.depth_m[window])) + interval.entry_index
    volume_m3 = section_area_m2 * width_m
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
        width_m=float(width_m),
        volume_m3=volume_m3,
        mass_kg=volume_m3 * float(bulk_density_kg_m3),
        bulk_density_kg_m3=float(bulk_density_kg_m3),
        submerged_duration_s=interval.duration_s,
    )


@dataclass(frozen=True)
class DigSkidCalibration:
    """Whether a dig-versus-skid verdict may be quoted, and why not.

    Carried on every :class:`DigSkidResult` so the statement travels with the
    number rather than living in report prose, in the same shape as
    :class:`bunkershot3d.solvers.envelope.ValidityVerdict` and
    :class:`bunkershot3d.sand.provenance.SandProvenance`.

    Attributes:
        calibrated: ``False``, unconditionally, until the verdict rests on a
            quantity that varies with the design and on an absolute threshold
            somebody has published (issue #8703).
        undeflected_entry: The entry chord equalled the delivered chord to
            within floating point, so the ratio is 1 by arithmetic.
        dig_slope_ratio: The DIG threshold the verdict was formed against.
        skid_slope_ratio: The SKID threshold the verdict was formed against.
        entry_window_samples: Entry window in samples of along-track travel.
            Below :data:`MIN_INFORMATIVE_ENTRY_WINDOW_SAMPLES` the window
            cannot contain a change of slope.
        entry_window_divot_fraction: Entry window as a share of divot length.
            The ratio is pinned at 1 as this goes to 0 and at 0 as it goes to
            1, for every design, so it says where on that arc the answer sits.
        reasons: Everything wrong with quoting this verdict, most general
            first.
    """

    calibrated: bool
    undeflected_entry: bool
    dig_slope_ratio: float
    skid_slope_ratio: float
    entry_window_samples: float
    entry_window_divot_fraction: float
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
            f"(window {self.entry_window_samples:.2f} samples, "
            f"{self.entry_window_divot_fraction:.1%} of the divot)"
        )
        return "\n".join([head, *(f"  - {reason}" for reason in self.reasons)])


@dataclass(frozen=True)
class DigSkidResult:
    """The dig-versus-skid discriminator and the vertical impulse balance.

    Attributes:
        verdict: DIG, SKID or MARGINAL. **Uncalibrated**: read
            :attr:`calibration` before quoting it (issue #8703).
        calibration: Whether the verdict may be quoted, and why not.
        entry_penetration_slope: ``d(depth)/d(travel)`` over the entry window;
            dimensionless, positive when still descending.
        incoming_path_slope: ``tan|attack angle|`` from the chord across the
            last free-flight step; dimensionless.
        slope_ratio: ``entry_penetration_slope / incoming_path_slope``.
        entry_attack_angle_rad: ``-arctan(incoming_path_slope)``; negative for a
            descending blow, matching the -2 to -12 deg delivery convention.
        entry_window_m: Travel window the entry slope was evaluated over.
        vertical_sand_impulse_Ns: ``integral of F_z dt`` over the submerged
            window; positive is upward, i.e. the sand holding the sole up.
        gravity_impulse_Ns: ``-m g * submerged duration``; always negative.
        measured_vertical_momentum_change_Ns: ``m * (v_z_exit - v_z_entry)`` of
            the head centre of mass.
        constraint_vertical_impulse_Ns: The residual -- what the shaft, hands
            and any unmodelled contact supplied. Reported rather than absorbed.
    """

    verdict: DigSkidVerdict
    calibration: DigSkidCalibration
    entry_penetration_slope: float
    incoming_path_slope: float
    slope_ratio: float
    entry_attack_angle_rad: float
    entry_window_m: float
    vertical_sand_impulse_Ns: float
    gravity_impulse_Ns: float
    measured_vertical_momentum_change_Ns: float
    constraint_vertical_impulse_Ns: float


def _incoming_path_slope(
    profile: SoleDepthProfile, scene: StrikeScene, entry_index: int
) -> float:
    """Return the chord slope of the last free-flight step before entry.

    A **backward** difference across the two samples preceding entry, not a
    centred one at the last free-flight sample: a centred difference there
    straddles the entry and would average the delivered slope with the first
    submerged one, hiding exactly the change the discriminator is looking for.

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


def _entry_sample_travel_m(
    profile: SoleDepthProfile, interval: StrikeInterval
) -> float:
    """Return the along-track sample spacing the entry window is resolved at.

    The median forward step over the submerged window and the two samples
    before it -- the stretch the discriminator actually reads. A median rather
    than the single step across the crossing, so one irregular sample cannot
    decide whether the trace is refused.

    Args:
        profile: Sole depth profile.
        interval: Submerged window.

    Returns:
        Sample spacing along travel [m].

    Raises:
        BunkerShot3DValueError: If the sole never advances over that stretch,
            so no spacing exists to judge the window against.
    """
    start = max(interval.entry_index - 2, 0)
    stop = min(interval.exit_index + 2, profile.travel_m.size)
    steps = np.diff(profile.travel_m[start:stop])
    forward = steps[steps > 0.0]
    if forward.size == 0:
        raise BunkerShot3DValueError(
            "the sole does not advance across the submerged window, so the "
            "entry window cannot be expressed in samples of travel"
        )
    return float(np.median(forward))


def _dig_skid_calibration(
    *,
    ratio: float,
    window_m: float,
    sample_travel_m: float,
    divot_length_m: float,
    dig_slope_ratio: float,
    skid_slope_ratio: float,
) -> DigSkidCalibration:
    """Build the calibration record that travels with a verdict.

    Args:
        ratio: The slope ratio the verdict was formed on.
        window_m: Entry window actually used, after clipping.
        sample_travel_m: Along-track sample spacing.
        divot_length_m: Length of the divot the window is a chord of.
        dig_slope_ratio: The DIG threshold used.
        skid_slope_ratio: The SKID threshold used.

    Returns:
        The record, never reporting itself calibrated.
    """
    samples = window_m / sample_travel_m
    fraction = window_m / divot_length_m
    undeflected = abs(ratio - 1.0) <= _UNDEFLECTED_RATIO_ATOL
    reasons = [DIG_SKID_UNCALIBRATED_REASON]
    if undeflected:
        reasons.append(DIG_SKID_UNDEFLECTED_ENTRY_REASON)
    if samples < MIN_INFORMATIVE_ENTRY_WINDOW_SAMPLES:
        reasons.append(
            DIG_SKID_THIN_WINDOW_REASON.format(
                samples=samples, minimum=MIN_INFORMATIVE_ENTRY_WINDOW_SAMPLES
            )
        )
    if fraction > _WIDE_CHORD_DIVOT_FRACTION:
        reasons.append(DIG_SKID_INVERTED_SPREAD_REASON.format(fraction=fraction))
    return DigSkidCalibration(
        calibrated=False,
        undeflected_entry=undeflected,
        dig_slope_ratio=float(dig_slope_ratio),
        skid_slope_ratio=float(skid_slope_ratio),
        entry_window_samples=samples,
        entry_window_divot_fraction=fraction,
        reasons=tuple(reasons),
    )


def dig_vs_skid(
    trace: StrikeTrace,
    head: HeadModel,
    scene: StrikeScene,
    *,
    entry_window_m: float = DEFAULT_ENTRY_WINDOW_M,
    dig_slope_ratio: float = DEFAULT_DIG_SLOPE_RATIO,
    skid_slope_ratio: float = DEFAULT_SKID_SLOPE_RATIO,
    gravity_mps2: float = STANDARD_GRAVITY_MPS2,
) -> DigSkidResult:
    """Classify the strike as digging or skidding, and balance vertical impulse.

    Args:
        trace: Strike trace.
        head: Head the trace was recorded for.
        scene: Sand surface, ball and travel axis.
        entry_window_m: Travel window after entry over which the penetration
            slope is evaluated. Must be longer than one sample of along-track
            travel and shorter than the divot: outside that band the ratio is
            1 or 0 by construction rather than by physics (issue #8703).
        dig_slope_ratio: Slope ratio at or above which the verdict is DIG.
            A convention, not a calibration; carried on the result.
        skid_slope_ratio: Slope ratio at or below which the verdict is SKID.
            A convention, not a calibration; carried on the result.
        gravity_mps2: Gravitational acceleration used for the impulse balance.

    Returns:
        The discriminator and the impulse balance. The verdict is
        **uncalibrated**; :attr:`DigSkidResult.calibration` says so and says
        why, and :meth:`DigSkidCalibration.require_calibrated` refuses.

    Raises:
        ValueError: If the thresholds are not ordered, the window is not
            positive, the window is degenerate at either end (shorter than one
            sample of travel, or as long as the divot), or the sole path does
            not support a slope measurement.
    """
    require(
        entry_window_m > 0.0, "entry_window_m must be positive", value=entry_window_m
    )
    require(
        skid_slope_ratio < dig_slope_ratio,
        "skid_slope_ratio must be below dig_slope_ratio, leaving a marginal band",
        value=(skid_slope_ratio, dig_slope_ratio),
    )
    profile = sole_depth_profile(trace, head, scene)
    interval = _interval_from_profile(profile, trace, scene)
    divot_length_m = interval.exit_travel_m - interval.entry_travel_m
    if float(entry_window_m) >= divot_length_m:
        raise BunkerShot3DValueError(
            f"the entry window of {entry_window_m * 1e3:.4g} mm reaches the "
            f"whole {divot_length_m * 1e3:.4g} mm divot. The entry chord is "
            "taken from the entry crossing, and the depth returns to zero at "
            "the exit crossing, so a chord that spans the divot is identically "
            "zero for every design and the verdict would be SKID whatever the "
            "sole did (issue #8703)"
        )
    window_m = float(entry_window_m)
    sample_travel_m = _entry_sample_travel_m(profile, interval)
    if window_m <= sample_travel_m:
        raise BunkerShot3DValueError(
            f"the entry window of {window_m * 1e3:.4g} mm is shorter than one "
            f"sample of travel ({sample_travel_m * 1e3:.4g} mm), so the entry "
            "penetration slope is interpolated inside the single step the "
            "incoming path slope was measured across and the ratio is 1 by "
            "construction rather than by physics. Record the strike more "
            "finely, or widen the window (issue #8703)"
        )
    depth_at_window = profile.depth_at_travel_m(interval.entry_travel_m + window_m)
    entry_slope = depth_at_window / window_m
    incoming_slope = _incoming_path_slope(profile, scene, interval.entry_index)
    ratio = entry_slope / incoming_slope
    if ratio >= dig_slope_ratio:
        verdict = DigSkidVerdict.DIG
    elif ratio <= skid_slope_ratio:
        verdict = DigSkidVerdict.SKID
    else:
        verdict = DigSkidVerdict.MARGINAL
    return DigSkidResult(
        verdict=verdict,
        calibration=_dig_skid_calibration(
            ratio=ratio,
            window_m=window_m,
            sample_travel_m=sample_travel_m,
            divot_length_m=divot_length_m,
            dig_slope_ratio=dig_slope_ratio,
            skid_slope_ratio=skid_slope_ratio,
        ),
        entry_penetration_slope=entry_slope,
        incoming_path_slope=incoming_slope,
        slope_ratio=ratio,
        entry_attack_angle_rad=-float(np.arctan(incoming_slope)),
        entry_window_m=window_m,
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
