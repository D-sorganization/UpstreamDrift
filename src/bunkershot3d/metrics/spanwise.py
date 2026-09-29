"""Spanwise (heel-to-toe) sole load -- how a grind shares the strike (#8699).

BunkerShot3D exists to compare wedge sole *shapes*, and the shape parameters a
designer actually grinds -- heel relief, toe relief, heel-toe rocker -- run
across the blade. F0 already resolves that: :mod:`bunkershot3d.geometry.lofting`
builds the mesh from those fields and
:class:`~bunkershot3d.solvers.drft.DRFTSolver` integrates a per-element response
over the whole span. What was missing was the *report*: ``ShotResult`` carries
the resultant wrench and nothing else, so the heel-to-toe structure the solver
computes was summed away before anyone could see it. ADR-0044 records the
distinction -- F0 is reporting-blind, not geometrically blind; F1 is the tier
blind by construction.

This module is the reporting layer. It consumes the same
:class:`~bunkershot3d.metrics.bounce_map.SoleLoadTrace` that
:func:`~bunkershot3d.metrics.bounce_map.bounce_utilisation` does, so it is
defined on the *result artifact* and means the same thing at every tier that
can produce one.

The axis
--------

:mod:`bunkershot3d.geometry.lofting` fixes the head body frame: ``+x`` rearward,
``+y`` **heel to toe**, ``+z`` up. Every signed quantity here follows that axis,
so *negative is toward the heel* and *positive is toward the toe*, without
exception. The sign is the physical claim -- shedding load from the toe must
move the balance negative -- and it is pinned by test, not by this sentence.

What is reported, and in what units
-----------------------------------

============================== ================= =============================================
Quantity                       Unit              Meaning
============================== ================= =============================================
Spanwise distribution          N.s and m^2       Impulse and area binned across the span.
Heel/toe balance               dimensionless     ``(I_toe - I_heel) / I_total`` about mid-span.
                                                 ``0`` for a symmetric sole, ``-1`` when the
                                                 heel half carries everything.
Spanwise centroid              m                 Impulse-weighted mean station, body axes.
Normalised centroid            dimensionless     The same, as a fraction of the half-span.
Centroid migration             m                 Where the load sits at each *instant*, and
                                                 how far that walks across the strike.
Outer-third fractions          dimensionless     Share of impulse carried by each end third of
                                                 the span, and by the two ends together.
============================== ================= =============================================

The summaries are computed from the **elements**, not from the bins, so they do
not move when a caller changes ``n_bins``. The bins are the chart; the
summaries are the numbers.

What this is not
----------------

The distribution is the load **the sole carried**. It is not the sand's
response to it, and nothing here is a measurement:

* No tier below F2 resolves a grain. F0 integrates an empirical resistive
  stress element by element and never forms a sand velocity at any resolution
  (epic #8699), so a heel-to-toe distribution from F0 is a statement about the
  swept geometry and the fitted response -- not an observation of how sand
  shared itself across the blade.
* No spanwise sole-pressure corpus exists for a bunker shot at any tier, so
  none of these numbers has ever been checked against a measurement.
* **F1 is refused outright.** ADR-0033 makes it a 2-D plane-strain tier and
  ADR-0044 records that it is blind to out-of-plane geometry by construction;
  a spanwise distribution extruded from one section would be an artifact of the
  extrusion, which is worse than no number.

Those statements travel on :class:`SpanwiseCredibility`, attached to every
result, in the same shape as
:class:`~bunkershot3d.solvers.envelope.ValidityVerdict` and
:class:`~bunkershot3d.metrics.divot.DigSkidCalibration` -- so a report cannot
show the chart without them.

Refusals rather than smoothing
------------------------------

A distribution binned finer than the elements that produced it is an
interpolation dressed as a measurement. So the bin count is **explicit** and
checked against the element resolution: fewer than
:data:`MIN_ELEMENTS_PER_SPANWISE_BIN` stations per bin, or any bin holding no
element at all, is a :class:`~bunkershot3d.exceptions.BunkerShot3DValueError`
naming the largest bin count the sole can support. A sole whose elements all
sit at one spanwise station is refused outright -- it resolves no span, so it
distributes nothing.

Every guard is a plain ``raise``: ``python -O`` strips assertions and
``DBC_LEVEL=off`` disables contracts, and an honesty guard that evaporates
under an optimisation flag is not a guard.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..exceptions import BunkerShot3DValueError
from ..solvers import EnvelopeStatus, FidelityTier, ValidityVerdict
from .bounce_map import LoadProfile, SoleLoadTrace

__all__ = [
    "MIN_ELEMENTS_PER_SPANWISE_BIN",
    "MIN_LOADED_SAMPLES_FOR_MIGRATION",
    "MIN_SPANWISE_BINS",
    "MIN_SPANWISE_STATIONS",
    "SPANWISE_AXIS_INDEX",
    "SPANWISE_F0_ANALYTIC_REASON",
    "SPANWISE_PLANE_STRAIN_REASON",
    "SPANWISE_SOLE_NOT_SAND_REASON",
    "SPANWISE_UNMEASURED_REASON",
    "SpanwiseCredibility",
    "SpanwiseDistribution",
    "SpanwiseLoad",
    "SpanwiseMigration",
    "spanwise_load",
]

#: Body axis the span runs along. ``bunkershot3d.geometry.lofting`` fixes
#: ``+x`` rearward, ``+y`` heel to toe, ``+z`` up -- so heel is negative ``y``
#: and toe is positive ``y``, and every signed quantity here follows that.
SPANWISE_AXIS_INDEX = 1

#: Distinct spanwise stations a bin must average over before its height means
#: anything. One element per bin is not a distribution, it is the elements
#: redrawn; two is the smallest count that averages at all.
MIN_ELEMENTS_PER_SPANWISE_BIN = 2

#: Bins below which there is no heel-to-toe information left to carry.
MIN_SPANWISE_BINS = 2

#: Distinct spanwise stations below which the sole resolves no span.
MIN_SPANWISE_STATIONS = 2

#: Loaded samples below which a centroid is a position, not a migration.
MIN_LOADED_SAMPLES_FOR_MIGRATION = 2

SPANWISE_SOLE_NOT_SAND_REASON = (
    "this is the load the sole carried heel to toe, not the sand's response to "
    "it: no tier below F2 resolves a grain, so nothing here says how sand "
    "shared itself across the blade"
)
"""Why a spanwise distribution is never a statement about the sand."""

SPANWISE_UNMEASURED_REASON = (
    "no spanwise sole-pressure measurement of a bunker shot has been published "
    "at any tier, so this distribution has never been checked against one"
)
"""Why no constant behind this distribution has been measured."""

SPANWISE_F0_ANALYTIC_REASON = (
    "F0 integrates an empirical resistive stress over the swept surface and "
    "never solves the sand's motion, so the heel-to-toe structure it reports "
    "follows from the sole geometry and the fitted response alone"
)
"""What an F0 spanwise distribution is a statement about."""

SPANWISE_PLANE_STRAIN_REASON = (
    "F1 is a 2-D plane-strain tier (ADR-0033) and has no span: it is blind to "
    "out-of-plane geometry by construction (ADR-0044), so a heel-to-toe "
    "distribution taken from it would report the extrusion, not the grind"
)
"""Why F1 cannot produce a spanwise distribution at all."""


@dataclass(frozen=True)
class SpanwiseCredibility:
    """The statement a spanwise distribution has to be read under.

    Attributes:
        fidelity_tier: Which rung of the ADR-0032 ladder produced the trace.
        verdict: The solver's envelope verdict, when the caller supplied one.
            ``None`` means *unstated*, never *fine*.
        reasons: Everything wrong with reading the distribution as a
            measurement, most general first.
    """

    fidelity_tier: FidelityTier
    verdict: ValidityVerdict | None
    reasons: tuple[str, ...]

    @property
    def status(self) -> EnvelopeStatus | None:
        """How much of the distribution may be believed, when it was stated."""
        return None if self.verdict is None else self.verdict.status

    def measured_constants(self) -> tuple[str, ...]:
        """Names of every constant measured on a real bunker shot.

        Empty, and required to stay empty: mirrors
        :meth:`bunkershot3d.metrics.divot.DigSkidCalibration.measured_constants`
        and :meth:`bunkershot3d.solvers.MaterialResponse.measured_constants`.

        Returns:
            An empty tuple.
        """
        return ()

    def require_sand_response(self) -> None:
        """Refuse to let a caller read the sole's load as the sand's response.

        Raises:
            BunkerShot3DValueError: Always. The quantity does not exist at any
                tier this metric accepts.
        """
        raise BunkerShot3DValueError(
            "a spanwise sole-load distribution is not a sand response and may "
            "not be quoted as one: " + " ".join(self.reasons)
        )

    def summary(self) -> str:
        """Return the statement a report shows beside the chart.

        Returns:
            One line naming the tier and status, then one line per reason.
        """
        status = "unstated" if self.status is None else self.status.value
        head = (
            f"spanwise sole load: tier {self.fidelity_tier.value}, "
            f"envelope {status}, UNMEASURED"
        )
        return "\n".join([head, *(f"  - {reason}" for reason in self.reasons)])


@dataclass(frozen=True)
class SpanwiseDistribution:
    """Impulse binned across the span -- the heel-to-toe grind chart.

    Attributes:
        profile: The binned impulse and area, along
            :data:`SPANWISE_AXIS_INDEX`. Reuses
            :class:`~bunkershot3d.metrics.bounce_map.LoadProfile` so the binned
            arrays have one definition in the package.
        peak_force_N: ``(n_bins,)`` largest load the whole bin carried at any
            one sample [N] -- the bin's own peak, not a sum of element peaks
            that never coincided.
        element_count: ``(n_bins,)`` elements falling in each bin.
    """

    profile: LoadProfile
    peak_force_N: np.ndarray
    element_count: np.ndarray

    def __post_init__(self) -> None:
        """Validate the distribution.

        Raises:
            BunkerShot3DValueError: If the three arrays disagree in length, or
                the profile does not run along the spanwise axis.
        """
        if self.profile.axis_index != SPANWISE_AXIS_INDEX:
            raise BunkerShot3DValueError(
                "a spanwise distribution must be binned along body axis "
                f"{SPANWISE_AXIS_INDEX}, got {self.profile.axis_index}"
            )
        counts = {
            "impulse_Ns": np.size(self.profile.impulse_Ns),
            "peak_force_N": np.size(self.peak_force_N),
            "element_count": np.size(self.element_count),
        }
        if len(set(counts.values())) != 1:
            raise BunkerShot3DValueError(
                f"a spanwise distribution needs one value per bin, got {counts}"
            )

    @property
    def n_bins(self) -> int:
        """Number of spanwise bins."""
        return int(np.size(self.profile.impulse_Ns))

    @property
    def bin_edges_m(self) -> np.ndarray:
        """``(n_bins + 1,)`` bin edges in body coordinates [m], heel to toe."""
        return self.profile.bin_edges_m

    @property
    def impulse_Ns(self) -> np.ndarray:  # noqa: N802 - the unit belongs in the name
        """``(n_bins,)`` impulse carried by each bin [N.s]."""
        return self.profile.impulse_Ns

    @property
    def area_m2(self) -> np.ndarray:
        """``(n_bins,)`` sole area falling in each bin [m^2]."""
        return self.profile.area_m2

    @property
    def impulse_fraction(self) -> np.ndarray:
        """``(n_bins,)`` each bin's share of the total impulse; sums to one."""
        return self.profile.impulse_fraction

    @property
    def bin_centre_m(self) -> np.ndarray:
        """``(n_bins,)`` spanwise station at the middle of each bin [m]."""
        edges = self.profile.bin_edges_m
        return 0.5 * (edges[:-1] + edges[1:])

    @property
    def impulse_density_Pa_s(self) -> np.ndarray:  # noqa: N802 - unit in name
        """``(n_bins,)`` impulse per unit area [Pa.s], so wide bins do not win."""
        return self.profile.impulse_Ns / self.profile.area_m2


@dataclass(frozen=True)
class SpanwiseMigration:
    """Where across the span the load sits at each instant.

    A single centroid over the whole strike cannot say that a sole engaged at
    the toe and finished on the heel, which is exactly what relief is ground to
    change.

    Attributes:
        time_s: ``(T,)`` sample times [s].
        centroid_body_m: ``(T,)`` load-weighted spanwise station at each sample
            [m], **NaN** where the sole carried nothing. NaN rather than zero,
            because zero is mid-span and would read as a centred strike.
        loaded_sample_mask: ``(T,)`` samples at which the sole carried load.
    """

    time_s: np.ndarray
    centroid_body_m: np.ndarray
    loaded_sample_mask: np.ndarray

    @property
    def loaded_sample_count(self) -> int:
        """Number of samples at which the sole carried any load."""
        return int(np.count_nonzero(self.loaded_sample_mask))

    def _loaded_centroids(self) -> np.ndarray:
        """Return the centroid at every loaded sample.

        Returns:
            ``(n_loaded,)`` spanwise stations [m].

        Raises:
            BunkerShot3DValueError: If fewer than
                :data:`MIN_LOADED_SAMPLES_FOR_MIGRATION` samples carried load.
        """
        if self.loaded_sample_count < MIN_LOADED_SAMPLES_FOR_MIGRATION:
            raise BunkerShot3DValueError(
                "this strike has only one loaded sample, so the spanwise "
                "centroid is a position and not a migration; report the "
                "centroid instead of a travel across it"
            )
        return self.centroid_body_m[self.loaded_sample_mask]

    def range_m(self) -> float:
        """Return how far the centroid walked across the span [m].

        Unsigned: the full spread between the heel-most and toe-most instant,
        which a centroid that crosses back over itself still reports.

        Returns:
            ``max - min`` of the loaded centroids [m].

        Raises:
            BunkerShot3DValueError: If the strike has fewer than two loaded
                samples.
        """
        loaded = self._loaded_centroids()
        return float(loaded.max() - loaded.min())

    def net_travel_m(self) -> float:
        """Return the signed walk from first engagement to last [m].

        Negative is toward the heel, positive toward the toe, following
        :data:`SPANWISE_AXIS_INDEX`.

        Returns:
            Last loaded centroid minus the first [m].

        Raises:
            BunkerShot3DValueError: If the strike has fewer than two loaded
                samples.
        """
        loaded = self._loaded_centroids()
        return float(loaded[-1] - loaded[0])


@dataclass(frozen=True)
class SpanwiseLoad:
    """How a sole shared one strike across its span.

    Attributes:
        distribution: The binned chart.
        migration: Where the load sat at each instant.
        credibility: The statement all of this is read under.
        total_impulse_Ns: Impulse summed over every element [N.s].
        span_m: Distance from the heel-most to the toe-most element [m].
        mid_span_body_m: The station the heel and toe halves are split at [m].
        heel_toe_balance: ``(I_toe - I_heel) / I_total`` about mid-span.
            Dimensionless in ``[-1, 1]``; **0 for a symmetric sole**, negative
            when the heel half carries more.
        centroid_body_m: Impulse-weighted spanwise station over the whole
            strike [m], body axes.
        centroid_normalised: The same, as a fraction of the half-span from
            mid-span; dimensionless in ``[-1, 1]``.
        heel_third_fraction: Share of the impulse carried by the heel third of
            the span.
        toe_third_fraction: The same for the toe third.
        outer_third_fraction: The two ends together -- how much of the strike
            the extremities of the blade take.
    """

    distribution: SpanwiseDistribution
    migration: SpanwiseMigration
    credibility: SpanwiseCredibility
    total_impulse_Ns: float
    span_m: float
    mid_span_body_m: float
    heel_toe_balance: float
    centroid_body_m: float
    centroid_normalised: float
    heel_third_fraction: float
    toe_third_fraction: float
    outer_third_fraction: float

    def summary(self) -> str:
        """Return the numbers a designer reads, one line each.

        Returns:
            A multi-line statement, ending in the credibility statement so the
            two cannot be separated.
        """
        toward = "heel" if self.heel_toe_balance < 0.0 else "toe"
        lines = [
            f"span {self.span_m * 1e3:.1f} mm about {self.mid_span_body_m * 1e3:+.1f} "
            f"mm, {self.total_impulse_Ns:.4g} N.s carried",
            f"heel/toe balance {self.heel_toe_balance:+.4f} (toward the {toward}; "
            "0 is symmetric, - is heel, + is toe)",
            f"spanwise centroid {self.centroid_body_m * 1e3:+.2f} mm "
            f"({self.centroid_normalised:+.4f} of the half-span)",
            f"outer thirds {self.outer_third_fraction:.4f} of the impulse "
            f"(heel {self.heel_third_fraction:.4f}, toe {self.toe_third_fraction:.4f})",
            self.credibility.summary(),
        ]
        return "\n".join(lines)


def _spanwise_stations(load: SoleLoadTrace) -> np.ndarray:
    """Return the spanwise station of every element, refusing a collapsed sole.

    Args:
        load: The per-element sole loading.

    Returns:
        ``(E,)`` body-``y`` stations [m].

    Raises:
        BunkerShot3DValueError: If every element sits at one spanwise station,
            in which case the sole resolves no span to distribute anything
            across.
    """
    stations = np.asarray(load.element_centroid_body_m, dtype=float)[
        :, SPANWISE_AXIS_INDEX
    ]
    if np.unique(stations).size < MIN_SPANWISE_STATIONS:
        raise BunkerShot3DValueError(
            "every element of this sole sits at one spanwise station, so it "
            "resolves no heel-to-toe span; a distribution of it would be a "
            "single number drawn as a chart"
        )
    return stations


def _check_bin_count(stations: np.ndarray, n_bins: int) -> None:
    """Refuse a bin count the element resolution cannot support.

    Args:
        stations: ``(E,)`` spanwise stations [m].
        n_bins: Bins the caller asked for.

    Raises:
        BunkerShot3DValueError: If the count is below
            :data:`MIN_SPANWISE_BINS`, or finer than the elements support.
    """
    if n_bins < MIN_SPANWISE_BINS:
        raise BunkerShot3DValueError(
            f"a spanwise distribution needs at least {MIN_SPANWISE_BINS} bins "
            f"to carry any heel-to-toe information, got {n_bins}"
        )
    resolved = int(np.unique(stations).size)
    supported = resolved // MIN_ELEMENTS_PER_SPANWISE_BIN
    if n_bins > supported:
        raise BunkerShot3DValueError(
            f"this sole resolves {resolved} spanwise stations, which supports "
            f"at most {supported} bins at {MIN_ELEMENTS_PER_SPANWISE_BIN} "
            f"stations per bin; {n_bins} were asked for. Binning finer than "
            "the elements returns a smoothed picture of a resolution the "
            "model does not have"
        )


def _bin_spanwise(
    load: SoleLoadTrace, stations: np.ndarray, n_bins: int
) -> SpanwiseDistribution:
    """Bin the strike across the span, refusing a bin no element falls in.

    Args:
        load: The per-element sole loading.
        stations: ``(E,)`` spanwise stations [m].
        n_bins: Equal-width bins across the span.

    Returns:
        The binned distribution.

    Raises:
        BunkerShot3DValueError: If any bin holds no element, which means the
            elements are clustered and the empty bins are an artifact of the
            bin width rather than a statement about the sole.
    """
    edges = np.linspace(float(stations.min()), float(stations.max()), n_bins + 1)
    index = np.clip(np.digitize(stations, edges[1:-1]), 0, n_bins - 1)
    counts = np.bincount(index, minlength=n_bins)
    if np.any(counts == 0):
        raise BunkerShot3DValueError(
            f"{int(np.count_nonzero(counts == 0))} of {n_bins} spanwise bins "
            "hold no element at all: the elements are clustered, so an empty "
            "bin here describes the bin width and not the sole"
        )
    # One column per bin, one row per element: right-multiplying by it sums any
    # per-element quantity into its bin, and applied to the (T, E) force block
    # it gives each bin's own load at each sample -- so the peak below is a load
    # the bin really carried at one instant, not a sum of element peaks that
    # never coincided.
    selector = np.zeros((stations.size, n_bins))
    selector[np.arange(stations.size), index] = 1.0
    forces = np.asarray(load.element_normal_force_N, dtype=float)
    bin_force = forces @ selector
    return SpanwiseDistribution(
        profile=LoadProfile(
            axis_index=SPANWISE_AXIS_INDEX,
            bin_edges_m=edges,
            impulse_Ns=np.trapezoid(forces, load.time_s, axis=0) @ selector,
            area_m2=np.asarray(load.element_area_m2, dtype=float) @ selector,
        ),
        peak_force_N=bin_force.max(axis=0),
        element_count=counts,
    )


def _migration(load: SoleLoadTrace, stations: np.ndarray) -> SpanwiseMigration:
    """Track the load-weighted spanwise station sample by sample.

    Args:
        load: The per-element sole loading.
        stations: ``(E,)`` spanwise stations [m].

    Returns:
        The migration, NaN at every sample carrying nothing.
    """
    forces = np.asarray(load.element_normal_force_N, dtype=float)
    per_sample = forces.sum(axis=1)
    loaded = per_sample > 0.0
    centroid = np.full(per_sample.shape, np.nan)
    centroid[loaded] = (forces[loaded] @ stations) / per_sample[loaded]
    return SpanwiseMigration(
        time_s=np.asarray(load.time_s, dtype=float),
        centroid_body_m=centroid,
        loaded_sample_mask=loaded,
    )


def _credibility(
    fidelity_tier: FidelityTier, verdict: ValidityVerdict | None
) -> SpanwiseCredibility:
    """Assemble the statement the distribution is read under.

    Args:
        fidelity_tier: The tier that produced the trace.
        verdict: The solver's envelope verdict, or ``None`` when unstated.

    Returns:
        The credibility record.
    """
    reasons = [SPANWISE_SOLE_NOT_SAND_REASON, SPANWISE_UNMEASURED_REASON]
    if fidelity_tier is FidelityTier.F0:
        reasons.append(SPANWISE_F0_ANALYTIC_REASON)
    return SpanwiseCredibility(
        fidelity_tier=fidelity_tier, verdict=verdict, reasons=tuple(reasons)
    )


def spanwise_load(
    load: SoleLoadTrace,
    *,
    n_bins: int,
    fidelity_tier: FidelityTier,
    verdict: ValidityVerdict | None = None,
) -> SpanwiseLoad:
    """Resolve one strike heel to toe across the sole.

    Args:
        load: Per-element sole loading over the strike, the same artifact
            :func:`~bunkershot3d.metrics.bounce_map.bounce_utilisation` takes.
        n_bins: Equal-width spanwise bins. **Explicit and required**: a default
            would silently fix the resolution of every chart drawn from this.
        fidelity_tier: Which rung of the ADR-0032 ladder produced the trace.
            Required, because a spanwise distribution means different things at
            different tiers and F1 cannot produce one at all.
        verdict: The solver's envelope verdict. Optional, but a distribution
            without one carries ``status is None`` -- *unstated*, not *fine*.

    Returns:
        The distribution, its summaries, its migration and its credibility.

    Raises:
        BunkerShot3DValueError: If the tier is F1 (plane strain has no span);
            if the sole resolves fewer than
            :data:`MIN_SPANWISE_STATIONS` spanwise stations; if ``n_bins`` is
            below :data:`MIN_SPANWISE_BINS` or finer than the elements support;
            if any bin holds no element; or if the sole carried no load, in
            which case there is no distribution and every fraction would be a
            division by zero dressed up as an answer.
        OutOfEnvelopeError: If a supplied ``verdict`` refuses the query.
    """
    if fidelity_tier is FidelityTier.F1:
        raise BunkerShot3DValueError(
            "F1 cannot produce a spanwise distribution: " + SPANWISE_PLANE_STRAIN_REASON
        )
    if verdict is not None:
        verdict.require_usable()
    stations = _spanwise_stations(load)
    _check_bin_count(stations, n_bins)
    impulse = np.trapezoid(
        np.asarray(load.element_normal_force_N, dtype=float), load.time_s, axis=0
    )
    total = float(impulse.sum())
    if total <= 0.0:
        raise BunkerShot3DValueError(
            "this sole carried no load over the trace, so there is no spanwise "
            "distribution of it; check that the strike is inside the window"
        )
    low, high = float(stations.min()), float(stations.max())
    span = high - low
    mid = 0.5 * (low + high)
    third = span / 3.0
    heel_third = float(impulse[stations < low + third].sum()) / total
    toe_third = float(impulse[stations > high - third].sum()) / total
    balance = (
        float(impulse[stations > mid].sum()) - float(impulse[stations < mid].sum())
    ) / total
    centroid = float(impulse @ stations) / total
    return SpanwiseLoad(
        distribution=_bin_spanwise(load, stations, n_bins),
        migration=_migration(load, stations),
        credibility=_credibility(fidelity_tier, verdict),
        total_impulse_Ns=total,
        span_m=span,
        mid_span_body_m=mid,
        heel_toe_balance=balance,
        centroid_body_m=centroid,
        centroid_normalised=(centroid - mid) / (0.5 * span),
        heel_third_fraction=heel_third,
        toe_third_fraction=toe_third,
        outer_third_fraction=heel_third + toe_third,
    )
