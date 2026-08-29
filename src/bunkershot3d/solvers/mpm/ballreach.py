"""What actually reaches the ball, resolved over its surface (#8712).

At F0 the ball is a downstream result: ``compute_bunker_launch`` derives a
launch from the solver's impulse (#8657) and nothing ever strikes the
ball, so "the sand arriving at the ball" has no referent to resolve.
ADR-0033 and #8733 §1 gave it one -- the ball is a rigid circular section
inside the F1 solve -- and #8733 §2 gave every body its own exact momentum
ledger.  This module is the reading of that ledger: **where on the ball
the sand arrives, how hard, when, and how much of what the club put into
the sand comes back out on the other side**.

It is a re-reading, not a second force model
--------------------------------------------

Nothing here computes a force.  :class:`~.contact.BodyContact` already
carries the exact ledger ``m_i (v_i^after - v_i^before)`` the projection
produced, and :attr:`~.contact.BodyContact.nodes` carries it *before* it
was summed.  Every number below is that ledger regrouped -- by half, by
sector, by step -- which is why adding this changes no momentum budget
and why the conservation identity #8733 §2 pinned still closes to
round-off with the ball's term in it.

Signs, once, so they are not guessed at
---------------------------------------

The solver's ledger is signed **body on sand**: ``impulse_n_s`` is what
the body pushed into the sand.  What #8712 asks for is the reaction, the
sand on the ball, so everything this module reports is the negative of
that, and every field is named for the direction it is in.

What travels with every number here
------------------------------------

* **Per unit out-of-plane width, on an infinite cylinder.**  Plane strain
  has no third dimension; the body the sand meets is a cylinder of
  unlimited length whose section is the ball's great circle.  So the
  quantities are ``_n_s_per_m`` and ``_n_per_m`` in their *names*, and
  the absolute force on the ball -- the per-unit-width flux multiplied by
  some width -- **raises**
  :attr:`~.envelope.RefusedQuantity.OUT_OF_PLANE`, because the ball has
  no width to multiply by.  A club may be given a declared effective
  width, which is a stated assumption; an infinite cylinder cannot,
  because there is no assumption to state.
* **The in-plane resolution is qualitative.**  Below-equator against
  face-side, and the sector resolution that refines it, are directions
  the plane-strain section genuinely has, so they may be reported -- as
  shares of an impulse on a body of the wrong three-dimensional shape,
  never as a load distribution on a golf ball.  Heel-toe and any lateral
  distribution raise.
* **Ball launch is still F0's.**  :meth:`BallReachHistory.launch_velocity_m_s`
  raises :attr:`~.envelope.RefusedQuantity.BALL_LAUNCH`.  This module
  answers what reaches the ball; it does not answer where the ball goes,
  and the boundary between those two is what keeps the first answer
  honest.
* **The tier, on every result.**  A number about the ball is not more
  trustworthy than the tier that produced it, so
  :class:`BallReachHistory` and :class:`SandVersusClub` carry the
  :class:`~bunkershot3d.solvers.envelope.ValidityVerdict` itself and
  restate :data:`BALL_REACH_TIER_NOTE` in their summaries.

The comparison the epic was built for
--------------------------------------

:class:`SandVersusClub` is the one that matters: what the sand delivers
to the ball against what the club delivers to the sand.  It is reported
as a **dimensionless fraction and a pair of timings**, and not as two
forces, for two separate reasons that happen to point the same way --
absolute club force is refused at this tier
(:attr:`~.envelope.RefusedQuantity.CLUB_FORCE`, F1 is deliberately
under-resolved at the leading edge where club force lives), and an
absolute ball force does not exist at all in plane strain.  The ratio of
two per-unit-width ledger terms from the *same* solve divides both of
those objections out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NoReturn

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..envelope import MAX_VALIDATED_SPEED_M_S, ValidityVerdict
from ..exceptions import SolverInputError
from ..protocol import FidelityTier
from .ball import PLANE_STRAIN_BALL_NOTE, BallContactSplit, BallSection
from .body import ContactImpulse
from .contact import BodyContact
from .envelope import RefusedQuantity, require_quotable
from .solver import MPMRun

__all__ = [
    "BALL_REACH_TIER_NOTE",
    "DEFAULT_BALL_SECTORS",
    "MIN_BALL_SECTORS",
    "BallReachHistory",
    "BallSurfaceSectors",
    "BallTractionSample",
    "SandVersusClub",
    "ball_reach_history",
    "compare_sand_and_club",
    "resolve_ball_traction",
    "resolve_sectors",
]

DEFAULT_BALL_SECTORS = 12
"""Sectors the ball's in-plane surface is resolved into by default.

Thirty degrees each. Fine enough that "below the equator" resolves into
front, bottom and back of the bottom half rather than one lump; coarse
enough that a 4 mm grid puts more than one projected node in the sectors
that matter, so the shares are not counting single nodes."""

MIN_BALL_SECTORS = 4
"""Fewest sectors the surface may be resolved into.

Below four there is no resolution left that
:class:`~.ball.BallContactSplit` does not already give for free."""

BALL_REACH_TIER_NOTE = (
    "F1, status BEYOND_VALIDATION and no better: no published measurement "
    "exists for any quantity this tier produces (#8616), so its NASA-STD-7009B "
    "validation level is 0 of 4, and the fastest granular intrusion in the "
    f"published corpus is {MAX_VALIDATED_SPEED_M_S} m/s, which a bunker strike "
    "passes by an order of magnitude. Nothing here is a prediction of what a "
    "golf ball does."
)
"""The tier statement that travels with every quantity in this module."""

_DIMENSION = 2
_MIN_DIRECTION_NORM = 1e-12
_MIN_LEVER_M = 1e-12
_FULL_TURN_RAD = 2.0 * math.pi


def _unit_direction(direction: ArrayLike, *, name: str) -> NDArray[np.float64]:
    """A finite 2-vector normalised, or a raise saying why it could not be."""
    vector = np.asarray(direction, dtype=np.float64).reshape(-1)
    if vector.shape != (_DIMENSION,) or not np.all(np.isfinite(vector)):
        raise SolverInputError(f"{name} must be a finite 2-vector, got {direction!r}")
    norm = float(np.hypot(vector[0], vector[1]))
    if norm < _MIN_DIRECTION_NORM:
        raise SolverInputError(
            f"{name} has no length, so the ball has no near half to name; pass "
            "the club's travel direction"
        )
    return vector / norm


def _summed(splits: list[BallContactSplit], region: str) -> NDArray[np.float64]:
    """One named half of every step's split, added up over the run."""
    total: NDArray[np.float64] = np.zeros(_DIMENSION, dtype=np.float64)
    for split in splits:
        total = total + np.asarray(getattr(split, region), dtype=np.float64)
    return total


def _sector_edges(n_sectors: int) -> NDArray[np.float64]:
    """Bin edges in ``[0, 2 pi]``, counter-clockwise from ``+x``.

    The count must be **even** so that the equator falls on a boundary.
    An odd count puts the horizontal through the middle of a sector, and
    then the sector view and :class:`~.ball.BallContactSplit`'s
    below/above view of the same contact disagree about which half it
    was in -- two answers to one question, from one ledger.
    """
    count = int(n_sectors)
    if count < MIN_BALL_SECTORS:
        raise SolverInputError(
            f"n_sectors must be at least {MIN_BALL_SECTORS}, got {n_sectors!r}: "
            "below that the sectors say nothing BallContactSplit does not"
        )
    if count % 2 != 0:
        raise SolverInputError(
            f"n_sectors must be even, got {n_sectors!r}: an odd count puts the "
            "equator inside a sector, so the sector view and the below/above "
            "split would disagree about the same contact"
        )
    return np.linspace(0.0, _FULL_TURN_RAD, count + 1, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class BallSurfaceSectors:
    """The sand's impulse on the ball, binned around its in-plane surface.

    Every vector is an impulse **per unit out-of-plane width** applied
    **by the sand to the ball**, on a body that is an infinite cylinder
    rather than a sphere. The sectors partition the contact set, so
    :attr:`impulse_n_s_per_m` sums to the step's total exactly.

    The radial and tangential columns are taken in **each node's own**
    surface frame -- the outward normal through that contact -- rather
    than in the sector's mean direction, because the node frame is the
    one the sand actually pushed in. The consequence is stated rather
    than hidden: ``hypot(radial, tangential)`` recovers
    :attr:`magnitude_n_s_per_m` exactly only where a sector holds a
    single contact, and elsewhere differs by the spread of normals
    across the sector, which is at most one sector width.

    Attributes:
        edges_rad: ``(n + 1,)`` bin edges, counter-clockwise from ``+x``
            in the world plane. The equator is always an edge.
        impulse_n_s_per_m: ``(n, 2)`` impulse the sand delivered in each
            sector, per metre of out-of-plane width.
        radial_n_s_per_m: ``(n,)`` the **compressive** part, positive
            when the sand pushes toward the ball centre.
        tangential_n_s_per_m: ``(n,)`` the shear part, positive
            counter-clockwise about ``+y``.
        n_contacts: ``(n,)`` projected grid nodes in each sector.
        approach_direction: ``(2,)`` unit direction the club travels in,
            kept so the face side is recoverable from the bin index
            without the caller having to remember which way it came.
    """

    edges_rad: NDArray[np.float64]
    impulse_n_s_per_m: NDArray[np.float64]
    radial_n_s_per_m: NDArray[np.float64]
    tangential_n_s_per_m: NDArray[np.float64]
    n_contacts: NDArray[np.int64]
    approach_direction: NDArray[np.float64]

    @property
    def n_sectors(self) -> int:
        """How many sectors the surface was resolved into."""
        return int(self.impulse_n_s_per_m.shape[0])

    @property
    def is_qualitative(self) -> bool:
        """Always True. The resolution is a direction, never a load."""
        return True

    @property
    def total_n_s_per_m(self) -> NDArray[np.float64]:
        """``(2,)`` impulse over every sector, per metre of width."""
        return np.asarray(self.impulse_n_s_per_m).sum(axis=0)

    @property
    def magnitude_n_s_per_m(self) -> NDArray[np.float64]:
        """``(n,)`` magnitude of each sector's impulse, per metre of width."""
        return np.hypot(self.impulse_n_s_per_m[:, 0], self.impulse_n_s_per_m[:, 1])

    @property
    def fractions(self) -> NDArray[np.float64]:
        """``(n,)`` share of the delivered magnitude in each sector.

        Taken on magnitudes rather than on the resultant, so two sectors
        pushing against each other do not report as no contact at all.
        """
        magnitudes = self.magnitude_n_s_per_m
        total = float(magnitudes.sum())
        if total <= 0.0:
            return np.zeros_like(magnitudes)
        return magnitudes / total

    def total_force_n(self) -> NoReturn:
        """Refuse: an infinite cylinder has no width to multiply by.

        Raises:
            OutOfEnvelopeError: Always.
        """
        require_quotable(RefusedQuantity.OUT_OF_PLANE)
        raise AssertionError("unreachable")  # pragma: no cover

    def heel_toe_fraction(self) -> NoReturn:
        """Refuse: heel-toe is a direction plane strain does not have.

        Raises:
            OutOfEnvelopeError: Always.
        """
        require_quotable(RefusedQuantity.OUT_OF_PLANE)
        raise AssertionError("unreachable")  # pragma: no cover

    def lateral_distribution(self) -> NoReturn:
        """Refuse: there is no lateral axis to distribute anything over.

        Raises:
            OutOfEnvelopeError: Always.
        """
        require_quotable(RefusedQuantity.OUT_OF_PLANE)
        raise AssertionError("unreachable")  # pragma: no cover

    def summary(self) -> str:
        """A statement fit for a figure caption or a run manifest."""
        shares = self.fractions
        dominant = int(np.argmax(shares)) if shares.size else 0
        lower = math.degrees(float(self.edges_rad[dominant]))
        upper = math.degrees(float(self.edges_rad[dominant + 1]))
        return (
            f"ball surface sectors (qualitative, in-plane only): "
            f"{self.n_sectors} sectors, most of the impulse "
            f"({shares[dominant] * 100:.0f}%) in [{lower:.0f}, {upper:.0f}) deg "
            f"about the ball centre, total "
            f"{float(np.hypot(*self.total_n_s_per_m)):.4g} N.s per metre of "
            f"width over {int(np.asarray(self.n_contacts).sum())} grid nodes.\n  "
            + PLANE_STRAIN_BALL_NOTE
        )


def resolve_sectors(
    nodes: ContactImpulse,
    *,
    centre_m: ArrayLike,
    approach_direction: ArrayLike,
    n_sectors: int = DEFAULT_BALL_SECTORS,
) -> BallSurfaceSectors:
    """Bin one step's node-resolved ledger around the ball's surface.

    A node at the ball centre has no outward direction: it is assigned to
    the first sector by the ``atan2`` convention and contributes nothing
    to the radial and tangential columns. That is not a workaround for a
    case that cannot happen -- a body several cells across covers its own
    interior nodes -- but it costs nothing either, because an interior
    node holds no sand mass and so carries no impulse.

    Args:
        nodes: The projection's ledger, signed **body on sand**.
        centre_m: ``(2,)`` ball centre at the pose of this step.
        approach_direction: ``(2,)`` direction the club travels in.
        n_sectors: Sectors to resolve into. Even, at least
            :data:`MIN_BALL_SECTORS`.

    Returns:
        The resolution, signed **sand on ball**.

    Raises:
        SolverInputError: If the sector count is odd or too small, if the
            centre is not a finite 2-vector, or if the approach direction
            has no length.
    """
    edges = _sector_edges(n_sectors)
    count = edges.size - 1
    approach = _unit_direction(approach_direction, name="approach_direction")
    centre = np.asarray(centre_m, dtype=np.float64).reshape(-1)
    if centre.shape != (_DIMENSION,) or not np.all(np.isfinite(centre)):
        raise SolverInputError(f"centre_m must be a finite 2-vector, got {centre_m!r}")

    impulse = np.zeros((count, _DIMENSION), dtype=np.float64)
    radial = np.zeros(count, dtype=np.float64)
    tangential = np.zeros(count, dtype=np.float64)
    counts = np.zeros(count, dtype=np.int64)
    on_ball = -np.asarray(nodes.impulse_n_s, dtype=np.float64).reshape(-1, _DIMENSION)
    if on_ball.shape[0] == 0:
        return BallSurfaceSectors(edges, impulse, radial, tangential, counts, approach)

    lever = np.asarray(nodes.position_m, dtype=np.float64).reshape(-1, 2) - centre
    angle = np.mod(np.arctan2(lever[:, 1], lever[:, 0]), _FULL_TURN_RAD)
    index = np.minimum((angle / (_FULL_TURN_RAD / count)).astype(np.int64), count - 1)

    span = np.hypot(lever[:, 0], lever[:, 1])
    outward = np.zeros_like(lever)
    resolved = span > _MIN_LEVER_M
    outward[resolved] = lever[resolved] / span[resolved, None]
    # Compression is positive: the sand pushes toward the centre.
    node_radial = -np.einsum("ij,ij->i", on_ball, outward)
    node_tangential = on_ball[:, 0] * -outward[:, 1] + on_ball[:, 1] * outward[:, 0]

    np.add.at(impulse, index, on_ball)
    np.add.at(radial, index, node_radial)
    np.add.at(tangential, index, node_tangential)
    np.add.at(counts, index, 1)
    return BallSurfaceSectors(edges, impulse, radial, tangential, counts, approach)


@dataclass(frozen=True, slots=True)
class BallTractionSample:
    """What the sand delivered to the ball in one step.

    Attributes:
        time_s: Simulation time at the end of the step.
        traction_n_per_m: ``(2,)`` force the sand exerts on the ball, per
            metre of out-of-plane width.
        impulse_n_s_per_m: ``(2,)`` impulse over the step, per metre of
            width. Read straight off the ledger rather than from
            ``traction * dt``, so the history's total is the ledger's own
            sum to the last bit.
        split: The below-equator / face-side split, qualitative and
            in-plane, signed the solver's way (body on sand).
        sectors: The same contact set resolved around the surface,
            signed sand on ball.
        n_contacts: Grid nodes the ball projected this step.
    """

    time_s: float
    traction_n_per_m: NDArray[np.float64]
    impulse_n_s_per_m: NDArray[np.float64]
    split: BallContactSplit
    sectors: BallSurfaceSectors
    n_contacts: int

    @property
    def reached(self) -> bool:
        """Whether any sand reached the ball at all this step."""
        return self.n_contacts > 0

    @property
    def traction_magnitude_n_per_m(self) -> float:
        """Magnitude of the traction, per metre of out-of-plane width."""
        return float(np.hypot(self.traction_n_per_m[0], self.traction_n_per_m[1]))


def resolve_ball_traction(
    contact: BodyContact,
    ball: BallSection,
    *,
    time_s: float,
    time_step_s: float,
    approach_direction: ArrayLike,
    n_sectors: int = DEFAULT_BALL_SECTORS,
) -> BallTractionSample:
    """Read one step's ledger as traction on the ball.

    Args:
        contact: The ball's ledger for this step, carrying the
            node-resolved impulse on
            :attr:`~.contact.BodyContact.nodes`.
        ball: The ball **at the pose this step was taken from**.
        time_s: Simulation time at the end of the step.
        time_step_s: The step, for the traction the impulse implies.
        approach_direction: ``(2,)`` direction the club travels in.
        n_sectors: Sectors to resolve the surface into.

    Returns:
        The sample.

    Raises:
        SolverInputError: If the step is not positive, or if the sector
            count or the approach direction is malformed.
    """
    if not math.isfinite(time_step_s) or time_step_s <= 0.0:
        raise SolverInputError(f"time_step_s must be positive, got {time_step_s!r}")
    nodes = contact.nodes
    return BallTractionSample(
        time_s=float(time_s),
        traction_n_per_m=np.asarray(contact.force_n_per_m, dtype=np.float64),
        impulse_n_s_per_m=-np.asarray(contact.impulse_on_sand_n_s, dtype=np.float64),
        split=ball.split_contact(nodes, approach_direction=approach_direction),
        sectors=resolve_sectors(
            nodes,
            centre_m=ball.centre_m,
            approach_direction=approach_direction,
            n_sectors=n_sectors,
        ),
        n_contacts=contact.n_contacts,
    )


@dataclass(frozen=True)
class BallReachHistory:
    """The whole time history of what reached the ball, with its tier.

    Attributes:
        samples: One :class:`BallTractionSample` per marched step, in
            order.
        time_step_s: The step the march ran at.
        verdict: The F1 validity verdict this history was produced
            under. Carried on the value, not quoted in a caption, so a
            number cannot be separated from the tier that made it.
        ball_radius_m: Radius of the circle the section stood for.
    """

    samples: tuple[BallTractionSample, ...]
    time_step_s: float
    verdict: ValidityVerdict
    ball_radius_m: float

    def __post_init__(self) -> None:
        if not self.samples:
            raise SolverInputError(
                "a ball-reach history with no samples has nothing to report; a "
                "zero-step history would return a zero traction that reads as "
                "'the sand never arrived'"
            )

    @property
    def fidelity_tier(self) -> FidelityTier:
        """Always F1. This ledger only exists inside the F1 solve."""
        return FidelityTier.F1

    @property
    def is_qualitative(self) -> bool:
        """Always True for the region splits carried here."""
        return True

    @property
    def first_arrival_s(self) -> float | None:
        """When sand first reached the ball, or ``None`` if it never did."""
        for sample in self.samples:
            if sample.reached:
                return sample.time_s
        return None

    def loading_onset_s(self, *, fraction_of_peak: float) -> float | None:
        """When the traction first passed a stated share of its own peak.

        :attr:`first_arrival_s` answers "when did sand touch the ball",
        and for a ball lying in a bunker the answer is "before the swing
        started" -- it is resting *on* sand. The question #8712 actually
        asks is when the sand the club moved gets there, and that has no
        answer without a threshold, so the threshold is an argument
        rather than a constant: a caller cannot be handed an onset whose
        definition it did not choose.

        Args:
            fraction_of_peak: Share of :attr:`peak_traction_n_per_m` the
                traction must exceed, in ``(0, 1]``.

        Returns:
            The first sample time above the threshold, or ``None`` if the
            ball was never loaded at all.

        Raises:
            SolverInputError: If the fraction is outside ``(0, 1]``.
        """
        share = float(fraction_of_peak)
        if not math.isfinite(share) or not 0.0 < share <= 1.0:
            raise SolverInputError(
                f"fraction_of_peak must lie in (0, 1], got {fraction_of_peak!r}"
            )
        threshold = share * self.peak_traction_n_per_m
        if threshold <= 0.0:
            return None
        for sample in self.samples:
            if sample.traction_magnitude_n_per_m >= threshold:
                return sample.time_s
        return None

    @property
    def peak_traction_n_per_m(self) -> float:
        """Largest traction magnitude over the run, per metre of width."""
        return float(max(sample.traction_magnitude_n_per_m for sample in self.samples))

    @property
    def peak_traction_time_s(self) -> float:
        """When that peak occurred, on the run's own clock."""
        peak = max(self.samples, key=lambda sample: sample.traction_magnitude_n_per_m)
        return peak.time_s

    @property
    def total_impulse_n_s_per_m(self) -> NDArray[np.float64]:
        """``(2,)`` impulse the sand delivered in total, per metre of width."""
        return np.sum([sample.impulse_n_s_per_m for sample in self.samples], axis=0)

    @property
    def total_impulse_magnitude_n_s_per_m(self) -> float:
        """Magnitude of :attr:`total_impulse_n_s_per_m`."""
        return float(np.hypot(*self.total_impulse_n_s_per_m))

    @property
    def aggregate_split(self) -> BallContactSplit:
        """The below/above and near/far halves summed over the whole run."""
        splits = [sample.split for sample in self.samples]
        return BallContactSplit(
            below_equator_n_s=_summed(splits, "below_equator_n_s"),
            above_equator_n_s=_summed(splits, "above_equator_n_s"),
            face_side_n_s=_summed(splits, "face_side_n_s"),
            far_side_n_s=_summed(splits, "far_side_n_s"),
            total_n_s=_summed(splits, "total_n_s"),
            n_contacts=int(sum(sample.n_contacts for sample in self.samples)),
        )

    @property
    def below_equator_fraction(self) -> float:
        """Share of the run's impulse magnitude that landed below the equator."""
        return self.aggregate_split.below_equator_fraction

    @property
    def face_side_fraction(self) -> float:
        """Share that landed on the half the club arrives from."""
        return self.aggregate_split.face_side_fraction

    def time_history_s(self) -> NDArray[np.float64]:
        """``(n_steps,)`` step end times."""
        return np.array([sample.time_s for sample in self.samples])

    def traction_history_n_per_m(self) -> NDArray[np.float64]:
        """``(n_steps, 2)`` traction on the ball, per metre of width."""
        return np.array([sample.traction_n_per_m for sample in self.samples])

    def sector_impulse_n_s_per_m(self) -> NDArray[np.float64]:
        """``(n_sectors, 2)`` impulse per sector, summed over the run."""
        return np.sum(
            [sample.sectors.impulse_n_s_per_m for sample in self.samples], axis=0
        )

    def total_force_on_ball_n(self) -> NoReturn:
        """Refuse: an infinite cylinder has no width to multiply by.

        Raises:
            OutOfEnvelopeError: Always.
        """
        require_quotable(RefusedQuantity.OUT_OF_PLANE)
        raise AssertionError("unreachable")  # pragma: no cover

    def heel_toe_history(self) -> NoReturn:
        """Refuse: heel-toe is a direction plane strain does not have.

        Raises:
            OutOfEnvelopeError: Always.
        """
        require_quotable(RefusedQuantity.OUT_OF_PLANE)
        raise AssertionError("unreachable")  # pragma: no cover

    def launch_velocity_m_s(self) -> NoReturn:
        """Refuse: ball launch stays on F0's momentum-transfer path (#8657).

        This module says what reaches the ball. Turning that into a
        launch is a different model, and it is not this tier's.

        Raises:
            OutOfEnvelopeError: Always.
        """
        require_quotable(RefusedQuantity.BALL_LAUNCH)
        raise AssertionError("unreachable")  # pragma: no cover

    def summary(self) -> str:
        """A statement fit for a run manifest, tier included."""
        arrival = self.first_arrival_s
        arrived = "never" if arrival is None else f"{arrival * 1e3:.4g} ms"
        return (
            f"sand reaching the ball ({self.fidelity_tier.value}, "
            f"{self.verdict.status.value}): first arrival {arrived}, peak "
            f"{self.peak_traction_n_per_m:.4g} N per metre of width at "
            f"{self.peak_traction_time_s * 1e3:.4g} ms, total "
            f"{self.total_impulse_magnitude_n_s_per_m:.4g} N.s per metre of "
            f"width, {self.below_equator_fraction * 100:.0f}% of it below the "
            f"equator and {self.face_side_fraction * 100:.0f}% on the near "
            f"half (qualitative, in-plane only).\n  "
            + PLANE_STRAIN_BALL_NOTE
            + "\n  "
            + BALL_REACH_TIER_NOTE
        )


def _require_body_index(run: MPMRun, index: int, *, name: str) -> int:
    """Check a body index against the run, naming the argument that is wrong."""
    available = run.steps[0].n_bodies
    if not 0 <= int(index) < available:
        raise SolverInputError(
            f"{name} {index!r} is outside the {available} body/bodies this run "
            "marched; a ledger read at the wrong index would attribute one "
            "body's load to another and still look plausible"
        )
    return int(index)


def ball_reach_history(
    run: MPMRun,
    ball: BallSection,
    *,
    verdict: ValidityVerdict,
    body_index: int = 1,
    approach_direction: ArrayLike | None = None,
    n_sectors: int = DEFAULT_BALL_SECTORS,
) -> BallReachHistory:
    """Read a whole march's ledger as the history of what reached the ball.

    The ball's pose is **replayed**, not stored: extra bodies in a march
    are prescribed -- they are advanced, never accelerated -- so applying
    the same :meth:`~.ball.BallSection.advanced` the march applied
    reproduces the pose every step was taken at, exactly. That is why the
    ball passed here must be the one the march *started* from.

    Args:
        run: The march, from :meth:`~.solver.PlaneStrainMPMSolver.march_bodies`
            or off :attr:`~.wholeshot.F1ShotResult.run`.
        ball: The ball at its **starting** pose.
        verdict: The F1 verdict the run was produced under, carried onto
            the history.
        body_index: Which body of the step the ball was. One by
            convention: the head is the primary body.
        approach_direction: ``(2,)`` direction the club travels in.
            ``None`` takes the primary body's own velocity, which is the
            only direction in the run that is not an invention.
        n_sectors: Sectors to resolve the surface into.

    Returns:
        The history.

    Raises:
        SolverInputError: If the index is outside the run's bodies, or if
            no approach direction is given and the run has no primary
            body to take one from.
    """
    index = _require_body_index(run, body_index, name="body_index")
    if approach_direction is None:
        primary = run.section
        if primary is None:
            raise SolverInputError(
                "this run had no primary body, so there is no club direction to "
                "name the ball's near half by; pass approach_direction"
            )
        approach_direction = primary.velocity_m_s
    direction = _unit_direction(approach_direction, name="approach_direction")

    step_s = run.time_step_s
    posed = ball
    samples: list[BallTractionSample] = []
    for step in run.steps:
        samples.append(
            resolve_ball_traction(
                step.body_contacts[index],
                posed,
                time_s=step.time_s,
                time_step_s=step_s,
                approach_direction=direction,
                n_sectors=n_sectors,
            )
        )
        posed = posed.advanced(step_s)
    return BallReachHistory(tuple(samples), step_s, verdict, ball.radius_m)


@dataclass(frozen=True, slots=True)
class SandVersusClub:
    """What the sand delivers to the ball against what the club delivers.

    The physically interesting question epic #8699 was built for, kept to
    the two forms this tier may answer it in: a **dimensionless share**
    and a **pair of timings**. Both absolute forces are refused, for
    reasons that are different and both binding --
    :meth:`club_force_n` because ADR-0033 refuses F1 for club force at
    all, and :meth:`ball_force_n` because an infinite cylinder has no
    out-of-plane width to multiply a flux by.

    Attributes:
        club_impulse_on_sand_n_s_per_m: ``(2,)`` impulse the club put
            into the sand over the run, per metre of width.
        sand_impulse_on_ball_n_s_per_m: ``(2,)`` impulse the sand
            delivered to the ball over the same run, same units, same
            ledger.
        club_peak_time_s: When the club's load peaked.
        ball_peak_time_s: When the ball's traction peaked.
        club_first_contact_s: When the club first touched sand, or
            ``None``.
        ball_first_arrival_s: When sand first reached the ball, or
            ``None``.
        verdict: The verdict both sides were produced under.
    """

    club_impulse_on_sand_n_s_per_m: NDArray[np.float64]
    sand_impulse_on_ball_n_s_per_m: NDArray[np.float64]
    club_peak_time_s: float
    ball_peak_time_s: float
    club_first_contact_s: float | None
    ball_first_arrival_s: float | None
    verdict: ValidityVerdict

    @property
    def transmitted_fraction(self) -> float:
        """Share of the club's impulse magnitude that arrives at the ball.

        A ratio of two per-unit-width terms from the *same* momentum
        ledger, so the declared effective width and the plane-strain
        geometry divide out of it. It is still qualitative: the ball is
        the wrong three-dimensional shape and the sand constants are
        borrowed.
        """
        delivered = float(np.hypot(*self.club_impulse_on_sand_n_s_per_m))
        if delivered <= 0.0:
            return 0.0
        return float(np.hypot(*self.sand_impulse_on_ball_n_s_per_m)) / delivered

    @property
    def arrival_lag_s(self) -> float | None:
        """How long after the club entered the sand the ball first felt it."""
        if self.club_first_contact_s is None or self.ball_first_arrival_s is None:
            return None
        return self.ball_first_arrival_s - self.club_first_contact_s

    @property
    def peak_lag_s(self) -> float:
        """How long after the club's peak the ball's peak arrives."""
        return self.ball_peak_time_s - self.club_peak_time_s

    def club_force_n(self) -> NoReturn:
        """Refuse: ADR-0033 does not let F1 be quoted for club force.

        Raises:
            OutOfEnvelopeError: Always.
        """
        require_quotable(RefusedQuantity.CLUB_FORCE)
        raise AssertionError("unreachable")  # pragma: no cover

    def ball_force_n(self) -> NoReturn:
        """Refuse: an infinite cylinder has no width to multiply by.

        Raises:
            OutOfEnvelopeError: Always.
        """
        require_quotable(RefusedQuantity.OUT_OF_PLANE)
        raise AssertionError("unreachable")  # pragma: no cover

    def summary(self) -> str:
        """A statement fit for a run manifest, tier included."""
        lag = self.arrival_lag_s
        lag_text = "no arrival" if lag is None else f"{lag * 1e3:.4g} ms after entry"
        return (
            f"sand to the ball against club to the sand "
            f"({self.verdict.status.value}): the ball receives "
            f"{self.transmitted_fraction * 100:.2f}% of the club's delivered "
            f"impulse magnitude, "
            f"{float(np.hypot(*self.sand_impulse_on_ball_n_s_per_m)):.4g} against "
            f"{float(np.hypot(*self.club_impulse_on_sand_n_s_per_m)):.4g} N.s per "
            f"metre of width; first felt {lag_text}, peaking "
            f"{self.peak_lag_s * 1e3:.4g} ms after the club's own peak. Absolute "
            "force is refused on both sides.\n  " + BALL_REACH_TIER_NOTE
        )


def compare_sand_and_club(
    run: MPMRun, history: BallReachHistory, *, club_index: int = 0
) -> SandVersusClub:
    """Compare the ball's side of one march's ledger against the club's.

    Args:
        run: The march both sides were read from. Using one run is the
            point: a ratio across two solves would compare two different
            beds.
        history: The ball's side, from :func:`ball_reach_history`.
        club_index: Which body of the step the club was. Zero by
            convention: the head is the primary body.

    Returns:
        The comparison.

    Raises:
        SolverInputError: If the index is outside the run's bodies.
    """
    index = _require_body_index(run, club_index, name="club_index")
    contacts = [step.body_contacts[index] for step in run.steps]
    times = [step.time_s for step in run.steps]
    magnitudes = [float(np.hypot(*contact.force_n_per_m)) for contact in contacts]
    touched = [
        time for time, spot in zip(times, contacts, strict=True) if spot.n_contacts
    ]
    return SandVersusClub(
        club_impulse_on_sand_n_s_per_m=np.sum(
            [contact.impulse_on_sand_n_s for contact in contacts], axis=0
        ),
        sand_impulse_on_ball_n_s_per_m=history.total_impulse_n_s_per_m,
        club_peak_time_s=times[int(np.argmax(magnitudes))],
        ball_peak_time_s=history.peak_traction_time_s,
        club_first_contact_s=touched[0] if touched else None,
        ball_first_arrival_s=history.first_arrival_s,
        verdict=history.verdict,
    )
