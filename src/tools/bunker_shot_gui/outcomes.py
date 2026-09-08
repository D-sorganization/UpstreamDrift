"""The workbench's result value objects (issues #8618, #9243).

Split out of :mod:`~src.tools.bunker_shot_gui.model` so that the orchestration
(``WorkbenchModel``, which runs solvers) and the things it produces are two
files rather than one over the module-size budget. Every name here is
re-exported from ``model``, so existing ``from .model import ShotOutcome``
imports are unaffected.

What these types enforce, and why it is here rather than in a caller
--------------------------------------------------------------------

They are the boundary that a display cannot get around. ADR-0032's refusal
rule (a refused shot carries no number), issue #8657's pairing rule (a carry
never travels without the verdict it may be quoted under) and issue #9243's
banding rule (a number that has a band never appears without it, and a band
around a different number is refused) are all constructor invariants here.
A view that wanted to show a bare carry beside a ``REFUSED`` verdict would
have to build an object that cannot be built.

:class:`WorkbenchComparison` is where the last of those bites hardest. Its
:attr:`~WorkbenchComparison.winner` is ``None`` whenever the two designs'
bands overlap, so a caller that reads a name off every comparison gets nothing
to print rather than a leader the model cannot support.
"""

from __future__ import annotations

from dataclasses import dataclass

from bunkershot3d.geometry import DeliveredGeometry, StationCamber, WedgeGeometry
from bunkershot3d.metrics import DigSkidResult, DivotMetrics, HeadLoadMetrics
from bunkershot3d.sand import SandState
from bunkershot3d.solvers import (
    EnvelopeStatus,
    FidelityTier,
    ValidityVerdict,
    worst_of,
)
from bunkershot3d.study.comparison import DesignComparison
from bunkershot3d.study.ranking import BandedRanking
from bunkershot3d.vandv.band import ConsistencyBand

from .bridge import SoleLoadMap
from .design import WedgeDesign
from .field import ContactPatch, SoleLoadField
from .shot3d import ShotScene
from .traces import ShotTraces
from .uncertainty import PlayabilityOutcome

__all__ = [
    "DesignEvaluation",
    "ShotOutcome",
    "WorkbenchComparison",
]


@dataclass(frozen=True)
class ShotOutcome:
    """One shot: its verdict first, its numbers only if there are any.

    Attributes:
        verdict: The validity statement for the whole trace.
        fidelity_tier: Which rung of the ADR-0032 ladder produced it.
        refused: True when the solver declined to answer. Every numeric
            field is ``None`` in that case, by construction.
        delivered: Effective loft, bounce and aim at impact.
        peak_force_n: Largest resultant sand force.
        impulse_n_s: Magnitude of the time-integrated sand force.
        entry_speed_mps: Head speed at the first sample.
        exit_speed_mps: Head speed at the last sample.
        max_depth_m: Deepest submerged point.
        contact_duration_s: Time with at least one engaged element.
        peak_inertial_fraction: Largest share of force carried by the
            dynamic term. ADR-0032 predicts roughly 0.9 at 25 m/s.
        runtime_s: Wall-clock cost of the integration.
        loads: Peak/mean head loads, when the trace supports them.
        divot: Divot geometry, when the sole entered and left the sand.
        dig_skid: The dig-versus-skid discriminator.
        sole_load: The bounce-utilisation map: the strike binned onto a
            12x12 sole grid and summed over time.
        sole_field: The same load *before* either reduction -- per element,
            per sample, with the depth-linear and inertial terms separated
            (issue #8705).
        contact_patch: The engaged element set followed through the shot
            (issue #8707).
        scene: The 3-D scene -- pose, free surface and swept divot section --
            for the animated view (issue #8706).
        traces: The scalar traces and the per-sample validity band, on the
            same time axis as the scene and the field (issue #8708).
        carry_m: Carry from the splash and flight models.
        carry_verdict: The validity statement the carry may be quoted under.
            Present whenever ``carry_m`` is, and absent whenever it is not.
        carry_band: Carry at the two edges of the accelerated-mass interval
            (issue #9243), or ``None`` when there was no interval to
            propagate. The centre is ``carry_m``.
        carry_band_reasons: What had to be said while propagating that band.
        unavailable: One line per metric that could not be computed, with
            the reason. Empty when everything was.
    """

    verdict: ValidityVerdict
    fidelity_tier: FidelityTier
    refused: bool
    delivered: DeliveredGeometry
    peak_force_n: float | None = None
    impulse_n_s: float | None = None
    entry_speed_mps: float | None = None
    exit_speed_mps: float | None = None
    max_depth_m: float | None = None
    contact_duration_s: float | None = None
    peak_inertial_fraction: float | None = None
    runtime_s: float | None = None
    loads: HeadLoadMetrics | None = None
    divot: DivotMetrics | None = None
    dig_skid: DigSkidResult | None = None
    sole_load: SoleLoadMap | None = None
    sole_field: SoleLoadField | None = None
    contact_patch: ContactPatch | None = None
    scene: ShotScene | None = None
    traces: ShotTraces | None = None
    carry_m: float | None = None
    carry_verdict: ValidityVerdict | None = None
    carry_band: ConsistencyBand | None = None
    carry_band_reasons: tuple[str, ...] = ()
    unavailable: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Enforce the refusal rule and the carry-verdict rule.

        Raises:
            ValueError: If a refused outcome carries any number, or if a carry
                number arrives without the verdict it may be quoted under.
                ADR-0032 makes the first the single most important behaviour
                of the tier and issue #8657 makes the second the condition on
                displaying carry at all, so both are a ``raise`` rather than a
                contract decorator -- ``DBC_LEVEL=off`` must not switch either
                off.
        """
        if (self.carry_m is None) != (self.carry_verdict is None):
            raise ValueError(
                "a carry number and its validity verdict travel together. No "
                "published measurement of ball speed or launch angle out of "
                "sand exists (issue #8616), so a carry shown on its own would "
                "read as though it had been measured"
            )
        if not self.refused:
            return
        numbers = (
            self.peak_force_n,
            self.impulse_n_s,
            self.max_depth_m,
            self.carry_m,
            self.carry_band,
            self.sole_field,
            self.contact_patch,
            self.scene,
            self.traces,
        )
        if any(value is not None for value in numbers):
            raise ValueError(
                "a refused shot must not carry a force, a depth, a carry, a "
                "load field or a 3-D scene: 3D-RFT declined this query, and "
                "animating a head through sand beside a REFUSED verdict is "
                "exactly what ADR-0032 forbids"
            )

    @property
    def status(self) -> EnvelopeStatus:
        """How much of the answer may be believed."""
        return self.verdict.status

    @property
    def is_within_stated_envelope(self) -> bool:
        """True only inside 3D-RFT's own published limits."""
        return self.verdict.is_within_stated_envelope


@dataclass(frozen=True)
class DesignEvaluation:
    """Everything the workbench knows about one candidate sole.

    Attributes:
        design: The designer's inputs.
        geometry: The resolved design vector.
        sand: The sand state the shot was run in.
        shot: The nominal shot.
        playability: The playability window over the delivery sweep.
        effective_camber_area_m2: The camber area the lofted head actually
            carries. A sole cannot host an arbitrarily large camber for its
            width and bounce, so the declared area in :attr:`geometry` is
            fitted to what the sole admits; carrying the realised value here
            is what lets the report state both (issue #8698).
        camber_stations: The lofter's per-station camber account, heel to
            toe. Carried because :attr:`effective_camber_area_m2` alone
            cannot answer whether the sole was substituted: it is the
            *declared* width's number, and the relieved heel and toe
            stations have their own, narrower bands.
    """

    design: WedgeDesign
    geometry: WedgeGeometry
    sand: SandState
    shot: ShotOutcome
    playability: PlayabilityOutcome
    effective_camber_area_m2: float
    camber_stations: tuple[StationCamber, ...]

    @property
    def aggregate_camber_was_clamped(self) -> bool:
        """Whether the *declared* camber area itself had to be substituted.

        Narrowly scoped, and ``False`` on the shipped presets even when
        stations were refitted; see :attr:`any_camber_was_clamped`.
        """
        return self.effective_camber_area_m2 != self.geometry.sole_camber_area_m2

    @property
    def clamped_camber_stations(self) -> tuple[StationCamber, ...]:
        """Every spanwise station refitted to its own constructible band."""
        return tuple(station for station in self.camber_stations if station.was_clamped)

    @property
    def any_camber_was_clamped(self) -> bool:
        """Whether any camber substitution occurred, aggregate or per station.

        Cannot read ``False`` while :attr:`clamped_camber_stations` holds
        anything, which is the property that stops a one-boolean caller being
        told a substituted sole was built as declared (issue #8698).
        """
        return self.aggregate_camber_was_clamped or bool(self.clamped_camber_stations)

    @property
    def verdict(self) -> ValidityVerdict:
        """The validity statement the nominal shot carried."""
        return self.shot.verdict

    @property
    def status(self) -> EnvelopeStatus:
        """How much of this evaluation may be believed."""
        return self.shot.status


@dataclass(frozen=True)
class WorkbenchComparison:
    """Two candidate soles, side by side, with the uncertainty attached.

    Attributes:
        left: The first design's evaluation.
        right: The second design's evaluation.
        ranking: Bootstrap ranking on carry error over the shared delivery
            sweep, or ``None`` when the two designs share too few answerable
            grid points to say anything. It sees **only** the spread across
            delivery conditions; :attr:`banded` is the one that carries the
            model-form width as well.
        ranking_unavailable_reason: Why, when ``ranking`` is ``None``.
        shared_points: Number of grid points both designs answered.
        banded: The comparison over the whole uncertainty budget -- sampling,
            model form and numerics kept apart -- whose verdict is one of A
            better, B better, or **indistinguishable at this uncertainty**
            (issue #9243). ``None`` on the same condition as ``ranking``.
    """

    left: DesignEvaluation
    right: DesignEvaluation
    ranking: DesignComparison | None = None
    ranking_unavailable_reason: str = ""
    shared_points: int = 0
    banded: BandedRanking | None = None

    @property
    def separated(self) -> bool:
        """True when the ranking distinguishes the leader from the rival.

        Reads the **banded** verdict when there is one, so that a design whose
        bootstrap interval is tight but whose model-form band swallows the
        difference is not reported as separated. Falls back to the bootstrap
        alone only when no budget could be built.
        """
        if self.banded is not None:
            return self.banded.is_decided
        return self.ranking is not None and self.ranking.is_separated()

    @property
    def winner(self) -> str | None:
        """The better design, or ``None`` when the two cannot be told apart.

        ``None`` is the point: a caller that reads a name off every comparison
        cannot accidentally present a tie as a win. The bootstrap's own
        ``best`` is never consulted here, because it always names somebody.
        """
        return None if self.banded is None else self.banded.winner

    @property
    def verdict_statement(self) -> str:
        """The comparison in words, or why there is not one."""
        if self.banded is None:
            return (
                self.ranking_unavailable_reason or "the two designs were not compared"
            )
        return self.banded.statement()

    @property
    def worst_status(self) -> EnvelopeStatus:
        """The worse of the two verdicts.

        A comparison is only as trustworthy as its least trustworthy half, so
        this is what the verdict banner shows. Combining the verdicts here
        rather than in the view keeps the ordering of the statuses in the one
        place that defines it.
        """
        return worst_of((self.left.verdict, self.right.verdict)).status
