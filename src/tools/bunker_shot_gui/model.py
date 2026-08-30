"""Headless workbench model for the BunkerShot3D designer GUI (issue #8618).

The widget in :mod:`src.tools.bunker_shot_gui.gui` renders what this module
computes and does no arithmetic of its own. Everything here runs without a
display, without Qt, and without a GUI event loop, so the same model can back
the Tauri/React app (epic #7462) and can be tested on a machine where PyQt6
does not import.

What it wires together
----------------------

======================================= ==========================================
Concern                                 Source
======================================= ==========================================
Sole geometry (W2)                      :mod:`bunkershot3d.geometry`
Sand state (W3)                         :mod:`bunkershot3d.sand`
F0 solver, ~ms/shot (ADR-0032)          :mod:`bunkershot3d.solvers`
Designer metrics (W7)                   :mod:`bunkershot3d.metrics`
Ball launch and carry                   :mod:`bunkershot3d.ball` + the shared
                                        ball-flight simulator
Ranking with uncertainty                :mod:`bunkershot3d.study.comparison`
Trace and per-element plumbing          :mod:`src.tools.bunker_shot_gui.bridge`
======================================= ==========================================

The refusal rule
----------------

ADR-0032 requires that a solver used outside its calibrated envelope say so
rather than return a plausible number, and a greenside bunker shot sits about
60x outside 3D-RFT's stated Froude limit. The solver therefore runs under
:attr:`~bunkershot3d.solvers.envelope.RefusalPolicy.STRICT`, and a refusal
arrives here as an exception that becomes a
:class:`ShotOutcome` **with no force, no depth and no carry** -- only the
verdict. There is deliberately no code path that reports a number alongside a
``REFUSED`` verdict.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from bunkershot3d.ball import (
    BallLie,
    BunkerShotState,
    SandDelivery,
    compute_bunker_launch,
)
from bunkershot3d.geometry import (
    DeliveredGeometry,
    StationCamber,
    WedgeGeometry,
    deliver_wedge,
)
from bunkershot3d.metrics import (
    ACCELERATED_MASS_CONSISTENCY_REASON,
    DigSkidResult,
    DivotMetrics,
    HeadLoadMetrics,
    HeadModel,
    PlayabilityAxis,
    PlayabilityWindow,
    StrikeScene,
    StrikeTrace,
    bounce_utilisation,
    dig_vs_skid,
    divot_metrics,
    head_load_metrics,
    playability_window,
)
from bunkershot3d.sand import SandState, firmness_pa_from_kg_per_cm2
from bunkershot3d.solvers import (
    DRFTSolver,
    EnvelopeStatus,
    FidelityTier,
    HeadKinematics,
    MaterialResponse,
    OutOfEnvelopeError,
    ShotResult,
    ShotSettings,
    ValidityVerdict,
    simulate_shot,
    worst_of,
)
from bunkershot3d.study.comparison import DesignComparison, compare_designs
from src.shared.python.core.contracts import require

from .bridge import (
    HeadBuild,
    SoleLoadMap,
    build_head,
    entry_kinematics,
    sole_load_field,
    sole_load_map,
    strike_trace,
    validity_band,
)
from .design import (
    FIRMNESS_RANGE_KG_PER_CM2,
    SandCondition,
    SolverSetup,
    SwingSetup,
    WedgeDesign,
    WorkbenchInputError,
)
from .field import ContactPatch, SoleLoadField, contact_patch
from .shot3d import ShotScene, shot_scene
from .traces import ShotTraces, shot_traces

__all__ = [
    "ATTACK_ANGLE_SWEEP_DEG",
    "COMPARISON_SEED",
    "CarryEstimate",
    "ContactPatch",
    "DesignEvaluation",
    "HeadBuild",
    "PlayabilityOutcome",
    "ShotOutcome",
    "ShotScene",
    "ShotTraces",
    "SoleLoadField",
    "SoleLoadMap",
    "WorkbenchComparison",
    "WorkbenchModel",
]

logger = logging.getLogger(__name__)

_MetricT = TypeVar("_MetricT")

ATTACK_ANGLE_SWEEP_DEG = (-12.0, -2.0)
"""The registered attack-angle sweep: the largest single delivery term."""

COMPARISON_SEED = 20260814
"""Fixed entropy for the A/B bootstrap, so a comparison is reproducible."""

_MAX_FLIGHT_TIME_S = 10.0


@dataclass(frozen=True, slots=True)
class CarryEstimate:
    """A carry number and the verdict it may only ever be quoted with.

    Issue #8657: carry is derived from the impulse the solver delivered and
    the divot mass the metrics layer measured, through an **uncalibrated**
    transfer efficiency, and there is no published measurement of ball speed
    or launch angle out of sand to calibrate it against (issue #8616). Pairing
    the number with its verdict in one value object is what stops the two
    being separated on the way to a display.

    Attributes:
        carry_m: Carry distance [m].
        verdict: The shot's verdict combined with the launch model's own,
            never better than ``BEYOND_VALIDATION``.
    """

    carry_m: float
    verdict: ValidityVerdict


def _strike_scene(ball_depth_m: float) -> StrikeScene:
    """Return the flat-surface scene the W7 metrics are measured against.

    Args:
        ball_depth_m: How far the ball centre sits below the sand surface.

    Returns:
        The scene.
    """
    return StrikeScene(
        sand_surface_height_m=0.0,
        ball_position_m=np.array(
            [0.0, 0.0, BallLie(depth_m=ball_depth_m).center_z_m()], dtype=np.float64
        ),
        travel_axis=np.array([1.0, 0.0, 0.0], dtype=np.float64),
    )


def _sand_delivery(
    result: ShotResult, divot: DivotMetrics, sand: SandState
) -> SandDelivery:
    """Bundle what the solver and the metrics layer measured about one strike.

    The mass handed over is the **accelerated** mass, not the swept prism:
    dividing the delivered impulse by the prism implied sand leaving faster
    than the head that threw it (issue #8659). The interval it was drawn from
    travels with it, so the launch can report how wide the denominator was.

    Args:
        result: The F0 shot.
        divot: The divot the same shot cut.
        sand: The bed it was struck in, supplying the relative density the
            sand-to-ball transfer efficiency depends on (issue #8704).

    Returns:
        The delivery the ball model derives launch from.
    """
    accelerated = divot.accelerated_mass
    return SandDelivery(
        impulse_n_s=float(np.linalg.norm(result.impulse_n_s)),
        displaced_mass_kg=accelerated.central_kg,
        displaced_mass_bounds_kg=accelerated.bounds_kg,
        displaced_mass_reason=ACCELERATED_MASS_CONSISTENCY_REASON,
        contact_duration_s=result.contact_duration_s,
        entry_speed_m_s=result.entry_speed_m_s,
        exit_speed_m_s=result.exit_speed_m_s,
        bed_relative_density=sand.relative_density,
        verdict=result.verdict,
    )


def _measured_divot(divot: DivotMetrics | None) -> DivotMetrics:
    """Return the divot, refusing to carry on without one.

    Args:
        divot: The measured divot, or ``None``.

    Returns:
        The divot.

    Raises:
        ValueError: If the divot was not measured. Without it the mass the
            sand carried is unknown, and #8657 removed the box-volume estimate
            that used to stand in for it.
    """
    if divot is None:
        raise ValueError(
            "the divot was not measured, so the mass of sand the strike moved "
            "is unknown and no carry can be derived from the delivered impulse"
        )
    return divot


@dataclass(frozen=True)
class _ShotViews:
    """The five drawable products of one shot, each optional.

    Bundled so :meth:`WorkbenchModel._reduce` receives one value rather than
    five positional results whose order a reader has to keep straight.

    Attributes:
        field: The per-element sole load field (#8705).
        sole_load: That field binned to the 12x12 bounce-utilisation map.
        patch: The engaged element set through the shot (#8707).
        scene: The 3-D scene of the head through the sand (#8706).
        traces: The scalar traces and the validity band (#8708).
    """

    field: SoleLoadField | None
    sole_load: SoleLoadMap | None
    patch: ContactPatch | None
    scene: ShotScene | None
    traces: ShotTraces | None


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
class PlayabilityOutcome:
    """The playability window, or the reason there is not one.

    Attributes:
        window: The measured window, or ``None`` when it could not be
            measured.
        unavailable_reason: Why, when ``window`` is ``None``.
        carry_m: ``(na, nb)`` carry grid [m]; NaN where the solver refused.
        attack_angle_deg: ``(na,)`` swept attack angles, for display.
        firmness_kg_per_cm2: ``(nb,)`` swept penetrometer readings.
        carry_verdict: The worst verdict over the answered grid points, which
            the whole grid must be read under. Present whenever any cell of
            ``carry_m`` is finite.
    """

    window: PlayabilityWindow | None
    unavailable_reason: str = ""
    carry_m: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.float64)
    )
    attack_angle_deg: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    firmness_kg_per_cm2: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    carry_verdict: ValidityVerdict | None = None

    def __post_init__(self) -> None:
        """Enforce the carry-verdict rule on the grid as well as the shot.

        Raises:
            ValueError: If any grid point holds a carry number and no verdict
                accompanies the grid (issue #8657).
        """
        if bool(np.isfinite(self.carry_m).any()) and self.carry_verdict is None:
            raise ValueError(
                "a carry grid and its validity verdict travel together; "
                f"{int(np.isfinite(self.carry_m).sum())} point(s) carry a "
                "number with no verdict to read them under"
            )

    @property
    def available(self) -> bool:
        """True when a window was measured."""
        return self.window is not None


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
            grid points to say anything.
        ranking_unavailable_reason: Why, when ``ranking`` is ``None``.
        shared_points: Number of grid points both designs answered.
    """

    left: DesignEvaluation
    right: DesignEvaluation
    ranking: DesignComparison | None = None
    ranking_unavailable_reason: str = ""
    shared_points: int = 0

    @property
    def separated(self) -> bool:
        """True when the ranking distinguishes the leader from the rival."""
        return self.ranking is not None and self.ranking.is_separated()

    @property
    def worst_status(self) -> EnvelopeStatus:
        """The worse of the two verdicts.

        A comparison is only as trustworthy as its least trustworthy half, so
        this is what the verdict banner shows. Combining the verdicts here
        rather than in the view keeps the ordering of the statuses in the one
        place that defines it.
        """
        return worst_of((self.left.verdict, self.right.verdict)).status


def _firmness_kpa(readings: NDArray[np.float64] | float) -> NDArray[np.float64]:
    """Convert penetrometer readings in kg/cm^2 to the SI unit the axis uses.

    Args:
        readings: One reading or an array of them, in kg/cm^2.

    Returns:
        The same values in kilopascals.
    """
    values = np.atleast_1d(np.asarray(readings, dtype=np.float64))
    converted = np.array(
        [firmness_pa_from_kg_per_cm2(float(value)) / 1e3 for value in values],
        dtype=np.float64,
    )
    return converted if np.ndim(readings) else converted[0]


class WorkbenchModel:
    """Runs the F0 tier for the designer workbench.

    The model owns no widgets and no state beyond its settings, so a caller
    may build one per evaluation or keep one for the session; the expensive
    lofted meshes are cached at module level and shared either way.
    """

    def __init__(self, settings: SolverSetup | None = None) -> None:
        """Initialise the model.

        Args:
            settings: Discretisation and study settings; the defaults are
                tuned so a single shot costs tens of milliseconds.
        """
        self._settings = SolverSetup() if settings is None else settings

    @property
    def settings(self) -> SolverSetup:
        """The discretisation and study settings in force."""
        return self._settings

    # ------------------------------------------------------------- one shot

    def head_build(self, geometry: WedgeGeometry) -> HeadBuild:
        """Return the cached lofted head for a design vector.

        Args:
            geometry: The design vector.

        Returns:
            The reusable build.

        Raises:
            WorkbenchInputError: If the sole cannot be lofted at this
                resolution. A design vector can satisfy every scalar
                invariant and still describe a sole whose camber segment has
                no solution, so the failure is relabelled here rather than
                escaping as a bare ``ValueError`` from deep inside the mesh
                generator.
        """
        try:
            return build_head(
                geometry, self._settings.n_profile_points, self._settings.n_stations
            )
        except ValueError as error:
            raise WorkbenchInputError(
                f"this sole cannot be lofted at {self._settings.n_stations} "
                f"stations x {self._settings.n_profile_points} sole samples: "
                f"{error}"
            ) from error

    def solver(self, sand: SandState, swing: SwingSetup) -> DRFTSolver:
        """Build the F0 solver for a sand state.

        The refusal policy is strict: an out-of-envelope query raises rather
        than returning a number that a caller might report.

        Args:
            sand: The sand state.
            swing: The delivery, supplying the dynamic-terms switch.

        Returns:
            The solver.
        """
        return DRFTSolver(
            material=MaterialResponse.from_sand_state(sand),
            dynamic_terms_active=bool(swing.dynamic_terms_active),
        )

    def shot_result(
        self, geometry: WedgeGeometry, sand: SandState, swing: SwingSetup
    ) -> ShotResult:
        """Run one shot and return the **record**, before any reduction.

        :meth:`run_shot` reduces the record to designer metrics and throws
        the poses away. The cross-tier check of issue #8713 needs them
        back: F1 has no instantaneous answer, so each of its probes is a
        march to a pose F0 recorded, and a re-derived pose would be a
        comparison of two different shots.

        Args:
            geometry: The resolved design vector.
            sand: The sand state.
            swing: The delivery.

        Returns:
            The whole strike.

        Raises:
            OutOfEnvelopeError: If the solver refuses any step. Unlike
                :meth:`run_shot` this does not translate the refusal into
                an outcome, because there is no outcome here to carry it.
        """
        build = self.head_build(geometry)
        return simulate_shot(
            self.solver(sand, swing),
            build.elements_body,
            head_mass_kg=geometry.head_mass_kg,
            kinematics=entry_kinematics(build, swing),
            settings=ShotSettings(
                time_step_s=self._settings.time_step_s,
                max_time_s=self._settings.max_time_s,
            ),
            sole_reference_body_m=build.sole_reference_body_m,
        )

    def run_shot(
        self, geometry: WedgeGeometry, sand: SandState, swing: SwingSetup
    ) -> ShotOutcome:
        """Run one shot and reduce it to the designer metrics.

        Args:
            geometry: The resolved design vector.
            sand: The sand state.
            swing: The delivery.

        Returns:
            The outcome. A refusal carries the verdict and nothing else.
        """
        build = self.head_build(geometry)
        solver = self.solver(sand, swing)
        delivered = deliver_wedge(geometry, swing.delivery())
        kinematics = entry_kinematics(build, swing)
        try:
            result = self.shot_result(geometry, sand, swing)
        except OutOfEnvelopeError as refusal:
            verdict = refusal.verdict
            require(
                isinstance(verdict, ValidityVerdict),
                "a refusal must carry the verdict that triggered it",
                value=type(verdict).__name__,
            )
            return ShotOutcome(
                verdict=verdict,
                fidelity_tier=FidelityTier.F0,
                refused=True,
                delivered=delivered,
            )
        return self._reduce(build, solver, result, kinematics, geometry, sand, swing)

    def _reduce(
        self,
        build: HeadBuild,
        solver: DRFTSolver,
        result: ShotResult,
        kinematics: HeadKinematics,
        geometry: WedgeGeometry,
        sand: SandState,
        swing: SwingSetup,
    ) -> ShotOutcome:
        """Turn a completed shot into the designer-facing metrics."""
        missing: list[str] = []
        delivered = deliver_wedge(geometry, swing.delivery())
        trace = strike_trace(result)
        if trace is None:
            missing.append(
                "trace metrics: the shot recorded fewer than 3 samples, which is "
                "too few to differentiate a velocity"
            )
        scene = _strike_scene(swing.ball_depth_m)
        head = build.head_model
        loads = _try(
            lambda: head_load_metrics(trace, head) if trace else None,
            "head loads",
            missing,
        )
        divot = _try(
            lambda: self._divot(trace, head, scene, geometry, sand),
            "divot geometry",
            missing,
        )
        skid = _try(
            lambda: dig_vs_skid(trace, head, scene) if trace else None,
            "dig-vs-skid",
            missing,
        )
        views = self._views(build, solver, result, kinematics, missing)
        carry = _try(
            lambda: self.carry_estimate(
                geometry, swing, _sand_delivery(result, _measured_divot(divot), sand)
            ),
            "carry",
            missing,
        )
        return ShotOutcome(
            verdict=result.verdict,
            fidelity_tier=result.fidelity_tier,
            refused=False,
            delivered=delivered,
            peak_force_n=result.peak_force_n,
            impulse_n_s=float(np.linalg.norm(result.impulse_n_s)),
            entry_speed_mps=result.entry_speed_m_s,
            exit_speed_mps=result.exit_speed_m_s,
            max_depth_m=result.max_sole_depth_m,
            contact_duration_s=result.contact_duration_s,
            peak_inertial_fraction=(
                float(result.inertial_fractions.max())
                if result.inertial_fractions.size
                else 0.0
            ),
            runtime_s=result.runtime_s,
            loads=loads,
            divot=divot,
            dig_skid=skid,
            sole_load=views.sole_load,
            sole_field=views.field,
            contact_patch=views.patch,
            scene=views.scene,
            traces=views.traces,
            carry_m=None if carry is None else carry.carry_m,
            carry_verdict=None if carry is None else carry.verdict,
            unavailable=tuple(missing),
        )

    def _views(
        self,
        build: HeadBuild,
        solver: DRFTSolver,
        result: ShotResult,
        kinematics: HeadKinematics,
        missing: list[str],
    ) -> _ShotViews:
        """Assemble everything the workbench draws from one completed shot.

        Grouped here rather than inline in :meth:`_reduce` because these five
        are one story -- the per-element load and everything derived from it,
        the 3-D scene, and the traces beside it -- and because each is
        individually optional: a shot too short for an impulse still has a
        verdict worth reporting.

        Args:
            build: The lofted head.
            solver: The solver the shot was run with.
            result: The shot trace.
            kinematics: The entry pose, supplying the constant orientation.
            missing: Reasons collected in place, one per view that could not
                be built.

        Returns:
            The views, each ``None`` where it could not be built.
        """
        # One replay of the recorded poses feeds all three load views: the
        # per-element field (#8705), the patch series (#8707) and the binned
        # map the workbench already reported. Re-deriving the element
        # response is the expensive half, so it happens once.
        field = _try(
            lambda: sole_load_field(solver, build, result, kinematics.orientation),
            "per-element sole load",
            missing,
        )
        patch = _try(
            lambda: None if field is None else contact_patch(field),
            "contact patch",
            missing,
        )
        # The band is a second replay, and a much cheaper one: it judges the
        # envelope per sample without integrating any force, which is the
        # half of a solve a verdict does not depend on (#8708).
        band = _try(
            lambda: validity_band(solver, build, result, kinematics.orientation),
            "validity band",
            missing,
        )
        return _ShotViews(
            field=field,
            sole_load=_try(
                lambda: self._sole_map(field), "bounce utilisation", missing
            ),
            patch=patch,
            # The 3-D scene needs no solving at all: the pose is the pose the
            # march recorded and the divot is accumulated from where the
            # head's own sole points went (#8706).
            scene=_try(lambda: shot_scene(build, result), "3-D shot scene", missing),
            traces=_try(
                lambda: (
                    None
                    if patch is None or band is None
                    else shot_traces(result, patch, band)
                ),
                "scalar traces",
                missing,
            ),
        )

    def _divot(
        self,
        trace: StrikeTrace | None,
        head: HeadModel,
        scene: StrikeScene,
        geometry: WedgeGeometry,
        sand: SandState,
    ) -> DivotMetrics | None:
        """Measure the divot one shot cut, when the trace supports it.

        Args:
            trace: The strike trace, or ``None`` when it was too short.
            head: The head the trace was recorded for.
            scene: The sand surface, ball and travel axis.
            geometry: The design vector, supplying the cutting width.
            sand: The sand state, supplying the bulk density.

        Returns:
            The divot, or ``None`` when there is no trace to measure it on.
        """
        if trace is None:
            return None
        return divot_metrics(
            trace,
            head,
            scene,
            width_m=geometry.sole_width_m,
            bulk_density_kg_m3=sand.bulk_density_kg_m3,
            friction_angle_deg=sand.friction_angle_deg,
        )

    def _sole_map(self, field: SoleLoadField | None) -> SoleLoadMap | None:
        """Bin an already-replayed load field into the utilisation map.

        Args:
            field: The per-element load field, or ``None`` when the trace was
                too short to replay.

        Returns:
            The binned map, or ``None``.
        """
        if field is None:
            return None
        load = field.load_trace()
        return sole_load_map(load, bounce_utilisation(load))

    # ------------------------------------------------------------ ball flight

    def carry_estimate(
        self, geometry: WedgeGeometry, swing: SwingSetup, delivery: SandDelivery
    ) -> CarryEstimate:
        """Carry the splash and flight models predict, with its verdict.

        The splash model is driven by the momentum the solver delivered and
        the divot mass the metrics layer measured, both of which arrive in
        ``delivery``. Since issue #8657 nothing here estimates displaced sand
        from a sole length and an entry depth.

        Args:
            geometry: The design vector, supplying loft and head mass.
            swing: The delivery.
            delivery: What the solver and metrics layer measured about the
                strike.

        Returns:
            The carry and the verdict it may be quoted under.

        Raises:
            RuntimeError: If the ball-flight kernel is unavailable. Reported
                as a missing metric rather than replaced by an estimate.
        """
        from src.shared.python.physics.ball_launch_conditions import LaunchConditions
        from src.shared.python.physics.ball_simulator import BallFlightSimulator

        delivered = deliver_wedge(geometry, swing.delivery())
        launch = compute_bunker_launch(
            BunkerShotState(
                club_loft_deg=float(delivered.effective_loft_deg),
                ball_lie=BallLie(depth_m=float(swing.ball_depth_m)),
                delivery=delivery,
                club_mass_kg=geometry.head_mass_kg,
            )
        )
        simulator = BallFlightSimulator()
        trajectory = simulator.simulate_trajectory(
            LaunchConditions(
                velocity=launch.ball_speed_m_s,
                launch_angle=launch.launch_angle_rad,
                azimuth_angle=launch.azimuth_rad,
                spin_rate=launch.spin_rate_rpm,
            ),
            max_time=_MAX_FLIGHT_TIME_S,
            dt=self._settings.flight_time_step_s,
        )
        return CarryEstimate(
            carry_m=float(simulator.analyze_trajectory(trajectory)["carry_distance"]),
            verdict=launch.verdict,
        )

    # ---------------------------------------------------------- playability

    def playability(
        self, geometry: WedgeGeometry, sand: SandCondition, swing: SwingSetup
    ) -> PlayabilityOutcome:
        """Measure the playability window over attack angle and firmness.

        The two axes are the delivery term the player controls least (attack
        angle, the largest single term in presentation) and the condition the
        player does not control at all (sand firmness). Entry distance behind
        the ball -- the other axis the epic names -- is **not** swept here:
        it does not enter the F0-to-splash chain, so sweeping it would report
        a flat carry and a meaninglessly perfect window.

        Args:
            geometry: The resolved design vector.
            sand: The playing condition; its firmness is swept.
            swing: The nominal delivery.

        Returns:
            The window, or the reason there is not one.
        """
        points = self._settings.playability_points
        attack_deg = np.linspace(*ATTACK_ANGLE_SWEEP_DEG, points, dtype=np.float64)
        firmness = np.linspace(*FIRMNESS_RANGE_KG_PER_CM2, points, dtype=np.float64)
        carry = np.full((points, points), np.nan, dtype=np.float64)
        reasons: list[str] = []
        verdicts: list[ValidityVerdict] = []
        for column, reading in enumerate(firmness):
            state = sand.with_firmness(float(reading)).sand_state()
            for row, angle in enumerate(attack_deg):
                estimate = self._grid_carry(
                    geometry, state, swing.with_attack_angle(float(angle)), reasons
                )
                if estimate is None:
                    continue
                carry[row, column] = estimate.carry_m
                verdicts.append(estimate.verdict)
        if not np.isfinite(carry).any():
            return PlayabilityOutcome(
                window=None,
                unavailable_reason=(
                    reasons[0]
                    if reasons
                    else "every point in the delivery sweep was refused"
                ),
                carry_m=carry,
                attack_angle_deg=attack_deg,
                firmness_kg_per_cm2=firmness,
            )
        window = playability_window(
            PlayabilityAxis("attack_angle", "rad", np.radians(attack_deg)),
            PlayabilityAxis("sand_firmness", "kPa", _firmness_kpa(firmness)),
            carry,
            target_carry_m=self._settings.target_carry_m,
            tolerance_fraction=self._settings.carry_tolerance_fraction,
            nominal=(
                math.radians(swing.attack_angle_deg),
                float(_firmness_kpa(sand.sand_state().firmness_kg_per_cm2)),
            ),
        )
        return PlayabilityOutcome(
            window=window,
            carry_m=carry,
            attack_angle_deg=attack_deg,
            firmness_kg_per_cm2=firmness,
            carry_verdict=worst_of(verdicts),
        )

    def _grid_carry(
        self,
        geometry: WedgeGeometry,
        sand: SandState,
        swing: SwingSetup,
        reasons: list[str],
    ) -> CarryEstimate | None:
        """Carry at one grid point, or ``None`` when it is unanswerable.

        Args:
            geometry: The design vector.
            sand: The sand state at this grid point.
            swing: The delivery at this grid point.
            reasons: Accumulator for why a point could not be answered.

        Returns:
            The carry and its verdict, or ``None``.
        """
        build = self.head_build(geometry)
        solver = self.solver(sand, swing)
        kinematics = entry_kinematics(build, swing)
        try:
            result = simulate_shot(
                solver,
                build.elements_body,
                head_mass_kg=geometry.head_mass_kg,
                kinematics=kinematics,
                settings=ShotSettings(
                    time_step_s=self._settings.time_step_s,
                    max_time_s=self._settings.max_time_s,
                ),
                sole_reference_body_m=build.sole_reference_body_m,
            )
        except OutOfEnvelopeError:
            reasons.append("the solver refused every point in the delivery sweep")
            return None
        try:
            trace = strike_trace(result)
            divot = self._divot(
                trace,
                build.head_model,
                _strike_scene(swing.ball_depth_m),
                geometry,
                sand,
            )
            return self.carry_estimate(
                geometry, swing, _sand_delivery(result, _measured_divot(divot), sand)
            )
        except (RuntimeError, ImportError, ValueError) as error:
            reasons.append(f"carry is unavailable: {error}")
            return None

    # -------------------------------------------------------------- top level

    def evaluate(
        self,
        design: WedgeDesign,
        sand: SandCondition,
        swing: SwingSetup,
        *,
        include_playability: bool = True,
    ) -> DesignEvaluation:
        """Evaluate one candidate sole.

        Args:
            design: The designer's inputs.
            sand: The playing condition.
            swing: The delivery.
            include_playability: Whether to sweep the playability grid, which
                costs ``playability_points ** 2`` extra shots.

        Returns:
            The evaluation.

        Raises:
            WorkbenchInputError: If the design is not a constructible sole.
        """
        geometry = design.geometry()
        state = sand.sand_state()
        shot = self.run_shot(geometry, state, swing)
        window = (
            self.playability(geometry, sand, swing)
            if include_playability
            else PlayabilityOutcome(
                window=None, unavailable_reason="the playability sweep was not run"
            )
        )
        # Free: the build is cached, and run_shot has already made it.
        build = self.head_build(geometry)
        return DesignEvaluation(
            design=design,
            geometry=geometry,
            sand=state,
            shot=shot,
            playability=window,
            effective_camber_area_m2=build.effective_camber_area_m2,
            camber_stations=build.camber_stations,
        )

    def compare(
        self,
        left: WedgeDesign,
        right: WedgeDesign,
        sand: SandCondition,
        swing: SwingSetup,
    ) -> WorkbenchComparison:
        """Rank two candidate soles against each other, with uncertainty.

        The objective is absolute carry error against the target, evaluated
        at every point of the shared delivery sweep, so a design wins by
        holding carry near target across the conditions the player does not
        control -- not by being long in one flattering condition.

        Args:
            left: First candidate.
            right: Second candidate.
            sand: The playing condition.
            swing: The nominal delivery.

        Returns:
            Both evaluations and, when the grids allow it, the ranking.

        Raises:
            ValueError: If the two designs share a name, which would make the
                ranking unreadable.
            WorkbenchInputError: If either design is not constructible.
        """
        if left.name == right.name:
            raise ValueError(
                "the two designs must have different names; the comparison "
                f"reports them by name and both are {left.name!r}"
            )
        first = self.evaluate(left, sand, swing)
        second = self.evaluate(right, sand, swing)
        target = self._settings.target_carry_m
        errors = [
            np.abs(evaluation.playability.carry_m.ravel() - target)
            for evaluation in (first, second)
        ]
        if errors[0].shape != errors[1].shape or errors[0].size == 0:
            return WorkbenchComparison(
                left=first,
                right=second,
                ranking_unavailable_reason=(
                    "the two designs were not evaluated on the same grid"
                ),
            )
        shared = np.isfinite(errors[0]) & np.isfinite(errors[1])
        if int(shared.sum()) < 2:
            return WorkbenchComparison(
                left=first,
                right=second,
                ranking_unavailable_reason=(
                    "fewer than two delivery conditions were answerable for both "
                    "designs, so the two cannot be told apart"
                ),
                shared_points=int(shared.sum()),
            )
        return WorkbenchComparison(
            left=first,
            right=second,
            ranking=compare_designs(
                (left.name, right.name),
                np.vstack([errors[0][shared], errors[1][shared]]),
                lower_is_better=True,
                seed=COMPARISON_SEED,
            ),
            shared_points=int(shared.sum()),
        )


def _try(
    compute: Callable[[], _MetricT | None], label: str, missing: list[str]
) -> _MetricT | None:
    """Run a metric, recording why it is missing instead of failing the shot.

    Only the errors a metric raises for a legitimately unmeasurable trace are
    caught -- a sole that never left the sand, a flight kernel that is not
    installed. A programming error still propagates.

    Args:
        compute: Zero-argument callable producing the metric.
        label: Metric name, used in the reason line.
        missing: Accumulator of reason lines.

    Returns:
        The metric, or ``None`` when it could not be computed.
    """
    try:
        return compute()
    except (ValueError, RuntimeError, ImportError) as error:
        missing.append(f"{label}: {error}")
        logger.debug("workbench metric %s unavailable: %s", label, error)
        return None
