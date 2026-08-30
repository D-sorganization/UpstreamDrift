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
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from bunkershot3d.ball import (
    BallLaunchResult,
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
from bunkershot3d.study.ranking import BandedRanking, RankingVerdict, rank_with_bands
from bunkershot3d.vandv.band import ConsistencyBand
from bunkershot3d.vandv.budget import (
    UncertaintyBudget,
    UncertaintyClass,
    UncertaintyTerm,
)
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
from .outcomes import DesignEvaluation, ShotOutcome, WorkbenchComparison
from .traces import ShotTraces, shot_traces
from .uncertainty import (
    CARRY_BAND_SOURCE,
    CARRY_NUMERICAL_UNQUANTIFIED,
    LAUNCH_DIRECTION_UNQUANTIFIED,
    TRANSFER_EFFICIENCY_UNQUANTIFIED,
    CarryEstimate,
    CarrySweep,
    PlayabilityOutcome,
    objective_band,
    objective_budget,
    propagate_carry_band,
)

__all__ = [
    "ATTACK_ANGLE_SWEEP_DEG",
    "CARRY_BAND_SOURCE",
    "CARRY_NUMERICAL_UNQUANTIFIED",
    "COMPARISON_SEED",
    "LAUNCH_DIRECTION_UNQUANTIFIED",
    "TRANSFER_EFFICIENCY_UNQUANTIFIED",
    "BandedRanking",
    "CarryEstimate",
    "ConsistencyBand",
    "ContactPatch",
    "DesignEvaluation",
    "HeadBuild",
    "PlayabilityOutcome",
    "RankingVerdict",
    "ShotOutcome",
    "ShotScene",
    "ShotTraces",
    "SoleLoadField",
    "SoleLoadMap",
    "UncertaintyBudget",
    "UncertaintyClass",
    "UncertaintyTerm",
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
            carry_band=None if carry is None else carry.band,
            carry_band_reasons=() if carry is None else carry.band_reasons,
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
        delivered = deliver_wedge(geometry, swing.delivery())
        state = BunkerShotState(
            club_loft_deg=float(delivered.effective_loft_deg),
            ball_lie=BallLie(depth_m=float(swing.ball_depth_m)),
            delivery=delivery,
            club_mass_kg=geometry.head_mass_kg,
        )
        launch = compute_bunker_launch(state)
        central = self._flight_carry_m(launch)
        band, reasons = propagate_carry_band(
            state,
            central,
            lambda edge: self._flight_carry_m(compute_bunker_launch(edge)),
        )
        return CarryEstimate(
            carry_m=central,
            verdict=launch.verdict,
            band=band,
            band_reasons=reasons,
        )

    def _flight_carry_m(self, launch: BallLaunchResult) -> float:
        """Fly one launch and report its carry.

        Args:
            launch: The launch conditions the splash model produced.

        Returns:
            Carry distance [m].

        Raises:
            RuntimeError: If the ball-flight kernel is unavailable.
        """
        from src.shared.python.physics.ball_launch_conditions import LaunchConditions
        from src.shared.python.physics.ball_simulator import BallFlightSimulator

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
        return float(simulator.analyze_trajectory(trajectory)["carry_distance"])

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
        sweep = CarrySweep(points)
        for column, reading in enumerate(firmness):
            state = sand.with_firmness(float(reading)).sand_state()
            for row, angle in enumerate(attack_deg):
                sweep.record(
                    row,
                    column,
                    self._grid_carry(
                        geometry,
                        state,
                        swing.with_attack_angle(float(angle)),
                        sweep.reasons,
                    ),
                )
        if not np.isfinite(sweep.carry).any():
            return PlayabilityOutcome(
                window=None,
                unavailable_reason=(
                    sweep.reasons[0]
                    if sweep.reasons
                    else "every point in the delivery sweep was refused"
                ),
                carry_m=sweep.carry,
                attack_angle_deg=attack_deg,
                firmness_kg_per_cm2=firmness,
            )
        window = playability_window(
            PlayabilityAxis("attack_angle", "rad", np.radians(attack_deg)),
            PlayabilityAxis("sand_firmness", "kPa", _firmness_kpa(firmness)),
            sweep.carry,
            target_carry_m=self._settings.target_carry_m,
            tolerance_fraction=self._settings.carry_tolerance_fraction,
            nominal=(
                math.radians(swing.attack_angle_deg),
                float(_firmness_kpa(sand.sand_state().firmness_kg_per_cm2)),
            ),
        )
        return PlayabilityOutcome(
            window=window,
            carry_m=sweep.carry,
            attack_angle_deg=attack_deg,
            firmness_kg_per_cm2=firmness,
            carry_verdict=worst_of(sweep.verdicts),
            carry_lower_m=sweep.lower,
            carry_upper_m=sweep.upper,
            band_reasons=sweep.unique_band_reasons(),
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
        ranking = compare_designs(
            (left.name, right.name),
            np.vstack([errors[0][shared], errors[1][shared]]),
            lower_is_better=True,
            seed=COMPARISON_SEED,
        )
        return WorkbenchComparison(
            left=first,
            right=second,
            ranking=ranking,
            shared_points=int(shared.sum()),
            banded=rank_with_bands(
                left.name,
                self.objective_budget(first, shared, float(ranking.std_error[0])),
                right.name,
                self.objective_budget(second, shared, float(ranking.std_error[1])),
                lower_is_better=True,
            ),
        )

    def objective_budget(
        self,
        evaluation: DesignEvaluation,
        shared: NDArray[np.bool_],
        sampling_std_error: float,
    ) -> UncertaintyBudget:
        """Assemble one design's uncertainty budget for the ranking objective.

        A thin delegate to
        :func:`~src.tools.bunker_shot_gui.uncertainty.objective_budget`, which
        holds the arithmetic and the reasons; this method only supplies the
        target the settings define.

        Args:
            evaluation: The design's evaluation, carrying the carry grids.
            shared: Boolean mask over the flattened grid, true where both
                designs answered.
            sampling_std_error: Bootstrap standard error of this design's mean
                objective, from
                :func:`~bunkershot3d.study.comparison.compare_designs`.

        Returns:
            The budget, whose centre is the mean absolute carry error.
        """
        return objective_budget(
            evaluation.design.name,
            evaluation.playability,
            shared,
            target_carry_m=self._settings.target_carry_m,
            sampling_std_error=sampling_std_error,
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
