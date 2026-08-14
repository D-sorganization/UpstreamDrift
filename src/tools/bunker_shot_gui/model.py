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

from bunkershot3d.ball import BallLie, BunkerShotState, compute_bunker_launch
from bunkershot3d.geometry import (
    DeliveredGeometry,
    WedgeGeometry,
    deliver_wedge,
)
from bunkershot3d.metrics import (
    DigSkidResult,
    DivotMetrics,
    HeadLoadMetrics,
    PlayabilityAxis,
    PlayabilityWindow,
    StrikeScene,
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
    sole_load_map,
    sole_load_trace,
    strike_trace,
)
from .design import (
    FIRMNESS_RANGE_KG_PER_CM2,
    SandCondition,
    SolverSetup,
    SwingSetup,
    WedgeDesign,
    WorkbenchInputError,
)

__all__ = [
    "ATTACK_ANGLE_SWEEP_DEG",
    "COMPARISON_SEED",
    "DesignEvaluation",
    "HeadBuild",
    "PlayabilityOutcome",
    "ShotOutcome",
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
        sole_load: The bounce-utilisation map.
        carry_m: Carry from the splash and flight models.
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
    carry_m: float | None = None
    unavailable: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Enforce the refusal rule.

        Raises:
            ValueError: If a refused outcome carries any number. ADR-0032
                makes this the single most important behaviour of the tier,
                so it is a ``raise`` rather than a contract decorator --
                ``DBC_LEVEL=off`` must not switch the envelope off.
        """
        if not self.refused:
            return
        numbers = (
            self.peak_force_n,
            self.impulse_n_s,
            self.max_depth_m,
            self.carry_m,
        )
        if any(value is not None for value in numbers):
            raise ValueError(
                "a refused shot must not carry a force, a depth or a carry: "
                "3D-RFT declined this query, and reporting a number beside a "
                "REFUSED verdict is exactly what ADR-0032 forbids"
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
    """

    design: WedgeDesign
    geometry: WedgeGeometry
    sand: SandState
    shot: ShotOutcome
    playability: PlayabilityOutcome

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
        kinematics = entry_kinematics(build, swing, self._settings)
        settings = ShotSettings(
            time_step_s=self._settings.time_step_s,
            max_time_s=self._settings.max_time_s,
            start_at_first_contact=False,
        )
        try:
            result = simulate_shot(
                solver,
                build.elements_body,
                head_mass_kg=geometry.head_mass_kg,
                kinematics=kinematics,
                settings=settings,
            )
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
        trace = strike_trace(
            result,
            kinematics.orientation,
            kinematics.orientation @ build.sole_reference_body_m,
        )
        if trace is None:
            missing.append(
                "trace metrics: the shot recorded fewer than 3 samples, which is "
                "too few to differentiate a velocity"
            )
        scene = StrikeScene(
            sand_surface_height_m=0.0,
            ball_position_m=np.array(
                [0.0, 0.0, BallLie(depth_m=swing.ball_depth_m).center_z_m()],
                dtype=np.float64,
            ),
            travel_axis=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        )
        head = build.head_model
        loads = _try(
            lambda: head_load_metrics(trace, head) if trace else None,
            "head loads",
            missing,
        )
        divot = _try(
            lambda: (
                divot_metrics(
                    trace,
                    head,
                    scene,
                    width_m=geometry.sole_width_m,
                    bulk_density_kg_m3=sand.bulk_density_kg_m3,
                )
                if trace
                else None
            ),
            "divot geometry",
            missing,
        )
        skid = _try(
            lambda: dig_vs_skid(trace, head, scene) if trace else None,
            "dig-vs-skid",
            missing,
        )
        sole_load = _try(
            lambda: self._sole_map(solver, build, result, kinematics),
            "bounce utilisation",
            missing,
        )
        carry = _try(
            lambda: self.carry_m(geometry, swing, result.max_depth_m),
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
            max_depth_m=result.max_depth_m,
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
            sole_load=sole_load,
            carry_m=carry,
            unavailable=tuple(missing),
        )

    def _sole_map(
        self,
        solver: DRFTSolver,
        build: HeadBuild,
        result: ShotResult,
        kinematics: HeadKinematics,
    ) -> SoleLoadMap | None:
        """Replay the trace and resolve the bounce utilisation."""
        load = sole_load_trace(solver, build, result, kinematics.orientation)
        if load is None:
            return None
        return sole_load_map(load, bounce_utilisation(load))

    # ------------------------------------------------------------ ball flight

    def carry_m(
        self, geometry: WedgeGeometry, swing: SwingSetup, entry_depth_m: float
    ) -> float:
        """Carry the splash and flight models predict for one shot.

        Args:
            geometry: The design vector, supplying loft, sole and mass.
            swing: The delivery.
            entry_depth_m: How deep the sole went, from the F0 shot.

        Returns:
            Carry distance [m].

        Raises:
            RuntimeError: If the ball-flight kernel is unavailable. Reported
                as a missing metric rather than replaced by an estimate.
        """
        from src.shared.python.physics.ball_launch_conditions import LaunchConditions
        from src.shared.python.physics.ball_simulator import BallFlightSimulator

        delivered = deliver_wedge(geometry, swing.delivery())
        launch = compute_bunker_launch(
            BunkerShotState(
                club_velocity_m_s=float(swing.clubhead_speed_mps),
                club_loft_deg=float(delivered.effective_loft_deg),
                ball_lie=BallLie(depth_m=float(swing.ball_depth_m)),
                entry_depth_m=max(float(entry_depth_m), 0.0),
                sole_width_m=geometry.sole_width_m,
                sole_length_m=geometry.blade_length_m,
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
        carry = np.full((points, points), np.nan, dtype=np.float64)
        reasons: list[str] = []
        for column, reading in enumerate(firmness):
            state = sand.with_firmness(float(reading)).sand_state()
            for row, angle in enumerate(attack_deg):
                carry[row, column] = self._grid_carry(
                    geometry, state, swing.with_attack_angle(float(angle)), reasons
                )
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
        )

    def _grid_carry(
        self,
        geometry: WedgeGeometry,
        sand: SandState,
        swing: SwingSetup,
        reasons: list[str],
    ) -> float:
        """Carry at one grid point, or NaN when the point is unanswerable."""
        build = self.head_build(geometry)
        solver = self.solver(sand, swing)
        try:
            result = simulate_shot(
                solver,
                build.elements_body,
                head_mass_kg=geometry.head_mass_kg,
                kinematics=entry_kinematics(build, swing, self._settings),
                settings=ShotSettings(
                    time_step_s=self._settings.time_step_s,
                    max_time_s=self._settings.max_time_s,
                    start_at_first_contact=False,
                ),
            )
        except OutOfEnvelopeError:
            reasons.append("the solver refused every point in the delivery sweep")
            return math.nan
        try:
            return self.carry_m(geometry, swing, result.max_depth_m)
        except (RuntimeError, ImportError, ValueError) as error:
            reasons.append(f"carry is unavailable: {error}")
            return math.nan

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
        return DesignEvaluation(
            design=design,
            geometry=geometry,
            sand=state,
            shot=shot,
            playability=window,
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
