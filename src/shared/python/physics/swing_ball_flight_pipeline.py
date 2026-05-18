"""SwingBallFlightPipeline — end-to-end swing-dynamics → ball-flight connector.

Implements the foundational pipeline tile from Issue #5337:

    Physics Engine (swing dynamics)
      → Impact Model (club-ball collision)
      → BallFlightSimulator (aerodynamic trajectory)
      → PipelineResult

Design-by-Contract invariants
------------------------------
- ``launch_speed`` of the derived ``LaunchConditions`` must be > 0.
- ``launch_angle_deg`` must be in [0°, 90°].
- ``PipelineResult`` is always fully populated (no None trajectory).

Law of Demeter
--------------
``SwingBallFlightPipeline`` calls ``ImpactSolverAPI`` (one layer) and
``BallFlightSimulator`` (one layer).  It never reaches into solver internals
or trajectory points directly.

DRY
---
``LaunchConditions`` and ``TrajectoryPoint`` come from the existing
``ball_launch_conditions`` module — they are NOT redefined here.

Implements Issue #5337 (foundation).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from src.shared.python.contracts import ensure, require
from src.shared.python.physics.ball_launch_conditions import (
    EnvironmentalConditions,
    LaunchConditions,
    TrajectoryPoint,
)
from src.shared.python.physics.ball_properties import BallProperties
from src.shared.python.physics.impact_model import (
    ImpactModelType,
    ImpactParameters,
    ImpactSolverAPI,
    PostImpactState,
    PreImpactState,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Protocol — flight simulator (dependency injection)
# ---------------------------------------------------------------------------


@runtime_checkable
class FlightSimulatorProtocol(Protocol):
    """Minimal protocol for any ball-flight simulator.

    Implementations include ``BallFlightSimulator`` and
    ``EnhancedBallFlightSimulator``.  Tests inject mocks.
    """

    def simulate_trajectory(
        self,
        launch: LaunchConditions,
        max_time: float = 10.0,
        dt: float = 0.01,
    ) -> list[TrajectoryPoint]:
        """Return a list of TrajectoryPoint from launch to landing."""
        ...


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_LAUNCH_ANGLE_DEG: float = 90.0
_MIN_LAUNCH_SPEED_MS: float = 0.0  # exclusive lower bound

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SwingState:
    """Represents the clubhead kinematics at the moment of impact.

    All velocity and angular-velocity vectors are expressed in the global
    Cartesian frame [x=forward, y=left, z=up].

    Attributes:
        clubhead_velocity:          Clubhead velocity at impact [m/s] (3,).
        clubhead_angular_velocity:  Clubhead angular velocity [rad/s] (3,).
        clubhead_orientation:       Clubface normal unit vector (3,).
        clubhead_mass:              Effective clubhead mass [kg].
        clubhead_loft_deg:          Clubface loft angle [degrees].
        clubhead_moi:               Clubhead moment of inertia [kg·m²].
        impact_offset:              Off-center offset on clubface [m] (2,) or None.
        engine_name:                Name of the physics engine that produced this state.
        metadata:                   Arbitrary extra data from the engine (e.g. residuals).
    """

    clubhead_velocity: np.ndarray
    clubhead_angular_velocity: np.ndarray
    clubhead_orientation: np.ndarray
    clubhead_mass: float = 0.200
    clubhead_loft_deg: float = 10.5
    clubhead_moi: float = 5e-3
    impact_offset: np.ndarray | None = None
    engine_name: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Full end-to-end simulation result.

    Attributes:
        swing_state:        The ``SwingState`` that initiated the pipeline.
        impact_state:       Post-impact ball and clubhead kinematics.
        launch_conditions:  Derived ``LaunchConditions`` fed to the ball-flight sim.
        trajectory:         Full aerodynamic trajectory as ``TrajectoryPoint`` list.
        carry_m:            Carry distance [m] (horizontal range to landing).
        max_height_m:       Maximum ball height [m].
        flight_time_s:      Total flight time [s].
        landing_angle_deg:  Descent angle at landing [degrees, positive].
        impact_params:      Impact model parameters used.
        environment:        Environmental conditions used.
        metadata:           Extra diagnostics (engine names, flags, etc.).
    """

    swing_state: SwingState
    impact_state: PostImpactState
    launch_conditions: LaunchConditions
    trajectory: list[TrajectoryPoint]
    carry_m: float
    max_height_m: float
    flight_time_s: float
    landing_angle_deg: float
    impact_params: ImpactParameters = field(default_factory=ImpactParameters)
    environment: EnvironmentalConditions = field(
        default_factory=EnvironmentalConditions
    )
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Private helpers — Law of Demeter boundary
# ---------------------------------------------------------------------------


class _LaunchConditionsDeriver:
    """Converts a ``PostImpactState`` to ``LaunchConditions``.

    Owned by ``SwingBallFlightPipeline``; not part of the public surface.
    """

    def derive(self, post: PostImpactState) -> LaunchConditions:
        """Return ``LaunchConditions`` from post-impact ball kinematics.

        Args:
            post: Post-impact state from the impact solver.

        Returns:
            LaunchConditions ready for the ball-flight simulator.
        """
        ball_vel = np.asarray(post.ball_velocity, dtype=float)
        speed = float(np.linalg.norm(ball_vel))

        # Launch angle from horizontal plane
        horiz_speed = float(np.linalg.norm(ball_vel[:2]))
        if horiz_speed < 1e-12:
            launch_angle_deg = 90.0
        else:
            launch_angle_deg = float(np.degrees(np.arctan2(ball_vel[2], horiz_speed)))

        # Azimuth angle (compass bearing, 0 = forward = +x)
        azimuth_deg = 0.0
        if horiz_speed > 1e-12:
            azimuth_deg = float(np.degrees(np.arctan2(ball_vel[1], ball_vel[0])))

        spin_rate = float(np.linalg.norm(post.ball_angular_velocity))
        spin_axis = (
            post.ball_angular_velocity / spin_rate
            if spin_rate > 1e-12
            else np.array([0.0, -1.0, 0.0])
        )

        return LaunchConditions(
            velocity=speed,
            launch_angle=launch_angle_deg,
            azimuth_angle=azimuth_deg,
            spin_rate=spin_rate,
            spin_axis=np.asarray(spin_axis, dtype=float),
        )


class _TrajectoryMetricsExtractor:
    """Extracts scalar metrics from a trajectory list.

    Owned by ``SwingBallFlightPipeline``; encapsulates LOD boundary.
    """

    def carry_m(self, trajectory: list[TrajectoryPoint]) -> float:
        """Return carry distance in metres."""
        if not trajectory:
            return 0.0
        pos = trajectory[-1].position
        return float(np.sqrt(pos[0] ** 2 + pos[1] ** 2))

    def max_height_m(self, trajectory: list[TrajectoryPoint]) -> float:
        """Return maximum height in metres."""
        if not trajectory:
            return 0.0
        return float(max(p.position[2] for p in trajectory))

    def flight_time_s(self, trajectory: list[TrajectoryPoint]) -> float:
        """Return flight time in seconds."""
        if not trajectory:
            return 0.0
        return float(trajectory[-1].time)

    def landing_angle_deg(self, trajectory: list[TrajectoryPoint]) -> float:
        """Return descent angle at landing in degrees (positive = downward)."""
        if len(trajectory) < 2:
            return 0.0
        v = trajectory[-1].velocity
        v_horiz = float(np.linalg.norm(v[:2]))
        if v_horiz < 1e-12:
            return 90.0
        return float(np.degrees(np.arctan2(-v[2], v_horiz)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class SwingBallFlightPipeline:
    """End-to-end pipeline: swing state → impact → ball flight → result.

    Accepts ``SwingState`` (clubhead kinematics), solves the club-ball
    impact with the chosen impact model, derives ``LaunchConditions``, runs
    a ball-flight simulation, and returns a ``PipelineResult``.

    Args:
        impact_params:  Impact model parameters (COR, friction, etc.).
            Defaults to ``ImpactParameters()``.
        environment:    Environmental conditions (air density, wind, gravity).
            Defaults to ``EnvironmentalConditions()``.
        ball:           Ball physical properties. Defaults to ``BallProperties()``.
        model_type:     Impact model type.
            Defaults to ``ImpactModelType.RIGID_BODY``.
        max_flight_time: Upper bound on ball-flight integration [s].
        flight_dt:      Time step for flight integration [s].

    Examples::

        pipeline = SwingBallFlightPipeline()
        swing = SwingState(
            clubhead_velocity=np.array([45.0, 0.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([0.0, 0.0, 1.0]),
        )
        result = pipeline.run(swing)
        print(f"Carry: {result.carry_m:.1f} m")
    """

    def __init__(
        self,
        impact_params: ImpactParameters | None = None,
        environment: EnvironmentalConditions | None = None,
        ball: BallProperties | None = None,
        model_type: ImpactModelType = ImpactModelType.RIGID_BODY,
        max_flight_time: float = 10.0,
        flight_dt: float = 0.01,
        flight_simulator: FlightSimulatorProtocol | None = None,
    ) -> None:
        require(
            max_flight_time > 0,
            "max_flight_time must be positive",
            max_flight_time,
        )
        require(
            0 < flight_dt <= max_flight_time,
            "flight_dt must be positive and <= max_flight_time",
            flight_dt,
        )
        self._impact_params = impact_params or ImpactParameters()
        self._environment = environment or EnvironmentalConditions()
        self._ball = ball or BallProperties()
        self._model_type = model_type
        self._max_flight_time = max_flight_time
        self._flight_dt = flight_dt

        # Private helpers (LOD)
        self._solver = ImpactSolverAPI(
            model_type=model_type, params=self._impact_params
        )
        self._deriver = _LaunchConditionsDeriver()
        self._metrics = _TrajectoryMetricsExtractor()
        # Accept injected simulator for testability; production code passes None
        # to defer construction until first use (lazy Rust-kernel check).
        self._flight_simulator = flight_simulator

        logger.debug(
            "swing_ball_flight_pipeline_created model_type=%s max_flight_time=%s",
            model_type.name,
            max_flight_time,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def impact_params(self) -> ImpactParameters:
        """Return the impact model parameters."""
        return self._impact_params

    @property
    def environment(self) -> EnvironmentalConditions:
        """Return the environmental conditions."""
        return self._environment

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, swing: SwingState) -> PipelineResult:
        """Execute the full swing → flight pipeline.

        Preconditions:
            - ``swing`` is a ``SwingState`` instance.
            - ``swing.clubhead_velocity`` must produce a post-impact ball speed > 0.

        Postcondition:
            Returns a ``PipelineResult`` with a non-empty trajectory list.

        Raises:
            TypeError/ValueError: If preconditions are violated.

        Args:
            swing: Clubhead kinematics at the moment of impact.

        Returns:
            ``PipelineResult`` with trajectory, carry, height, and impact data.
        """
        require(
            isinstance(swing, SwingState),
            "swing must be a SwingState instance",
            swing,
        )

        # Step 1: Build pre-impact state from swing
        pre = self._build_pre_impact_state(swing)

        # Step 2: Solve impact
        post = self._solve_impact(pre)

        # Step 3: Derive launch conditions
        launch = self._deriver.derive(post)
        self._validate_launch_conditions(launch)

        # Step 4: Run ball flight simulation
        trajectory = self._run_flight_simulation(launch)

        # Step 5: Extract metrics
        result = self._build_result(swing, post, launch, trajectory)

        ensure(
            len(result.trajectory) > 0,
            "run postcondition: trajectory must be non-empty",
        )
        logger.info(
            "pipeline_complete engine=%s carry_m=%.1f max_height_m=%.1f",
            swing.engine_name,
            result.carry_m,
            result.max_height_m,
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_pre_impact_state(self, swing: SwingState) -> PreImpactState:
        """Convert ``SwingState`` to ``PreImpactState`` (LOD boundary)."""
        loft_rad = float(np.radians(swing.clubhead_loft_deg))
        return PreImpactState(
            clubhead_velocity=np.asarray(swing.clubhead_velocity, dtype=float),
            clubhead_angular_velocity=np.asarray(
                swing.clubhead_angular_velocity, dtype=float
            ),
            clubhead_orientation=np.asarray(swing.clubhead_orientation, dtype=float),
            ball_position=np.zeros(3),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
            clubhead_mass=swing.clubhead_mass,
            clubhead_loft=loft_rad,
            clubhead_moi=swing.clubhead_moi,
            impact_offset=swing.impact_offset,
        )

    def _solve_impact(self, pre: PreImpactState) -> PostImpactState:
        """Delegate impact solving to ``ImpactSolverAPI`` (one layer)."""
        if pre.impact_offset is not None:
            return self._solver.solve_with_gear_effect(
                timestamp=0.0,
                clubhead_velocity=pre.clubhead_velocity,
                clubhead_orientation=pre.clubhead_orientation,
                impact_offset=pre.impact_offset,
                ball_velocity=pre.ball_velocity,
                clubhead_mass=pre.clubhead_mass,
            )
        return self._solver.solve_impact(
            timestamp=0.0,
            clubhead_velocity=pre.clubhead_velocity,
            clubhead_orientation=pre.clubhead_orientation,
            ball_velocity=pre.ball_velocity,
            ball_angular_velocity=pre.ball_angular_velocity,
            clubhead_mass=pre.clubhead_mass,
        )

    def _run_flight_simulation(self, launch: LaunchConditions) -> list[TrajectoryPoint]:
        """Delegate trajectory integration to the flight simulator (one layer)."""
        sim = self._get_flight_simulator()
        return sim.simulate_trajectory(
            launch,
            max_time=self._max_flight_time,
            dt=self._flight_dt,
        )

    def _get_flight_simulator(self) -> FlightSimulatorProtocol:
        """Return the flight simulator, constructing it lazily if needed.

        Lazy construction defers the Rust-kernel availability check until
        first use, keeping ``__init__`` side-effect-free.
        """
        if self._flight_simulator is not None:
            return self._flight_simulator
        # Lazy import to avoid circular dependencies and early Rust checks
        from src.shared.python.physics.ball_simulator import BallFlightSimulator

        return BallFlightSimulator(ball=self._ball, env=self._environment)

    def _build_result(
        self,
        swing: SwingState,
        post: PostImpactState,
        launch: LaunchConditions,
        trajectory: list[TrajectoryPoint],
    ) -> PipelineResult:
        """Assemble the final ``PipelineResult``."""
        return PipelineResult(
            swing_state=swing,
            impact_state=post,
            launch_conditions=launch,
            trajectory=trajectory,
            carry_m=self._metrics.carry_m(trajectory),
            max_height_m=self._metrics.max_height_m(trajectory),
            flight_time_s=self._metrics.flight_time_s(trajectory),
            landing_angle_deg=self._metrics.landing_angle_deg(trajectory),
            impact_params=self._impact_params,
            environment=self._environment,
            metadata={"engine": swing.engine_name, **swing.metadata},
        )

    @staticmethod
    def _validate_launch_conditions(launch: LaunchConditions) -> None:
        """Enforce DbC postconditions on derived launch conditions."""
        require(
            launch.velocity > _MIN_LAUNCH_SPEED_MS,
            "launch speed must be > 0 — check clubhead velocity",
            launch.velocity,
        )
        require(
            0.0 <= launch.launch_angle <= _MAX_LAUNCH_ANGLE_DEG,
            f"launch_angle must be in [0°, {_MAX_LAUNCH_ANGLE_DEG}°]",
            launch.launch_angle,
        )
