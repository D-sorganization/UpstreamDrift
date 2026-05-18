"""Tests for Issue #5337: SwingBallFlightPipeline foundation.

Coverage:
- SwingState construction (3 tests)
- PipelineResult construction (2 tests)
- SwingBallFlightPipeline construction and preconditions (4 tests)
- run() happy-path with mocked flight simulator (6 tests)
- DbC validation of launch conditions (3 tests)
- _LaunchConditionsDeriver unit tests (4 tests)
- _TrajectoryMetricsExtractor unit tests (5 tests)
- FlightSimulatorProtocol structural subtyping (1 test)

Total: 28 tests
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.contracts import PreconditionError
from src.shared.python.physics.ball_launch_conditions import (
    EnvironmentalConditions,
    LaunchConditions,
    TrajectoryPoint,
)
from src.shared.python.physics.ball_properties import BallProperties
from src.shared.python.physics.impact_model import (
    ImpactModelType,
    ImpactParameters,
    PostImpactState,
)
from src.shared.python.physics.swing_ball_flight_pipeline import (
    FlightSimulatorProtocol,
    PipelineResult,
    SwingBallFlightPipeline,
    SwingState,
    _LaunchConditionsDeriver,
    _TrajectoryMetricsExtractor,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_FORWARD_VELOCITY = np.array([45.0, 0.0, 0.0])  # 45 m/s ~ 100 mph
# Face normal: slightly downward so impact transfers horizontal momentum to the ball.
# A pure upward normal (0,0,1) with horizontal velocity produces zero approach speed.
_FACE_NORMAL = np.array([1.0, 0.0, 0.0])  # forward-facing face (approach direction)
_ZERO_SPIN = np.zeros(3)


def _make_swing(
    velocity: np.ndarray | None = None,
    orientation: np.ndarray | None = None,
    engine_name: str = "mock",
) -> SwingState:
    return SwingState(
        clubhead_velocity=velocity
        if velocity is not None
        else _FORWARD_VELOCITY.copy(),
        clubhead_angular_velocity=_ZERO_SPIN.copy(),
        clubhead_orientation=orientation
        if orientation is not None
        else _FACE_NORMAL.copy(),
        engine_name=engine_name,
    )


def _make_post_impact(speed: float = 60.0, angle_deg: float = 15.0) -> PostImpactState:
    """Return a PostImpactState with a ball speed at the given angle."""
    ca = math.cos(math.radians(angle_deg))
    sa = math.sin(math.radians(angle_deg))
    ball_vel = np.array([speed * ca, 0.0, speed * sa])
    return PostImpactState(
        ball_velocity=ball_vel,
        ball_angular_velocity=np.array([0.0, -300.0, 0.0]),  # backspin ~300 rad/s
        clubhead_velocity=np.array([5.0, 0.0, 0.0]),
        clubhead_angular_velocity=np.zeros(3),
        contact_duration=4.5e-4,
        energy_transfer=1500.0,
        impact_location=np.zeros(2),
    )


def _make_trajectory(n_points: int = 10) -> list[TrajectoryPoint]:
    """Return a synthetic parabolic trajectory."""
    points = []
    for i in range(n_points):
        t = i * 0.5
        x = 50.0 * t
        z = max(0.0, 20.0 * t - 9.81 * t**2 / 2)
        pos = np.array([x, 0.0, z])
        vel = np.array([50.0, 0.0, 20.0 - 9.81 * t])
        acc = np.array([0.0, 0.0, -9.81])
        points.append(
            TrajectoryPoint(
                time=t,
                position=pos,
                velocity=vel,
                acceleration=acc,
                forces={},
            )
        )
    return points


class _MockFlightSimulator:
    """Stub that returns a canned trajectory."""

    def __init__(self, trajectory: list[TrajectoryPoint] | None = None) -> None:
        self._trajectory = trajectory or _make_trajectory()
        self.calls: list[LaunchConditions] = []

    def simulate_trajectory(
        self,
        launch: LaunchConditions,
        max_time: float = 10.0,
        dt: float = 0.01,
    ) -> list[TrajectoryPoint]:
        self.calls.append(launch)
        return self._trajectory


class _MockImpactSolver:
    """Intercepts ImpactSolverAPI.solve_impact calls."""


def _make_pipeline(
    trajectory: list[TrajectoryPoint] | None = None,
    **kwargs,
) -> SwingBallFlightPipeline:
    """Return a pipeline with a mock flight simulator injected."""
    sim = _MockFlightSimulator(trajectory)
    return SwingBallFlightPipeline(flight_simulator=sim, **kwargs)


# ===========================================================================
# SwingState tests
# ===========================================================================


class TestSwingState:
    def test_defaults(self):
        swing = _make_swing()
        assert swing.engine_name == "mock"
        assert swing.clubhead_mass == pytest.approx(0.200)

    def test_metadata_defaults_to_empty(self):
        swing = _make_swing()
        assert swing.metadata == {}

    def test_custom_metadata(self):
        swing = SwingState(
            clubhead_velocity=np.array([40.0, 0.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([0.0, 0.0, 1.0]),
            metadata={"residual": 1e-6},
        )
        assert swing.metadata["residual"] == pytest.approx(1e-6)


# ===========================================================================
# PipelineResult tests
# ===========================================================================


class TestPipelineResult:
    def test_result_carries_swing_state(self):
        swing = _make_swing(engine_name="drake")
        post = _make_post_impact()
        lc = LaunchConditions(velocity=60.0, launch_angle=15.0)
        traj = _make_trajectory()
        result = PipelineResult(
            swing_state=swing,
            impact_state=post,
            launch_conditions=lc,
            trajectory=traj,
            carry_m=120.0,
            max_height_m=25.0,
            flight_time_s=5.0,
            landing_angle_deg=40.0,
        )
        assert result.swing_state.engine_name == "drake"
        assert result.carry_m == pytest.approx(120.0)

    def test_result_metadata_default_empty(self):
        swing = _make_swing()
        post = _make_post_impact()
        lc = LaunchConditions(velocity=60.0, launch_angle=15.0)
        result = PipelineResult(
            swing_state=swing,
            impact_state=post,
            launch_conditions=lc,
            trajectory=_make_trajectory(),
            carry_m=100.0,
            max_height_m=20.0,
            flight_time_s=4.5,
            landing_angle_deg=38.0,
        )
        assert result.metadata == {}


# ===========================================================================
# SwingBallFlightPipeline construction
# ===========================================================================


class TestPipelineConstruction:
    def test_default_construction(self):
        sim = _MockFlightSimulator()
        pipeline = SwingBallFlightPipeline(flight_simulator=sim)
        assert pipeline.impact_params is not None

    def test_custom_impact_params(self):
        params = ImpactParameters(cor=0.80)
        pipeline = _make_pipeline(impact_params=params)
        assert pipeline.impact_params.cor == pytest.approx(0.80)

    def test_negative_max_flight_time_raises(self):
        with pytest.raises((ValueError, PreconditionError)):
            SwingBallFlightPipeline(max_flight_time=-1.0)

    def test_flight_dt_larger_than_max_raises(self):
        with pytest.raises((ValueError, PreconditionError)):
            SwingBallFlightPipeline(max_flight_time=5.0, flight_dt=6.0)


# ===========================================================================
# run() happy-path tests
# ===========================================================================


class TestPipelineRun:
    def test_run_returns_pipeline_result(self):
        pipeline = _make_pipeline()
        result = pipeline.run(_make_swing())
        assert isinstance(result, PipelineResult)

    def test_run_trajectory_is_non_empty(self):
        pipeline = _make_pipeline()
        result = pipeline.run(_make_swing())
        assert len(result.trajectory) > 0

    def test_run_calls_flight_simulator(self):
        sim = _MockFlightSimulator()
        pipeline = SwingBallFlightPipeline(flight_simulator=sim)
        pipeline.run(_make_swing())
        assert len(sim.calls) == 1

    def test_run_result_contains_swing_state(self):
        pipeline = _make_pipeline()
        swing = _make_swing(engine_name="mujoco")
        result = pipeline.run(swing)
        assert result.swing_state.engine_name == "mujoco"

    def test_run_carry_computed_from_trajectory(self):
        trajectory = _make_trajectory(n_points=20)
        pipeline = _make_pipeline(trajectory=trajectory)
        result = pipeline.run(_make_swing())
        # carry = sqrt(x^2 + y^2) of last point
        last_x = trajectory[-1].position[0]
        expected_carry = float(np.sqrt(last_x**2))
        assert result.carry_m == pytest.approx(expected_carry, rel=1e-6)

    def test_run_metadata_includes_engine_name(self):
        pipeline = _make_pipeline()
        result = pipeline.run(_make_swing(engine_name="pinocchio"))
        assert result.metadata.get("engine") == "pinocchio"


# ===========================================================================
# DbC / precondition tests
# ===========================================================================


class TestPipelinePreconditions:
    def test_non_swing_state_raises(self):
        pipeline = _make_pipeline()
        with pytest.raises((TypeError, PreconditionError)):
            pipeline.run("not a SwingState")  # type: ignore[arg-type]

    def test_zero_clubhead_speed_raises(self):
        """Zero clubhead velocity → zero post-impact ball speed → contract failure."""
        pipeline = _make_pipeline()
        swing = _make_swing(velocity=np.zeros(3))
        with pytest.raises((ValueError, PreconditionError)):
            pipeline.run(swing)

    def test_negative_flight_dt_raises(self):
        with pytest.raises((ValueError, PreconditionError)):
            SwingBallFlightPipeline(flight_dt=-0.01)


# ===========================================================================
# _LaunchConditionsDeriver unit tests
# ===========================================================================


class TestLaunchConditionsDeriver:
    def setup_method(self):
        self.deriver = _LaunchConditionsDeriver()

    def test_forward_velocity_gives_positive_speed(self):
        post = _make_post_impact(speed=65.0, angle_deg=12.0)
        lc = self.deriver.derive(post)
        assert lc.velocity == pytest.approx(65.0, rel=1e-4)

    def test_launch_angle_matches_geometry(self):
        angle_deg = 15.0
        post = _make_post_impact(speed=60.0, angle_deg=angle_deg)
        lc = self.deriver.derive(post)
        assert lc.launch_angle == pytest.approx(angle_deg, abs=0.1)

    def test_zero_horizontal_speed_gives_90_degree_angle(self):
        post = _make_post_impact(speed=10.0, angle_deg=0.0)
        # Override to pure vertical
        post.ball_velocity[:] = np.array([0.0, 0.0, 10.0])
        lc = self.deriver.derive(post)
        assert lc.launch_angle == pytest.approx(90.0, abs=0.1)

    def test_spin_rate_matches_angular_velocity_magnitude(self):
        post = _make_post_impact()
        post.ball_angular_velocity[:] = np.array([0.0, -200.0, 0.0])
        lc = self.deriver.derive(post)
        assert lc.spin_rate == pytest.approx(200.0, rel=1e-4)


# ===========================================================================
# _TrajectoryMetricsExtractor unit tests
# ===========================================================================


class TestTrajectoryMetricsExtractor:
    def setup_method(self):
        self.extractor = _TrajectoryMetricsExtractor()

    def test_carry_empty_trajectory(self):
        assert self.extractor.carry_m([]) == pytest.approx(0.0)

    def test_carry_single_point(self):
        pt = TrajectoryPoint(
            time=5.0,
            position=np.array([200.0, 0.0, 0.0]),
            velocity=np.zeros(3),
            acceleration=np.zeros(3),
            forces={},
        )
        assert self.extractor.carry_m([pt]) == pytest.approx(200.0)

    def test_max_height(self):
        traj = _make_trajectory(n_points=10)
        h = self.extractor.max_height_m(traj)
        expected = max(p.position[2] for p in traj)
        assert h == pytest.approx(expected)

    def test_flight_time(self):
        traj = _make_trajectory(n_points=5)
        ft = self.extractor.flight_time_s(traj)
        assert ft == pytest.approx(traj[-1].time)

    def test_landing_angle_empty(self):
        assert self.extractor.landing_angle_deg([]) == pytest.approx(0.0)


# ===========================================================================
# FlightSimulatorProtocol structural subtyping
# ===========================================================================


class TestFlightSimulatorProtocol:
    def test_mock_satisfies_protocol(self):
        sim = _MockFlightSimulator()
        assert isinstance(sim, FlightSimulatorProtocol)
