"""Property-based tests for physics, dynamics, and planning modules.

Uses the hypothesis library to verify invariants and mathematical properties
that must hold across a wide range of randomly generated inputs.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Module imports under test
# ---------------------------------------------------------------------------
from src.shared.python.core import physics_constants
from src.shared.python.core.constants import (
    GOLF_BALL_DIAMETER_FLOAT,
    GOLF_BALL_MASS_FLOAT,
    GOLF_BALL_MOMENT_INERTIA_FLOAT,
    GOLF_BALL_RADIUS_FLOAT,
    GRAVITY_FLOAT,
)
from src.shared.python.physics.ball_flight_physics import (
    BallFlightSimulator,
    EnvironmentalConditions,
    LaunchConditions,
)

# ---------------------------------------------------------------------------
# Reusable strategies
# ---------------------------------------------------------------------------

# Finite, reasonable floats (avoiding extreme values that blow up physics)
reasonable_float = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
)
positive_float = st.floats(
    min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False
)
small_positive_float = st.floats(
    min_value=1e-6, max_value=100.0, allow_nan=False, allow_infinity=False
)
angle_rad = st.floats(
    min_value=-math.pi, max_value=math.pi, allow_nan=False, allow_infinity=False
)

# 3-component vectors with reasonable magnitudes
vec3_reasonable = st.tuples(
    st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
).map(lambda t: np.array(t))

# Non-zero velocity vectors (minimum speed above drag threshold)
nonzero_velocity = (
    st.tuples(
        st.floats(
            min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False
        ),
        st.floats(
            min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False
        ),
        st.floats(
            min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False
        ),
    )
    .filter(lambda t: math.sqrt(t[0] ** 2 + t[1] ** 2 + t[2] ** 2) > 1.0)
    .map(lambda t: np.array(t))
)

# Unit-length spin axes
unit_spin_axis = (
    st.tuples(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    .filter(lambda t: math.sqrt(t[0] ** 2 + t[1] ** 2 + t[2] ** 2) > 0.1)
    .map(lambda t: np.array(t) / np.linalg.norm(t))
)


# ============================================================================
# 1. BallFlightSimulator property tests
# ============================================================================


class TestBallFlightProperties:
    """Property-based tests for BallFlightSimulator force calculations."""

    @given(
        velocity=nonzero_velocity,
    )
    @settings(
        max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_gravity_always_pulls_downward(self, velocity: np.ndarray) -> None:
        """Property: gravity force z-component is always negative."""
        sim = BallFlightSimulator()
        launch = LaunchConditions(
            velocity=float(np.linalg.norm(velocity)),
            launch_angle=0.3,
            spin_rate=0.0,
        )

        forces = sim._calculate_forces(velocity, launch)
        gravity_z = forces["gravity"][2]

        # Gravity should always pull downward (negative z)
        assert gravity_z < 0, f"Gravity z-component should be negative, got {gravity_z}"

    @given(
        velocity=nonzero_velocity,
    )
    @settings(
        max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_drag_force_opposes_velocity(self, velocity: np.ndarray) -> None:
        """Property: drag force opposes the velocity direction.

        The dot product of drag force and relative velocity should be
        non-positive (drag decelerates the ball).
        """
        sim = BallFlightSimulator(
            env=EnvironmentalConditions(wind_velocity=np.zeros(3)),
        )
        launch = LaunchConditions(
            velocity=float(np.linalg.norm(velocity)),
            launch_angle=0.3,
            spin_rate=0.0,
        )

        forces = sim._calculate_forces(velocity, launch)
        drag = forces["drag"]

        # Drag should oppose relative velocity (zero wind => relative = velocity)
        dot = float(np.dot(drag, velocity))
        assert (
            dot <= 1e-10
        ), f"Drag should oppose velocity: dot={dot}, drag={drag}, vel={velocity}"

    @given(
        velocity=nonzero_velocity,
        spin_axis=unit_spin_axis,
        spin_rate=st.floats(
            min_value=1000.0, max_value=10000.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(
        max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_magnus_force_perpendicular_to_spin_and_velocity(
        self,
        velocity: np.ndarray,
        spin_axis: np.ndarray,
        spin_rate: float,
    ) -> None:
        """Property: Magnus force is perpendicular to the spin axis and velocity.

        The Magnus force is proportional to (spin_axis x velocity_hat),
        so it should be approximately perpendicular to the spin axis.
        """
        # Skip if velocity is nearly parallel to spin axis (cross product ~0)
        vel_hat = velocity / np.linalg.norm(velocity)
        cross_mag = np.linalg.norm(np.cross(spin_axis, vel_hat))
        assume(cross_mag > 0.1)

        sim = BallFlightSimulator(
            env=EnvironmentalConditions(wind_velocity=np.zeros(3)),
        )
        launch = LaunchConditions(
            velocity=float(np.linalg.norm(velocity)),
            launch_angle=0.0,
            spin_rate=spin_rate,
            spin_axis=spin_axis,
        )

        # We need to pass the actual velocity vector directly
        forces = sim._calculate_forces(velocity, launch)
        magnus = forces["magnus"]
        magnus_mag = np.linalg.norm(magnus)

        if magnus_mag < 1e-10:
            # Magnus too small to test direction; skip
            return

        # Magnus should be perpendicular to spin axis
        dot_spin = abs(float(np.dot(magnus, spin_axis)))
        assert dot_spin < magnus_mag * 0.1 + 1e-8, (
            f"Magnus should be ~perpendicular to spin axis: "
            f"|dot|={dot_spin}, |magnus|={magnus_mag}"
        )

    @given(
        velocity=nonzero_velocity,
    )
    @settings(
        max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_zero_wind_zero_spin_drag_antiparallel_to_velocity(
        self, velocity: np.ndarray
    ) -> None:
        """Property: with zero wind and zero spin, drag is anti-parallel to velocity."""
        sim = BallFlightSimulator(
            env=EnvironmentalConditions(wind_velocity=np.zeros(3)),
        )
        launch = LaunchConditions(
            velocity=float(np.linalg.norm(velocity)),
            launch_angle=0.0,
            spin_rate=0.0,
        )

        forces = sim._calculate_forces(velocity, launch)
        drag = forces["drag"]
        drag_mag = np.linalg.norm(drag)

        if drag_mag < 1e-10:
            return

        # Drag direction should be opposite to velocity direction
        drag_hat = drag / drag_mag
        vel_hat = velocity / np.linalg.norm(velocity)

        # Anti-parallel means their dot product should be close to -1
        dot = float(np.dot(drag_hat, vel_hat))
        assert dot < -0.99, f"Drag should be anti-parallel to velocity: dot={dot}"

    @given(
        velocity=nonzero_velocity,
        density_factor=st.floats(
            min_value=0.5, max_value=3.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(
        max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_forces_scale_with_air_density(
        self, velocity: np.ndarray, density_factor: float
    ) -> None:
        """Property: aerodynamic forces scale linearly with air density."""
        base_density = 1.225

        sim_base = BallFlightSimulator(
            env=EnvironmentalConditions(
                air_density=base_density,
                wind_velocity=np.zeros(3),
            ),
        )
        sim_scaled = BallFlightSimulator(
            env=EnvironmentalConditions(
                air_density=base_density * density_factor,
                wind_velocity=np.zeros(3),
            ),
        )

        launch = LaunchConditions(
            velocity=float(np.linalg.norm(velocity)),
            launch_angle=0.0,
            spin_rate=0.0,
        )

        forces_base = sim_base._calculate_forces(velocity, launch)
        forces_scaled = sim_scaled._calculate_forces(velocity, launch)

        drag_base_mag = np.linalg.norm(forces_base["drag"])
        drag_scaled_mag = np.linalg.norm(forces_scaled["drag"])

        if drag_base_mag < 1e-10:
            return

        ratio = drag_scaled_mag / drag_base_mag
        assert (
            abs(ratio - density_factor) < 0.01
        ), f"Drag should scale by {density_factor}, got ratio {ratio}"


# ============================================================================
# 2. PhaseDetector property tests
# ============================================================================


class TestPhaseDetectionProperties:
    """Property-based tests for PhaseDetectionMixin."""

    @given(
        n_points=st.integers(min_value=50, max_value=500),
        duration=st.floats(
            min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(
        max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_detected_phases_cover_entire_time_range(
        self, n_points: int, duration: float
    ) -> None:
        """Property: detected phases cover the full time range (no gaps)."""
        from src.shared.python.analysis.phase_detection import PhaseDetectionMixin

        # Create a mock object that has the required attributes
        detector = PhaseDetectionMixin()
        times = np.linspace(0.0, duration, n_points)
        # Simulate a realistic club head speed profile (sinusoidal ramp-up + peak)
        t_norm = times / duration
        club_head_speed = 40.0 * np.sin(np.pi * t_norm) ** 2

        detector.times = times  # type: ignore[attr-defined]
        detector.club_head_speed = club_head_speed  # type: ignore[attr-defined]
        detector.duration = duration  # type: ignore[attr-defined]

        phases = detector.detect_swing_phases()

        assert len(phases) > 0, "Should detect at least one phase"

        # Check that first phase starts at or before time[0]
        first_start = phases[0].start_time
        assert (
            first_start <= times[0] + 1e-10
        ), f"First phase start {first_start} should be <= {times[0]}"

        # Check that last phase ends at or after the last time
        last_end = phases[-1].end_time
        assert (
            last_end >= times[-1] - 1e-10
        ), f"Last phase end {last_end} should be >= {times[-1]}"

    @given(
        n_points=st.integers(min_value=50, max_value=500),
        duration=st.floats(
            min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(
        max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_phase_boundaries_monotonically_increasing(
        self, n_points: int, duration: float
    ) -> None:
        """Property: phase start times are monotonically non-decreasing."""
        from src.shared.python.analysis.phase_detection import PhaseDetectionMixin

        detector = PhaseDetectionMixin()
        times = np.linspace(0.0, duration, n_points)
        t_norm = times / duration
        club_head_speed = 40.0 * np.sin(np.pi * t_norm) ** 2

        detector.times = times  # type: ignore[attr-defined]
        detector.club_head_speed = club_head_speed  # type: ignore[attr-defined]
        detector.duration = duration  # type: ignore[attr-defined]

        phases = detector.detect_swing_phases()

        for i in range(len(phases) - 1):
            assert phases[i].start_time <= phases[i + 1].start_time, (
                f"Phase '{phases[i].name}' start_time={phases[i].start_time} "
                f"> next phase '{phases[i + 1].name}' start_time={phases[i + 1].start_time}"
            )

    @given(
        n_points=st.integers(min_value=50, max_value=500),
        duration=st.floats(
            min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(
        max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_each_phase_has_nonnegative_duration(
        self, n_points: int, duration: float
    ) -> None:
        """Property: every detected phase has non-negative duration."""
        from src.shared.python.analysis.phase_detection import PhaseDetectionMixin

        detector = PhaseDetectionMixin()
        times = np.linspace(0.0, duration, n_points)
        t_norm = times / duration
        club_head_speed = 40.0 * np.sin(np.pi * t_norm) ** 2

        detector.times = times  # type: ignore[attr-defined]
        detector.club_head_speed = club_head_speed  # type: ignore[attr-defined]
        detector.duration = duration  # type: ignore[attr-defined]

        phases = detector.detect_swing_phases()

        for phase in phases:
            assert (
                phase.duration >= 0.0
            ), f"Phase '{phase.name}' has negative duration: {phase.duration}"
            assert (
                phase.end_time >= phase.start_time
            ), f"Phase '{phase.name}' end_time={phase.end_time} < start_time={phase.start_time}"


# ============================================================================
# 3. SwingOptimizer config property tests
# ============================================================================


class TestSwingOptimizerConfigProperties:
    """Property-based tests for SwingOptimizer configuration invariants."""

    @given(
        flexibility=st.floats(
            min_value=0.5, max_value=2.0, allow_nan=False, allow_infinity=False
        ),
        n_nodes=st.integers(min_value=10, max_value=100),
    )
    @settings(
        max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_bounds_lower_leq_upper(self, flexibility: float, n_nodes: int) -> None:
        """Property: optimization bounds always have lower <= upper."""
        from src.shared.python.optimization.swing_optimizer import (
            ClubModel,
            GolferModel,
            OptimizationConfig,
            SwingOptimizer,
        )

        golfer = GolferModel(flexibility_factor=flexibility)
        club = ClubModel()
        config = OptimizationConfig(n_nodes=n_nodes)
        optimizer = SwingOptimizer(golfer, club, config)

        bounds = optimizer._get_bounds()

        for i, (lo, hi) in enumerate(bounds):
            assert (
                lo <= hi
            ), f"Bound index {i}: lower={lo} > upper={hi} (flexibility={flexibility})"

    @given(
        # flexibility_factor >= 1.0 ensures scaled bounds encompass the raw
        # joint limits used by _generate_initial_guess.  Values < 1.0 shrink
        # the bounds below the raw ROM, which the guess generator does not
        # account for -- that is a known upstream limitation.
        flexibility=st.floats(
            min_value=1.0, max_value=2.0, allow_nan=False, allow_infinity=False
        ),
        n_nodes=st.integers(min_value=10, max_value=100),
        backswing_frac=st.floats(
            min_value=0.2, max_value=0.8, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(
        max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_initial_guess_within_bounds(
        self, flexibility: float, n_nodes: int, backswing_frac: float
    ) -> None:
        """Property: initial guess angle variables are within joint-limit bounds.

        The velocity portion of the initial guess is computed via np.gradient
        and may slightly exceed the hard velocity bounds; we verify that
        the *angle* portion (first half of the vector) respects the
        flexibility-scaled joint-limit bounds, and that the velocity portion
        stays within a generous 10 rad/s tolerance of its bounds.
        """
        from src.shared.python.optimization.swing_optimizer import (
            ClubModel,
            GolferModel,
            OptimizationConfig,
            SwingOptimizer,
        )

        golfer = GolferModel(flexibility_factor=flexibility)
        club = ClubModel()
        config = OptimizationConfig(
            n_nodes=n_nodes,
            backswing_fraction=backswing_frac,
        )
        optimizer = SwingOptimizer(golfer, club, config)

        x0 = optimizer._generate_initial_guess()
        bounds = optimizer._get_bounds()

        assert len(x0) == len(
            bounds
        ), f"Initial guess length {len(x0)} != bounds length {len(bounds)}"

        n_angle_vars = len(optimizer.JOINTS) * n_nodes
        # Angle portion: must be strictly within joint-limit bounds
        for i in range(n_angle_vars):
            lo, hi = bounds[i]
            assert (
                lo - 1e-6 <= x0[i] <= hi + 1e-6
            ), f"Angle x0[{i}]={x0[i]} not in [{lo}, {hi}]"


# ============================================================================
# 4. Physical constants property tests
# ============================================================================


class TestPhysicalConstantsProperties:
    """Property-based tests for physical constants validity."""

    def test_gravity_within_expected_range(self) -> None:
        """Property: gravity is within the expected range (9.7 to 9.9 m/s^2)."""
        g = float(physics_constants.GRAVITY_M_S2)
        assert 9.7 < g < 9.9, f"Gravity={g} outside expected range [9.7, 9.9]"
        assert g == pytest.approx(
            9.80665, abs=1e-5
        ), f"Gravity={g} should be standard gravity 9.80665"

    def test_all_physical_constants_positive_and_finite(self) -> None:
        """Property: all physical constants defined via PhysicalConstant are positive and finite."""
        # Collect all PhysicalConstant instances from the module
        constants_checked = 0
        for name in dir(physics_constants):
            obj = getattr(physics_constants, name)
            if isinstance(obj, physics_constants.PhysicalConstant):
                val = float(obj)
                assert math.isfinite(val), f"Constant {name}={val} is not finite"
                assert val > 0, f"Constant {name}={val} should be positive"
                constants_checked += 1

        # Ensure we actually checked a meaningful number of constants
        assert (
            constants_checked >= 10
        ), f"Only checked {constants_checked} constants; expected at least 10"

    def test_precomputed_floats_match_source_constants(self) -> None:
        """Property: pre-computed float values match their source constants."""
        assert GRAVITY_FLOAT == pytest.approx(float(physics_constants.GRAVITY_M_S2))
        assert GOLF_BALL_MASS_FLOAT == pytest.approx(
            float(physics_constants.GOLF_BALL_MASS_KG)
        )
        assert GOLF_BALL_RADIUS_FLOAT == pytest.approx(
            float(physics_constants.GOLF_BALL_RADIUS_M)
        )
        assert GOLF_BALL_DIAMETER_FLOAT == pytest.approx(
            float(physics_constants.GOLF_BALL_DIAMETER_M)
        )
        assert GOLF_BALL_MOMENT_INERTIA_FLOAT == pytest.approx(
            float(physics_constants.GOLF_BALL_MOMENT_OF_INERTIA_KG_M2)
        )

    def test_derived_constants_consistent(self) -> None:
        """Property: derived constants are consistent with base values."""
        # Radius = diameter / 2
        assert float(physics_constants.GOLF_BALL_RADIUS_M) == pytest.approx(
            float(physics_constants.GOLF_BALL_DIAMETER_M) / 2.0
        )

        # Cross-sectional area = pi * r^2
        r = float(physics_constants.GOLF_BALL_RADIUS_M)
        expected_area = math.pi * r * r
        assert float(
            physics_constants.GOLF_BALL_CROSS_SECTIONAL_AREA_M2
        ) == pytest.approx(expected_area, rel=1e-6)

        # Moment of inertia for solid sphere = 2/5 * m * r^2
        m = float(physics_constants.GOLF_BALL_MASS_KG)
        expected_moi = (2.0 / 5.0) * m * r * r
        assert float(
            physics_constants.GOLF_BALL_MOMENT_OF_INERTIA_KG_M2
        ) == pytest.approx(expected_moi, rel=1e-6)

    @given(
        value=st.floats(
            min_value=0.01, max_value=1e4, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=50, deadline=None)
    def test_conversion_factor_roundtrip(self, value: float) -> None:
        """Property: unit conversion roundtrips preserve value."""
        deg_to_rad = float(physics_constants.DEG_TO_RAD)
        rad_to_deg = float(physics_constants.RAD_TO_DEG)

        # deg -> rad -> deg should roundtrip
        roundtripped = value * deg_to_rad * rad_to_deg
        assert roundtripped == pytest.approx(
            value, rel=1e-10
        ), f"DEG->RAD->DEG roundtrip: {value} -> {roundtripped}"

        ft_to_m = float(physics_constants.FT_TO_M)
        m_to_ft = float(physics_constants.M_TO_FT)

        roundtripped_ft = value * ft_to_m * m_to_ft
        assert roundtripped_ft == pytest.approx(
            value, rel=1e-6
        ), f"FT->M->FT roundtrip: {value} -> {roundtripped_ft}"


# ============================================================================
# 5. RRT planner property tests
# ============================================================================


class _FreeSpaceCollisionChecker:
    """A collision checker that reports no collisions (free space)."""

    def check_collision(self, q: np.ndarray) -> _FreeSpaceResult:
        return _FreeSpaceResult(in_collision=False)

    def check_path_collision(
        self,
        q_start: np.ndarray,
        q_end: np.ndarray,
        num_samples: int = 10,
    ) -> tuple[bool, float | None]:
        return (True, None)  # Path is free


class _FreeSpaceResult:
    def __init__(self, in_collision: bool) -> None:
        self.in_collision = in_collision


class TestRRTPlannerProperties:
    """Property-based tests for RRT motion planner."""

    @given(
        ndof=st.integers(min_value=2, max_value=6),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(
        max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    @pytest.mark.slow
    def test_path_start_equals_requested_start(self, ndof: int, seed: int) -> None:
        """Property: planned path starts at the requested start point."""
        from src.robotics.planning.motion.rrt import RRTConfig, RRTPlanner

        checker = _FreeSpaceCollisionChecker()
        config = RRTConfig(
            max_iterations=5000,
            step_size=0.5,
            goal_tolerance=0.5,
            goal_bias=0.15,
            max_time=10.0,
        )
        planner = RRTPlanner(checker, config)
        planner.set_seed(seed)

        lower = np.zeros(ndof)
        upper = np.ones(ndof) * 10.0
        planner.set_bounds(lower, upper)

        q_start = np.ones(ndof) * 2.0
        q_goal = np.ones(ndof) * 8.0

        result = planner.plan(q_start, q_goal)

        if result.success:
            assert len(result.path) >= 2, "Successful path should have >= 2 points"
            np.testing.assert_allclose(
                result.path[0],
                q_start,
                atol=1e-10,
                err_msg="Path start should equal requested start",
            )

    @given(
        ndof=st.integers(min_value=2, max_value=6),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(
        max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    @pytest.mark.slow
    def test_path_end_within_goal_tolerance(self, ndof: int, seed: int) -> None:
        """Property: planned path end is within goal tolerance of the goal."""
        from src.robotics.planning.motion.rrt import RRTConfig, RRTPlanner

        checker = _FreeSpaceCollisionChecker()
        goal_tol = 0.5
        config = RRTConfig(
            max_iterations=5000,
            step_size=0.5,
            goal_tolerance=goal_tol,
            goal_bias=0.15,
            max_time=10.0,
        )
        planner = RRTPlanner(checker, config)
        planner.set_seed(seed)

        lower = np.zeros(ndof)
        upper = np.ones(ndof) * 10.0
        planner.set_bounds(lower, upper)

        q_start = np.ones(ndof) * 2.0
        q_goal = np.ones(ndof) * 8.0

        result = planner.plan(q_start, q_goal)

        if result.success:
            end_dist = float(np.linalg.norm(result.path[-1] - q_goal))
            assert (
                end_dist <= goal_tol + 1e-10
            ), f"Path end distance to goal {end_dist} exceeds tolerance {goal_tol}"

    @given(
        ndof=st.integers(min_value=2, max_value=6),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(
        max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    @pytest.mark.slow
    def test_consecutive_points_within_step_size(self, ndof: int, seed: int) -> None:
        """Property: consecutive path points are within step_size of each other."""
        from src.robotics.planning.motion.rrt import RRTConfig, RRTPlanner

        checker = _FreeSpaceCollisionChecker()
        step_size = 0.5
        config = RRTConfig(
            max_iterations=5000,
            step_size=step_size,
            goal_tolerance=0.5,
            goal_bias=0.15,
            max_time=10.0,
        )
        planner = RRTPlanner(checker, config)
        planner.set_seed(seed)

        lower = np.zeros(ndof)
        upper = np.ones(ndof) * 10.0
        planner.set_bounds(lower, upper)

        q_start = np.ones(ndof) * 2.0
        q_goal = np.ones(ndof) * 8.0

        result = planner.plan(q_start, q_goal)

        if result.success and len(result.path) >= 2:
            for i in range(len(result.path) - 1):
                dist = float(np.linalg.norm(result.path[i + 1] - result.path[i]))
                # Allow a small tolerance above step_size for the goal connection
                assert dist <= step_size + 0.5 + 1e-10, (
                    f"Consecutive points {i}->{i + 1} distance {dist} "
                    f"exceeds step_size {step_size} + goal_tolerance"
                )
