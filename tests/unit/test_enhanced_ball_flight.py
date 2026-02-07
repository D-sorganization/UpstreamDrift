"""Unit tests for EnhancedBallFlightSimulator.

Tests verify:
- Integration with aerodynamics module
- Toggle on/off functionality for air resistance
- Wind effects on ball flight
- Environment randomization for Monte Carlo
- Comparison between with/without aerodynamics

Following Pragmatic Programmer principles:
- Reversible: Test toggling aerodynamics
- Reusable: Test modular composition
- DRY: Share fixtures
- Orthogonal: Independent test cases
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.aerodynamics import (
    AerodynamicsConfig,
    RandomizationConfig,
    WindConfig,
)
from src.shared.python.ball_flight_physics import (
    BallProperties,
    EnhancedBallFlightSimulator,
    EnvironmentalConditions,
    LaunchConditions,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def default_launch() -> LaunchConditions:
    """Typical driver launch conditions."""
    return LaunchConditions(
        velocity=70.0,  # m/s
        launch_angle=math.radians(12.0),
        spin_rate=2500.0,  # rpm
    )


@pytest.fixture
def default_ball() -> BallProperties:
    """Standard golf ball."""
    return BallProperties()


@pytest.fixture
def default_env() -> EnvironmentalConditions:
    """Standard sea-level conditions."""
    return EnvironmentalConditions()


@pytest.fixture
def aero_enabled() -> AerodynamicsConfig:
    """Aerodynamics fully enabled."""
    return AerodynamicsConfig(enabled=True)


@pytest.fixture
def aero_disabled() -> AerodynamicsConfig:
    """Aerodynamics disabled (vacuum simulation)."""
    return AerodynamicsConfig(enabled=False)


@pytest.fixture
def wind_headwind() -> WindConfig:
    """10 m/s headwind."""
    return WindConfig(base_velocity=np.array([-10.0, 0.0, 0.0]))


@pytest.fixture
def wind_tailwind() -> WindConfig:
    """10 m/s tailwind."""
    return WindConfig(base_velocity=np.array([10.0, 0.0, 0.0]))


@pytest.fixture
def gusty_wind() -> WindConfig:
    """Wind with gusts enabled."""
    return WindConfig(
        base_velocity=np.array([5.0, 0.0, 0.0]),
        gusts_enabled=True,
        gust_intensity=0.5,
        gust_frequency=0.5,
    )


# =============================================================================
# Basic Initialization Tests
# =============================================================================


class TestEnhancedSimulatorInit:
    """Tests for enhanced simulator initialization."""

    def test_default_initialization(self) -> None:
        """Test simulator initializes with defaults."""
        sim = EnhancedBallFlightSimulator()
        assert sim.ball is not None
        assert sim.environment is not None
        assert sim.aero_config is not None

    def test_custom_ball(self, default_ball: BallProperties) -> None:
        """Test with custom ball properties."""
        sim = EnhancedBallFlightSimulator(ball=default_ball)
        assert sim.ball.mass == default_ball.mass

    def test_custom_aero_config(self, aero_disabled: AerodynamicsConfig) -> None:
        """Test with custom aerodynamics config."""
        sim = EnhancedBallFlightSimulator(aero_config=aero_disabled)
        assert sim.aero_config.enabled is False

    def test_custom_wind_config(self, wind_headwind: WindConfig) -> None:
        """Test with custom wind config."""
        sim = EnhancedBallFlightSimulator(wind_config=wind_headwind)
        assert sim.wind_config.speed == pytest.approx(10.0)


# =============================================================================
# Aerodynamics Toggle Tests
# =============================================================================


class TestAerodynamicsToggle:
    """Tests for toggling aerodynamics on/off."""

    def test_aero_on_produces_drag(
        self,
        default_launch: LaunchConditions,
        aero_enabled: AerodynamicsConfig,
    ) -> None:
        """Test enabled aerodynamics produces drag force."""
        sim = EnhancedBallFlightSimulator(aero_config=aero_enabled)
        traj = sim.simulate_trajectory(default_launch, max_time=1.0, dt=0.01)

        # Check drag is non-zero in middle of flight
        mid_point = traj[len(traj) // 2]
        drag_magnitude = np.linalg.norm(mid_point.forces["drag"])
        assert drag_magnitude > 0

    def test_aero_off_no_drag(
        self,
        default_launch: LaunchConditions,
        aero_disabled: AerodynamicsConfig,
    ) -> None:
        """Test disabled aerodynamics produces zero drag."""
        sim = EnhancedBallFlightSimulator(aero_config=aero_disabled)
        traj = sim.simulate_trajectory(default_launch, max_time=1.0, dt=0.01)

        # All drag forces should be zero
        for point in traj:
            drag_magnitude = np.linalg.norm(point.forces["drag"])
            assert drag_magnitude == pytest.approx(0.0)

    def test_aero_toggle_affects_distance(
        self,
        default_launch: LaunchConditions,
        aero_enabled: AerodynamicsConfig,
        aero_disabled: AerodynamicsConfig,
    ) -> None:
        """Test toggling aerodynamics changes carry distance."""
        sim_with = EnhancedBallFlightSimulator(aero_config=aero_enabled)
        sim_without = EnhancedBallFlightSimulator(aero_config=aero_disabled)

        traj_with = sim_with.simulate_trajectory(default_launch, max_time=8.0)
        traj_without = sim_without.simulate_trajectory(default_launch, max_time=8.0)

        carry_with = sim_with.calculate_carry_distance(traj_with)
        carry_without = sim_without.calculate_carry_distance(traj_without)

        # Without drag, ball should go further (vacuum trajectory)
        assert carry_without > carry_with

    def test_individual_force_toggles(
        self,
        default_launch: LaunchConditions,
    ) -> None:
        """Test individual force components can be toggled (Orthogonal)."""
        # Drag only
        config_drag = AerodynamicsConfig(
            drag_enabled=True, lift_enabled=False, magnus_enabled=False
        )
        sim_drag = EnhancedBallFlightSimulator(aero_config=config_drag)
        traj = sim_drag.simulate_trajectory(default_launch, max_time=1.0)

        point = traj[len(traj) // 2]
        assert np.linalg.norm(point.forces["drag"]) > 0
        assert np.linalg.norm(point.forces["lift"]) == pytest.approx(0.0)
        assert np.linalg.norm(point.forces["magnus"]) == pytest.approx(0.0)


# =============================================================================
# Wind Effect Tests
# =============================================================================


class TestWindEffects:
    """Tests for wind effects on ball flight."""

    def test_headwind_reduces_distance(
        self,
        default_launch: LaunchConditions,
        wind_headwind: WindConfig,
    ) -> None:
        """Test headwind reduces carry distance."""
        sim_no_wind = EnhancedBallFlightSimulator()
        sim_headwind = EnhancedBallFlightSimulator(wind_config=wind_headwind)

        traj_no_wind = sim_no_wind.simulate_trajectory(default_launch, max_time=8.0)
        traj_headwind = sim_headwind.simulate_trajectory(default_launch, max_time=8.0)

        carry_no_wind = sim_no_wind.calculate_carry_distance(traj_no_wind)
        carry_headwind = sim_headwind.calculate_carry_distance(traj_headwind)

        assert carry_headwind < carry_no_wind

    def test_tailwind_increases_distance(
        self,
        default_launch: LaunchConditions,
        wind_tailwind: WindConfig,
    ) -> None:
        """Test tailwind increases carry distance."""
        sim_no_wind = EnhancedBallFlightSimulator()
        sim_tailwind = EnhancedBallFlightSimulator(wind_config=wind_tailwind)

        traj_no_wind = sim_no_wind.simulate_trajectory(default_launch, max_time=8.0)
        traj_tailwind = sim_tailwind.simulate_trajectory(default_launch, max_time=8.0)

        carry_no_wind = sim_no_wind.calculate_carry_distance(traj_no_wind)
        carry_tailwind = sim_tailwind.calculate_carry_distance(traj_tailwind)

        assert carry_tailwind > carry_no_wind

    def test_crosswind_causes_lateral_drift(
        self,
        default_launch: LaunchConditions,
    ) -> None:
        """Test crosswind causes lateral ball movement."""
        crosswind = WindConfig(base_velocity=np.array([0.0, 10.0, 0.0]))
        sim = EnhancedBallFlightSimulator(wind_config=crosswind)

        traj = sim.simulate_trajectory(default_launch, max_time=6.0)

        # Ball should drift in y-direction
        final_y = traj[-1].position[1]
        assert abs(final_y) > 1.0  # More than 1m lateral drift

    def test_gusty_wind_adds_variation(
        self,
        default_launch: LaunchConditions,
        gusty_wind: WindConfig,
    ) -> None:
        """Test gusty wind adds trajectory variation."""
        sim1 = EnhancedBallFlightSimulator(wind_config=gusty_wind, seed=111)
        sim2 = EnhancedBallFlightSimulator(wind_config=gusty_wind, seed=222)

        traj1 = sim1.simulate_trajectory(default_launch, max_time=6.0)
        traj2 = sim2.simulate_trajectory(default_launch, max_time=6.0)

        carry1 = sim1.calculate_carry_distance(traj1)
        carry2 = sim2.calculate_carry_distance(traj2)

        # Different seeds should give different results
        assert carry1 != carry2


# =============================================================================
# Comparison Method Tests
# =============================================================================


class TestComparisonMethod:
    """Tests for simulate_with_comparison method."""

    def test_comparison_returns_both_trajectories(
        self,
        default_launch: LaunchConditions,
    ) -> None:
        """Test comparison method returns both trajectories."""
        sim = EnhancedBallFlightSimulator()
        result = sim.simulate_with_comparison(default_launch, max_time=6.0)

        assert "with_aero" in result
        assert "without_aero" in result
        assert len(result["with_aero"]) > 0
        assert len(result["without_aero"]) > 0

    def test_comparison_shows_drag_effect(
        self,
        default_launch: LaunchConditions,
    ) -> None:
        """Test comparison clearly shows drag effect."""
        sim = EnhancedBallFlightSimulator()
        result = sim.simulate_with_comparison(default_launch, max_time=8.0)

        carry_with = sim.calculate_carry_distance(result["with_aero"])
        carry_without = sim.calculate_carry_distance(result["without_aero"])

        # Significant difference expected
        assert carry_without > carry_with * 1.1  # At least 10% more


# =============================================================================
# Monte Carlo Simulation Tests
# =============================================================================


class TestMonteCarlo:
    """Tests for Monte Carlo simulation."""

    def test_monte_carlo_produces_multiple_results(
        self,
        default_launch: LaunchConditions,
    ) -> None:
        """Test Monte Carlo produces requested number of samples."""
        rand_config = RandomizationConfig(
            enabled=True,
            air_density_variance=0.05,
        )
        sim = EnhancedBallFlightSimulator(
            randomization_config=rand_config,
            seed=42,
        )

        results = sim.monte_carlo_simulation(default_launch, n_samples=10, max_time=6.0)

        assert len(results) == 10

    def test_monte_carlo_results_have_variation(
        self,
        default_launch: LaunchConditions,
    ) -> None:
        """Test Monte Carlo results show variation."""
        rand_config = RandomizationConfig(
            enabled=True,
            air_density_variance=0.1,
            wind_variance=0.2,
        )
        gusty = WindConfig(
            base_velocity=np.array([5.0, 0.0, 0.0]),
            gusts_enabled=True,
            gust_intensity=0.3,
        )
        sim = EnhancedBallFlightSimulator(
            randomization_config=rand_config,
            wind_config=gusty,
            seed=42,
        )

        results = sim.monte_carlo_simulation(default_launch, n_samples=20, max_time=6.0)

        carries = [r["carry_distance"] for r in results]

        # Should have variation
        assert max(carries) != min(carries)

    def test_monte_carlo_each_result_has_analysis(
        self,
        default_launch: LaunchConditions,
    ) -> None:
        """Test each Monte Carlo result contains analysis."""
        rand_config = RandomizationConfig(enabled=True, air_density_variance=0.05)
        sim = EnhancedBallFlightSimulator(randomization_config=rand_config, seed=42)

        results = sim.monte_carlo_simulation(default_launch, n_samples=5, max_time=6.0)

        for result in results:
            assert "carry_distance" in result
            assert "max_height" in result
            assert "flight_time" in result
            assert "run" in result


# =============================================================================
# Analysis Method Tests
# =============================================================================


class TestAnalysisMethods:
    """Tests for trajectory analysis methods."""

    def test_analyze_trajectory_complete(
        self,
        default_launch: LaunchConditions,
    ) -> None:
        """Test analyze_trajectory returns all metrics."""
        sim = EnhancedBallFlightSimulator()
        traj = sim.simulate_trajectory(default_launch, max_time=6.0)
        analysis = sim.analyze_trajectory(traj)

        assert "carry_distance" in analysis
        assert "max_height" in analysis
        assert "flight_time" in analysis
        assert "landing_angle" in analysis
        assert "apex_time" in analysis
        assert "trajectory_points" in analysis

    def test_realistic_driver_metrics(
        self,
        default_launch: LaunchConditions,
    ) -> None:
        """Test metrics are in realistic ranges for driver shot."""
        sim = EnhancedBallFlightSimulator()
        traj = sim.simulate_trajectory(default_launch, max_time=8.0)
        analysis = sim.analyze_trajectory(traj)

        # Reasonable ranges for a 70 m/s driver shot
        assert 100 < analysis["carry_distance"] < 300  # meters
        assert 10 < analysis["max_height"] < 60  # meters
        assert 3 < analysis["flight_time"] < 10  # seconds
        assert 20 < analysis["landing_angle"] < 70  # degrees


# =============================================================================
# Reproducibility Tests
# =============================================================================


class TestReproducibility:
    """Tests for simulation reproducibility."""

    def test_same_seed_same_result(
        self,
        default_launch: LaunchConditions,
        gusty_wind: WindConfig,
    ) -> None:
        """Test same seed produces identical results."""
        sim1 = EnhancedBallFlightSimulator(wind_config=gusty_wind, seed=42)
        sim2 = EnhancedBallFlightSimulator(wind_config=gusty_wind, seed=42)

        traj1 = sim1.simulate_trajectory(default_launch, max_time=6.0)
        traj2 = sim2.simulate_trajectory(default_launch, max_time=6.0)

        # Should be identical
        for p1, p2 in zip(traj1, traj2, strict=True):
            np.testing.assert_array_almost_equal(p1.position, p2.position)
            np.testing.assert_array_almost_equal(p1.velocity, p2.velocity)

    def test_different_seed_different_result(
        self,
        default_launch: LaunchConditions,
        gusty_wind: WindConfig,
    ) -> None:
        """Test different seeds produce different results."""
        sim1 = EnhancedBallFlightSimulator(wind_config=gusty_wind, seed=111)
        sim2 = EnhancedBallFlightSimulator(wind_config=gusty_wind, seed=222)

        traj1 = sim1.simulate_trajectory(default_launch, max_time=6.0)
        traj2 = sim2.simulate_trajectory(default_launch, max_time=6.0)

        carry1 = sim1.calculate_carry_distance(traj1)
        carry2 = sim2.calculate_carry_distance(traj2)

        assert carry1 != carry2


# =============================================================================
# Physics Validation Tests
# =============================================================================


class TestPhysicsValidation:
    """Tests validating physics correctness."""

    def test_backspin_adds_lift(self) -> None:
        """Test backspin produces upward lift."""
        no_spin = LaunchConditions(
            velocity=60.0, launch_angle=math.radians(12.0), spin_rate=0.0
        )
        with_spin = LaunchConditions(
            velocity=60.0, launch_angle=math.radians(12.0), spin_rate=3000.0
        )

        sim = EnhancedBallFlightSimulator()

        traj_no_spin = sim.simulate_trajectory(no_spin, max_time=8.0)
        traj_with_spin = sim.simulate_trajectory(with_spin, max_time=8.0)

        max_h_no_spin = sim.calculate_max_height(traj_no_spin)
        max_h_with_spin = sim.calculate_max_height(traj_with_spin)

        # Backspin should increase max height
        assert max_h_with_spin > max_h_no_spin

    def test_higher_velocity_longer_distance(self) -> None:
        """Test higher velocity produces longer carry."""
        slow = LaunchConditions(
            velocity=40.0, launch_angle=math.radians(12.0), spin_rate=2500.0
        )
        fast = LaunchConditions(
            velocity=70.0, launch_angle=math.radians(12.0), spin_rate=2500.0
        )

        sim = EnhancedBallFlightSimulator()

        traj_slow = sim.simulate_trajectory(slow, max_time=8.0)
        traj_fast = sim.simulate_trajectory(fast, max_time=8.0)

        carry_slow = sim.calculate_carry_distance(traj_slow)
        carry_fast = sim.calculate_carry_distance(traj_fast)

        assert carry_fast > carry_slow

    def test_gravity_only_parabolic(self) -> None:
        """Test gravity-only trajectory is approximately parabolic."""
        launch = LaunchConditions(
            velocity=50.0, launch_angle=math.radians(45.0), spin_rate=0.0
        )

        # Disable all aerodynamics
        config = AerodynamicsConfig(enabled=False)
        sim = EnhancedBallFlightSimulator(aero_config=config)

        traj = sim.simulate_trajectory(launch, max_time=10.0, dt=0.01)

        # In vacuum at 45 degrees, range = v^2/g
        expected_range = (50.0**2) / 9.81
        actual_range = sim.calculate_carry_distance(traj)

        # Should be close (within 1%)
        assert actual_range == pytest.approx(expected_range, rel=0.01)
