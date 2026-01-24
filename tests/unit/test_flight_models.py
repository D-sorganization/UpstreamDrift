"""Unit tests for multi-model ball flight physics framework.

Tests all seven flight models for:
1. Basic trajectory generation
2. Consistent output format
3. Physical plausibility (carry, height, time bounds)
4. Wind effects
5. Model comparison consistency
"""

import math
import sys
from pathlib import Path
from src.shared.python.path_utils import get_repo_root, get_src_root


import numpy as np
import pytest

# Add shared directory to path
sys.path.insert(
    0, str(get_repo_root() / "src" / "shared" / "python")
)

from flight_models import (
    BallantyneModel,
    BallFlightModel,
    CharryL3Model,
    FlightModelRegistry,
    FlightModelType,
    FlightResult,
    JColeModel,
    MacDonaldHanzelyModel,
    NathanModel,
    RospieDLModel,
    TrajectoryPoint,
    UnifiedLaunchConditions,
    WaterlooPennerModel,
    compare_models,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def driver_launch() -> UnifiedLaunchConditions:
    """Standard driver launch conditions."""
    return UnifiedLaunchConditions.from_imperial(
        ball_speed_mph=163.0,
        launch_angle_deg=11.0,
        spin_rate_rpm=2500.0,
    )


@pytest.fixture
def iron7_launch() -> UnifiedLaunchConditions:
    """Standard 7-iron launch conditions."""
    return UnifiedLaunchConditions.from_imperial(
        ball_speed_mph=118.0,
        launch_angle_deg=16.0,
        spin_rate_rpm=7000.0,
    )


@pytest.fixture
def wedge_launch() -> UnifiedLaunchConditions:
    """Standard pitching wedge launch conditions."""
    return UnifiedLaunchConditions.from_imperial(
        ball_speed_mph=94.0,
        launch_angle_deg=23.0,
        spin_rate_rpm=9000.0,
    )


@pytest.fixture
def windy_launch() -> UnifiedLaunchConditions:
    """Driver launch with headwind."""
    return UnifiedLaunchConditions.from_imperial(
        ball_speed_mph=163.0,
        launch_angle_deg=11.0,
        spin_rate_rpm=2500.0,
        wind_speed_mph=15.0,  # 15 mph headwind
        wind_direction_deg=0.0,  # Headwind
    )


# =============================================================================
# Test Launch Conditions
# =============================================================================


class TestUnifiedLaunchConditions:
    """Tests for UnifiedLaunchConditions class."""

    def test_from_imperial_conversion(self) -> None:
        """Test imperial to SI conversion."""
        launch = UnifiedLaunchConditions.from_imperial(
            ball_speed_mph=100.0,
            launch_angle_deg=15.0,
        )
        # 100 mph ≈ 44.704 m/s
        assert abs(launch.ball_speed - 44.704) < 0.01
        # 15° in radians
        assert abs(launch.launch_angle - math.radians(15.0)) < 0.001

    def test_initial_velocity_vector(
        self, driver_launch: UnifiedLaunchConditions
    ) -> None:
        """Test initial velocity vector computation."""
        velocity = driver_launch.get_initial_velocity()

        assert velocity.shape == (3,)
        # Speed should match ball_speed
        speed = np.linalg.norm(velocity)
        assert abs(speed - driver_launch.ball_speed) < 0.1

        # Z component should be positive (upward launch)
        assert velocity[2] > 0

    def test_spin_vector_backspin(self) -> None:
        """Test spin vector for pure backspin."""
        launch = UnifiedLaunchConditions.from_imperial(
            ball_speed_mph=100.0,
            launch_angle_deg=10.0,
            spin_rate_rpm=3000.0,
            spin_axis_angle_deg=0.0,  # Pure backspin
        )
        spin = launch.get_spin_vector()

        assert spin.shape == (3,)
        # Pure backspin: axis pointing left (-Y)
        assert spin[1] < 0
        assert abs(spin[0]) < 1e-10  # No X component
        assert abs(spin[2]) < 1e-10  # No Z component

    def test_wind_vector_headwind(self) -> None:
        """Test wind vector for headwind."""
        launch = UnifiedLaunchConditions.from_imperial(
            ball_speed_mph=100.0,
            launch_angle_deg=10.0,
            wind_speed_mph=10.0,
            wind_direction_deg=0.0,  # Headwind
        )
        wind = launch.get_wind_vector()

        assert wind.shape == (3,)
        # Headwind: negative X direction
        assert wind[0] < 0
        assert abs(wind[1]) < 1e-10  # No Y component
        assert wind[2] == 0.0  # No vertical wind

    def test_wind_vector_crosswind(self) -> None:
        """Test wind vector for crosswind."""
        launch = UnifiedLaunchConditions.from_imperial(
            ball_speed_mph=100.0,
            launch_angle_deg=10.0,
            wind_speed_mph=10.0,
            wind_direction_deg=90.0,  # Right-to-left crosswind
        )
        wind = launch.get_wind_vector()

        # Crosswind: negative Y direction
        assert abs(wind[0]) < 0.1  # Small X component due to float precision
        assert wind[1] < 0  # Negative Y

    def test_no_wind(self) -> None:
        """Test wind vector when no wind."""
        launch = UnifiedLaunchConditions.from_imperial(
            ball_speed_mph=100.0,
            launch_angle_deg=10.0,
            wind_speed_mph=0.0,
        )
        wind = launch.get_wind_vector()

        assert np.allclose(wind, np.zeros(3))


# =============================================================================
# Test Individual Models
# =============================================================================


class TestWaterlooPennerModel:
    """Tests for Waterloo/Penner model."""

    def test_simulate_driver(self, driver_launch: UnifiedLaunchConditions) -> None:
        """Test driver trajectory simulation."""
        model = WaterlooPennerModel()
        result = model.simulate(driver_launch)

        assert isinstance(result, FlightResult)
        assert result.model_name == "Waterloo/Penner"
        assert len(result.trajectory) > 10
        assert result.carry_distance > 100  # At least 100m
        assert result.carry_distance < 350  # Less than 350m (realistic)
        assert result.max_height > 5  # At least 5m apex
        assert result.flight_time > 2.0  # At least 2 seconds

    def test_model_properties(self) -> None:
        """Test model property accessors."""
        model = WaterlooPennerModel()
        assert model.name == "Waterloo/Penner"
        assert "Waterloo" in model.description
        assert "Penner" in model.reference


class TestMacDonaldHanzelyModel:
    """Tests for MacDonald-Hanzely model."""

    def test_simulate_driver(self, driver_launch: UnifiedLaunchConditions) -> None:
        """Test driver trajectory simulation."""
        model = MacDonaldHanzelyModel()
        result = model.simulate(driver_launch)

        assert isinstance(result, FlightResult)
        assert result.model_name == "MacDonald-Hanzely"
        assert len(result.trajectory) > 10
        assert result.carry_distance > 100
        assert result.carry_distance < 350

    def test_spin_decay(self, driver_launch: UnifiedLaunchConditions) -> None:
        """Test that spin decay affects trajectory."""
        model_fast_decay = MacDonaldHanzelyModel(spin_decay_rate=0.2)
        model_slow_decay = MacDonaldHanzelyModel(spin_decay_rate=0.01)

        result_fast = model_fast_decay.simulate(driver_launch)
        result_slow = model_slow_decay.simulate(driver_launch)

        # Faster spin decay should result in less carry (less Magnus lift)
        assert result_fast.carry_distance < result_slow.carry_distance


class TestNathanModel:
    """Tests for Nathan model."""

    def test_simulate_driver(self, driver_launch: UnifiedLaunchConditions) -> None:
        """Test driver trajectory simulation."""
        model = NathanModel()
        result = model.simulate(driver_launch)

        assert isinstance(result, FlightResult)
        assert result.model_name == "Nathan"
        assert len(result.trajectory) > 10
        assert result.carry_distance > 100
        assert result.carry_distance < 350

    def test_reynolds_dependence(self) -> None:
        """Test Reynolds-dependent drag behavior."""
        model = NathanModel()

        # High speed (above critical Re)
        fast_launch = UnifiedLaunchConditions.from_imperial(
            ball_speed_mph=170.0, launch_angle_deg=10.0
        )
        # Lower speed (approaching critical Re)
        slow_launch = UnifiedLaunchConditions.from_imperial(
            ball_speed_mph=80.0, launch_angle_deg=10.0
        )

        fast_result = model.simulate(fast_launch)
        slow_result = model.simulate(slow_launch)

        # Both should produce valid trajectories
        assert fast_result.carry_distance > 0
        assert slow_result.carry_distance > 0


class TestBallantyneModel:
    """Tests for Ballantyne model."""

    def test_simulate_driver(self, driver_launch: UnifiedLaunchConditions) -> None:
        """Test driver trajectory simulation."""
        model = BallantyneModel()
        result = model.simulate(driver_launch)

        assert isinstance(result, FlightResult)
        assert result.model_name == "Ballantyne"
        assert result.carry_distance > 100
        assert result.carry_distance < 350


class TestJColeModel:
    """Tests for JCole model."""

    def test_simulate_driver(self, driver_launch: UnifiedLaunchConditions) -> None:
        """Test driver trajectory simulation."""
        model = JColeModel()
        result = model.simulate(driver_launch)

        assert isinstance(result, FlightResult)
        assert result.model_name == "JCole"
        assert result.carry_distance > 100
        assert result.carry_distance < 350


class TestRospieDLModel:
    """Tests for Rospie-DL model."""

    def test_simulate_driver(self, driver_launch: UnifiedLaunchConditions) -> None:
        """Test driver trajectory simulation."""
        model = RospieDLModel()
        result = model.simulate(driver_launch)

        assert isinstance(result, FlightResult)
        assert result.model_name == "Rospie-DL"
        assert result.carry_distance > 100
        assert result.carry_distance < 400  # DL model may predict higher


class TestCharryL3Model:
    """Tests for Charry-L3 model."""

    def test_simulate_driver(self, driver_launch: UnifiedLaunchConditions) -> None:
        """Test driver trajectory simulation."""
        model = CharryL3Model()
        result = model.simulate(driver_launch)

        assert isinstance(result, FlightResult)
        assert result.model_name == "Charry-L3"
        assert result.carry_distance > 100
        assert result.carry_distance < 350


# =============================================================================
# Test Model Registry
# =============================================================================


class TestFlightModelRegistry:
    """Tests for model registry."""

    def test_get_all_models(self) -> None:
        """Test getting all models."""
        models = FlightModelRegistry.get_all_models()

        assert len(models) == 7  # All 7 models
        assert all(isinstance(m, BallFlightModel) for m in models)

    def test_get_model_by_type(self) -> None:
        """Test getting model by type."""
        model = FlightModelRegistry.get_model(FlightModelType.WATERLOO_PENNER)
        assert isinstance(model, WaterlooPennerModel)

        model = FlightModelRegistry.get_model(FlightModelType.NATHAN)
        assert isinstance(model, NathanModel)

    def test_list_models(self) -> None:
        """Test listing models."""
        model_list = FlightModelRegistry.list_models()

        assert len(model_list) == 7
        assert all(len(item) == 3 for item in model_list)  # (enum, name, description)


# =============================================================================
# Test Model Comparison
# =============================================================================


class TestModelComparison:
    """Tests for multi-model comparison."""

    def test_compare_all_models(self, driver_launch: UnifiedLaunchConditions) -> None:
        """Test comparing all models."""
        results = compare_models(driver_launch)

        assert len(results) == 7
        assert all(isinstance(r, FlightResult) for r in results.values())

    def test_models_agree_on_direction(
        self, driver_launch: UnifiedLaunchConditions
    ) -> None:
        """Test that all models agree on general trajectory direction."""
        results = compare_models(driver_launch)

        for name, result in results.items():
            # All should have positive carry
            assert result.carry_distance > 0, f"{name} has negative carry"
            # All should have positive max height
            assert result.max_height > 0, f"{name} has negative max height"
            # All should have positive flight time
            assert result.flight_time > 0, f"{name} has negative flight time"

    def test_model_outputs_reasonable_range(
        self, driver_launch: UnifiedLaunchConditions
    ) -> None:
        """Test that models produce reasonably similar results."""
        results = compare_models(driver_launch)
        carries = [r.carry_distance for r in results.values()]

        # All carries should be within 2x of each other (reasonable tolerance)
        assert max(carries) / min(carries) < 2.0


# =============================================================================
# Test Physical Plausibility
# =============================================================================


class TestPhysicalPlausibility:
    """Tests for physical plausibility of results."""

    def test_higher_spin_more_carry_for_wedge(self) -> None:
        """Test that higher spin on wedge increases carry (up to a point)."""
        model = WaterlooPennerModel()

        low_spin = UnifiedLaunchConditions.from_imperial(
            ball_speed_mph=94.0,
            launch_angle_deg=23.0,
            spin_rate_rpm=5000.0,
        )
        high_spin = UnifiedLaunchConditions.from_imperial(
            ball_speed_mph=94.0,
            launch_angle_deg=23.0,
            spin_rate_rpm=9000.0,
        )

        low_result = model.simulate(low_spin)
        high_result = model.simulate(high_spin)

        # Higher spin should produce more carry on a wedge (more lift)
        assert high_result.carry_distance > low_result.carry_distance

    def test_trajectory_lands_at_ground_level(
        self, driver_launch: UnifiedLaunchConditions
    ) -> None:
        """Test that trajectory ends at ground level."""
        models = FlightModelRegistry.get_all_models()

        for model in models:
            result = model.simulate(driver_launch)
            final_pos = result.trajectory[-1].position

            # Should land at or very close to ground
            assert final_pos[2] <= 0.5, f"{model.name} final height: {final_pos[2]}"

    def test_landing_angle_is_descent(
        self, driver_launch: UnifiedLaunchConditions
    ) -> None:
        """Test that landing angle is a descent (positive angle down)."""
        models = FlightModelRegistry.get_all_models()

        for model in models:
            result = model.simulate(driver_launch)

            # Landing angle should be positive (descending)
            assert (
                result.landing_angle > 0
            ), f"{model.name} landing: {result.landing_angle}"
            # Should be less than 90°
            assert result.landing_angle < 90


# =============================================================================
# Test Trajectory Data Structure
# =============================================================================


class TestTrajectoryStructure:
    """Tests for trajectory data structure."""

    def test_trajectory_point_properties(
        self, driver_launch: UnifiedLaunchConditions
    ) -> None:
        """Test TrajectoryPoint properties."""
        model = WaterlooPennerModel()
        result = model.simulate(driver_launch)

        for point in result.trajectory:
            assert isinstance(point, TrajectoryPoint)
            assert isinstance(point.time, float)
            assert point.position.shape == (3,)
            assert point.velocity.shape == (3,)
            assert point.speed >= 0
            assert point.height == point.position[2]

    def test_trajectory_time_monotonic(
        self, driver_launch: UnifiedLaunchConditions
    ) -> None:
        """Test that trajectory time is monotonically increasing."""
        model = WaterlooPennerModel()
        result = model.simulate(driver_launch)

        times = [p.time for p in result.trajectory]
        for i in range(1, len(times)):
            assert times[i] > times[i - 1], f"Time not monotonic at {i}"

    def test_to_position_array(self, driver_launch: UnifiedLaunchConditions) -> None:
        """Test conversion to position array."""
        model = WaterlooPennerModel()
        result = model.simulate(driver_launch)

        positions = result.to_position_array()
        assert positions.shape == (len(result.trajectory), 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
