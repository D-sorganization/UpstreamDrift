"""Unit tests for PutterStroke module.

TDD Tests - These tests define the expected behavior of putter-ball
interaction including strike, energy transfer, and different putter types.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.engines.physics_engines.putting_green.python.ball_roll_physics import BallState
from src.engines.physics_engines.putting_green.python.putter_stroke import (
    PutterStroke,
    PutterType,
    StrokeParameters,
)


class TestPutterType:
    """Tests for PutterType enumeration."""

    def test_putter_types_exist(self) -> None:
        """Verify common putter types are defined."""
        assert hasattr(PutterType, "BLADE")
        assert hasattr(PutterType, "MALLET")
        assert hasattr(PutterType, "FACE_BALANCED")
        assert hasattr(PutterType, "TOE_HANG")

    def test_putter_type_properties(self) -> None:
        """Each putter type should have characteristic properties."""
        blade = PutterType.BLADE
        mallet = PutterType.MALLET

        # Blade typically has more toe hang
        assert blade.default_toe_hang > mallet.default_toe_hang


class TestStrokeParameters:
    """Tests for StrokeParameters dataclass."""

    def test_stroke_parameters_creation(self) -> None:
        """StrokeParameters should store stroke details."""
        params = StrokeParameters(
            speed=2.0,  # m/s clubhead speed at impact
            direction=np.array([1.0, 0.0]),
            face_angle=0.0,  # degrees, 0 = square
            attack_angle=0.0,  # degrees, 0 = level
            impact_location=np.array([0.0, 0.0]),  # center of face
        )
        assert params.speed == 2.0
        assert params.face_angle == 0.0

    def test_direction_normalization(self) -> None:
        """Direction should be automatically normalized."""
        params = StrokeParameters(
            speed=2.0,
            direction=np.array([3.0, 4.0]),
            face_angle=0.0,
            attack_angle=0.0,
        )
        assert np.isclose(np.linalg.norm(params.direction), 1.0)

    def test_stroke_from_backstroke_length(self) -> None:
        """Should create stroke from backstroke length."""
        params = StrokeParameters.from_backstroke_length(
            backstroke_length=0.15,  # 15cm backstroke
            direction=np.array([1.0, 0.0]),
            tempo=1.0,
        )
        # Longer backstroke = faster speed
        assert params.speed > 0

    def test_stroke_from_target_distance(self) -> None:
        """Should estimate stroke for target distance."""
        params = StrokeParameters.for_target_distance(
            distance=5.0,  # 5 meter putt
            stimp_rating=10.0,
            direction=np.array([1.0, 0.0]),
        )
        assert params.speed > 0

    def test_face_angle_affects_direction(self) -> None:
        """Non-zero face angle should change effective direction."""
        square = StrokeParameters(
            speed=2.0,
            direction=np.array([1.0, 0.0]),
            face_angle=0.0,
            attack_angle=0.0,
        )
        open_face = StrokeParameters(
            speed=2.0,
            direction=np.array([1.0, 0.0]),
            face_angle=5.0,  # 5 degrees open
            attack_angle=0.0,
        )

        # Effective directions should differ
        assert not np.allclose(
            square.effective_direction, open_face.effective_direction
        )


class TestPutterStroke:
    """Tests for PutterStroke class."""

    @pytest.fixture
    def putter(self) -> PutterStroke:
        """Create default putter."""
        return PutterStroke()

    @pytest.fixture
    def blade_putter(self) -> PutterStroke:
        """Create blade-style putter."""
        return PutterStroke(putter_type=PutterType.BLADE)

    @pytest.fixture
    def mallet_putter(self) -> PutterStroke:
        """Create mallet-style putter."""
        return PutterStroke(putter_type=PutterType.MALLET)

    def test_default_putter_properties(self, putter: PutterStroke) -> None:
        """Default putter should have sensible properties."""
        assert putter.mass > 0
        assert putter.loft > 0  # Putters have small loft (typically 3-4 degrees)
        assert putter.loft < 10  # But not too much
        assert putter.coefficient_of_restitution > 0
        assert putter.coefficient_of_restitution <= 1

    def test_execute_stroke_creates_ball_state(self, putter: PutterStroke) -> None:
        """Executing stroke should create initial ball state."""
        ball_position = np.array([5.0, 10.0])
        params = StrokeParameters(
            speed=2.0,
            direction=np.array([1.0, 0.0]),
            face_angle=0.0,
            attack_angle=0.0,
        )

        ball_state = putter.execute_stroke(ball_position, params)

        assert isinstance(ball_state, BallState)
        assert np.allclose(ball_state.position, ball_position)
        assert ball_state.speed > 0

    def test_energy_transfer(self, putter: PutterStroke) -> None:
        """Putter should transfer energy to ball with some loss."""
        params = StrokeParameters(
            speed=2.0,
            direction=np.array([1.0, 0.0]),
            face_angle=0.0,
            attack_angle=0.0,
        )

        ball_state = putter.execute_stroke(np.array([0.0, 0.0]), params)

        # Ball speed should be related to clubhead speed
        # With COR and mass ratio, ball gets majority of energy
        assert ball_state.speed > 0
        assert ball_state.speed < params.speed * 2  # Physics constraint

    def test_coefficient_of_restitution_effect(self) -> None:
        """Higher COR should result in faster ball speed."""
        low_cor = PutterStroke(coefficient_of_restitution=0.7)
        high_cor = PutterStroke(coefficient_of_restitution=0.9)

        params = StrokeParameters(
            speed=2.0,
            direction=np.array([1.0, 0.0]),
            face_angle=0.0,
            attack_angle=0.0,
        )

        ball_low = low_cor.execute_stroke(np.array([0.0, 0.0]), params)
        ball_high = high_cor.execute_stroke(np.array([0.0, 0.0]), params)

        assert ball_high.speed > ball_low.speed

    def test_loft_imparts_backspin(self, putter: PutterStroke) -> None:
        """Putter loft should impart backspin to ball."""
        params = StrokeParameters(
            speed=2.0,
            direction=np.array([1.0, 0.0]),
            face_angle=0.0,
            attack_angle=0.0,
        )

        ball_state = putter.execute_stroke(np.array([0.0, 0.0]), params)

        # Should have some backspin (perpendicular to direction)
        spin_magnitude = np.linalg.norm(ball_state.spin)
        assert spin_magnitude > 0

    def test_attack_angle_affects_spin(self, putter: PutterStroke) -> None:
        """Descending attack angle should increase backspin."""
        level_params = StrokeParameters(
            speed=2.0,
            direction=np.array([1.0, 0.0]),
            face_angle=0.0,
            attack_angle=0.0,
        )
        descending_params = StrokeParameters(
            speed=2.0,
            direction=np.array([1.0, 0.0]),
            face_angle=0.0,
            attack_angle=-3.0,  # Descending blow
        )

        level_state = putter.execute_stroke(np.array([0.0, 0.0]), level_params)
        desc_state = putter.execute_stroke(np.array([0.0, 0.0]), descending_params)

        # Descending should have more backspin
        assert np.linalg.norm(desc_state.spin) > np.linalg.norm(level_state.spin)

    def test_off_center_strike_creates_sidespin(self, putter: PutterStroke) -> None:
        """Off-center impact should create sidespin."""
        center_params = StrokeParameters(
            speed=2.0,
            direction=np.array([1.0, 0.0]),
            face_angle=0.0,
            attack_angle=0.0,
            impact_location=np.array([0.0, 0.0]),  # Center
        )
        toe_params = StrokeParameters(
            speed=2.0,
            direction=np.array([1.0, 0.0]),
            face_angle=0.0,
            attack_angle=0.0,
            impact_location=np.array([0.02, 0.0]),  # Toe hit
        )

        center_state = putter.execute_stroke(np.array([0.0, 0.0]), center_params)
        toe_state = putter.execute_stroke(np.array([0.0, 0.0]), toe_params)

        # Toe hit should have sidespin (z-component)
        assert abs(toe_state.spin[2]) > abs(center_state.spin[2])

    def test_off_center_reduces_ball_speed(self, putter: PutterStroke) -> None:
        """Off-center impact should reduce ball speed (gear effect)."""
        center = StrokeParameters(
            speed=2.0,
            direction=np.array([1.0, 0.0]),
            face_angle=0.0,
            attack_angle=0.0,
            impact_location=np.array([0.0, 0.0]),
        )
        off_center = StrokeParameters(
            speed=2.0,
            direction=np.array([1.0, 0.0]),
            face_angle=0.0,
            attack_angle=0.0,
            impact_location=np.array([0.03, 0.0]),  # Far off center
        )

        center_state = putter.execute_stroke(np.array([0.0, 0.0]), center)
        off_state = putter.execute_stroke(np.array([0.0, 0.0]), off_center)

        # Off-center should be slower
        assert off_state.speed < center_state.speed

    def test_putter_type_affects_sweetspot(self) -> None:
        """Different putter types have different sweet spot sizes."""
        blade = PutterStroke(putter_type=PutterType.BLADE)
        mallet = PutterStroke(putter_type=PutterType.MALLET)

        # Mallet typically has larger sweet spot
        assert mallet.sweet_spot_size > blade.sweet_spot_size

    def test_insert_material_affects_feel(self) -> None:
        """Different insert materials affect COR and feel."""
        metal_face = PutterStroke(insert_type="metal")
        soft_insert = PutterStroke(insert_type="polymer")

        # Metal typically has higher COR
        assert (
            metal_face.coefficient_of_restitution
            > soft_insert.coefficient_of_restitution
        )

    def test_compute_launch_velocity(self, putter: PutterStroke) -> None:
        """Should compute launch velocity from impact parameters."""
        params = StrokeParameters(
            speed=2.0,
            direction=np.array([1.0, 0.0]),
            face_angle=0.0,
            attack_angle=0.0,
        )

        launch_v = putter.compute_launch_velocity(params)

        # Launch velocity should be 2D on putting surface
        assert launch_v.shape == (2,) or launch_v.shape == (3,)
        assert np.linalg.norm(launch_v) > 0

    def test_compute_initial_spin(self, putter: PutterStroke) -> None:
        """Should compute initial spin from impact parameters."""
        params = StrokeParameters(
            speed=2.0,
            direction=np.array([1.0, 0.0]),
            face_angle=0.0,
            attack_angle=0.0,
        )

        spin = putter.compute_initial_spin(params)

        # Spin should be 3D vector
        assert spin.shape == (3,)
        assert np.all(np.isfinite(spin))


class TestStrokeSimulationHelpers:
    """Tests for stroke simulation helper functions."""

    @pytest.fixture
    def putter(self) -> PutterStroke:
        return PutterStroke()

    def test_estimate_required_speed(self, putter: PutterStroke) -> None:
        """Should estimate required speed for given distance."""
        distance = 5.0  # meters
        stimp = 10.0

        speed = putter.estimate_required_speed(distance, stimp)

        assert speed > 0
        # Longer distance = more speed needed
        speed_long = putter.estimate_required_speed(10.0, stimp)
        assert speed_long > speed

    def test_estimate_required_speed_with_slope(self, putter: PutterStroke) -> None:
        """Should account for slope in speed estimation."""
        distance = 5.0
        stimp = 10.0

        uphill = putter.estimate_required_speed(distance, stimp, slope_percent=3.0)
        downhill = putter.estimate_required_speed(distance, stimp, slope_percent=-3.0)

        # Uphill needs more speed
        assert uphill > downhill

    def test_optimal_aim_point(self, putter: PutterStroke) -> None:
        """Should compute optimal aim point accounting for break."""
        target = np.array([10.0, 10.0])
        ball_pos = np.array([5.0, 10.0])
        break_amount = 0.5  # meters of break

        aim_point = putter.compute_aim_point(ball_pos, target, break_amount)

        # Aim point should be different from target
        assert not np.allclose(aim_point, target)


class TestStrokeParametersPresets:
    """Tests for stroke parameter presets."""

    def test_lag_putt_preset(self) -> None:
        """Lag putt should be gentle."""
        params = StrokeParameters.create_preset(
            "lag_putt", direction=np.array([1.0, 0.0])
        )
        assert params.speed < 2.0  # Gentle speed

    def test_aggressive_putt_preset(self) -> None:
        """Aggressive putt should be firm."""
        params = StrokeParameters.create_preset(
            "aggressive", direction=np.array([1.0, 0.0])
        )
        assert params.speed > 2.5  # Firm speed

    def test_practice_stroke_preset(self) -> None:
        """Practice stroke with controlled speed."""
        params = StrokeParameters.create_preset(
            "practice", direction=np.array([1.0, 0.0])
        )
        assert params.speed > 0
        assert params.face_angle == 0.0  # Square face
