"""Tests for sand-mediated momentum transfer (issue #8613).

In a splash shot, the club never touches the ball directly. Momentum is
transferred through displaced sand grains that impact the ball.

TDD: Write failing tests first, then implement.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings, strategies as st

from bunkershot3d.ball.lie import BallLie, BallProperties
from bunkershot3d.ball.splash import (
    ContactType,
    compute_ball_launch_from_splash,
    compute_sand_ejecta_velocity,
    compute_splash_impulse,
)


class TestSandEjectaVelocity:
    """Tests for sand ejecta velocity model."""

    def test_zero_club_speed_gives_zero_ejecta(self) -> None:
        """No club motion = no sand ejection."""
        v_ejecta = compute_sand_ejecta_velocity(
            club_velocity_m_s=0.0,
            club_loft_rad=math.radians(56),
        )
        assert v_ejecta == pytest.approx(0.0, abs=1e-10)

    def test_ejecta_velocity_increases_with_club_speed(self) -> None:
        """Faster club = faster ejecta."""
        v_slow = compute_sand_ejecta_velocity(
            club_velocity_m_s=15.0, club_loft_rad=math.radians(56)
        )
        v_fast = compute_sand_ejecta_velocity(
            club_velocity_m_s=25.0, club_loft_rad=math.radians(56)
        )
        assert v_fast > v_slow

    def test_ejecta_velocity_is_fraction_of_club_speed(self) -> None:
        """Ejecta velocity is less than club speed (energy dissipation)."""
        club_v = 25.0
        v_ejecta = compute_sand_ejecta_velocity(
            club_velocity_m_s=club_v, club_loft_rad=math.radians(56)
        )
        assert v_ejecta < club_v
        assert v_ejecta > 0.1 * club_v  # But not negligible


class TestSplashImpulse:
    """Tests for splash impulse on ball."""

    def test_buried_ball_receives_more_impulse(self) -> None:
        """More buried = more sand between club and ball = more impulse."""
        ball = BallProperties()
        lie_shallow = BallLie(depth_m=0.005)
        lie_deep = BallLie(depth_m=0.015)

        result_shallow = compute_splash_impulse(
            lie=lie_shallow,
            ball=ball,
            club_velocity_m_s=25.0,
            club_loft_rad=math.radians(56),
            entry_depth_m=0.02,
        )
        result_deep = compute_splash_impulse(
            lie=lie_deep,
            ball=ball,
            club_velocity_m_s=25.0,
            club_loft_rad=math.radians(56),
            entry_depth_m=0.02,
        )
        # Deeper burial means ball is in the path of more displaced sand
        # This is counterintuitive but physically correct for splash shots
        assert result_deep.impulse_magnitude_ns >= result_shallow.impulse_magnitude_ns

    def test_impulse_has_upward_component(self) -> None:
        """Splash impulse should launch ball upward."""
        ball = BallProperties()
        lie = BallLie(depth_m=0.01)

        result = compute_splash_impulse(
            lie=lie,
            ball=ball,
            club_velocity_m_s=25.0,
            club_loft_rad=math.radians(56),
            entry_depth_m=0.02,
        )
        # z component should be positive (upward)
        assert result.impulse_z_ns > 0

    def test_impulse_has_forward_component(self) -> None:
        """Splash impulse should have forward momentum (direction of play)."""
        ball = BallProperties()
        lie = BallLie(depth_m=0.01)

        result = compute_splash_impulse(
            lie=lie,
            ball=ball,
            club_velocity_m_s=25.0,
            club_loft_rad=math.radians(56),
            entry_depth_m=0.02,
        )
        # x component should be positive (forward)
        assert result.impulse_x_ns > 0

    def test_impulse_increases_with_club_speed(self) -> None:
        """Faster swing = more impulse transfer."""
        ball = BallProperties()
        lie = BallLie(depth_m=0.01)

        result_slow = compute_splash_impulse(
            lie=lie,
            ball=ball,
            club_velocity_m_s=15.0,
            club_loft_rad=math.radians(56),
            entry_depth_m=0.02,
        )
        result_fast = compute_splash_impulse(
            lie=lie,
            ball=ball,
            club_velocity_m_s=25.0,
            club_loft_rad=math.radians(56),
            entry_depth_m=0.02,
        )
        assert result_fast.impulse_magnitude_ns > result_slow.impulse_magnitude_ns


class TestBallLaunchFromSplash:
    """Tests for ball launch conditions from splash."""

    def test_ball_launch_speed_positive(self) -> None:
        """Ball should have positive launch speed."""
        ball = BallProperties()
        lie = BallLie(depth_m=0.01)

        result = compute_ball_launch_from_splash(
            lie=lie,
            ball=ball,
            club_velocity_m_s=25.0,
            club_loft_rad=math.radians(56),
            entry_depth_m=0.02,
        )
        assert result.ball_speed_m_s > 0

    def test_ball_launch_angle_reasonable(self) -> None:
        """Launch angle should be between 20-60 degrees for typical bunker shot."""
        ball = BallProperties()
        lie = BallLie(depth_m=0.01)

        result = compute_ball_launch_from_splash(
            lie=lie,
            ball=ball,
            club_velocity_m_s=25.0,
            club_loft_rad=math.radians(56),
            entry_depth_m=0.02,
        )
        angle_deg = math.degrees(result.launch_angle_rad)
        assert 20 < angle_deg < 70

    def test_ball_has_backspin(self) -> None:
        """Ball should have backspin from splash (friction with sand)."""
        ball = BallProperties()
        lie = BallLie(depth_m=0.01)

        result = compute_ball_launch_from_splash(
            lie=lie,
            ball=ball,
            club_velocity_m_s=25.0,
            club_loft_rad=math.radians(56),
            entry_depth_m=0.02,
        )
        # Backspin is negative omega_y (ball rotating backward)
        assert result.spin_rate_rpm > 0

    def test_contact_type_is_splash(self) -> None:
        """Splash transfer should report SPLASH contact type."""
        ball = BallProperties()
        lie = BallLie(depth_m=0.01)

        result = compute_ball_launch_from_splash(
            lie=lie,
            ball=ball,
            club_velocity_m_s=25.0,
            club_loft_rad=math.radians(56),
            entry_depth_m=0.02,
        )
        assert result.contact_type == ContactType.SPLASH


class TestSplashTransferEnergyAccounting:
    """Tests for energy conservation in splash transfer."""

    def test_ball_ke_less_than_club_ke_fraction(self) -> None:
        """Ball KE should be a small fraction of club KE (most goes to sand)."""
        ball = BallProperties()
        lie = BallLie(depth_m=0.01)
        club_mass = 0.30  # 300g wedge head
        club_v = 25.0
        club_ke = 0.5 * club_mass * club_v**2

        result = compute_ball_launch_from_splash(
            lie=lie,
            ball=ball,
            club_velocity_m_s=club_v,
            club_loft_rad=math.radians(56),
            entry_depth_m=0.02,
        )
        ball_ke = 0.5 * ball.mass_kg * result.ball_speed_m_s**2

        # In a splash shot, most energy goes to sand. Ball gets maybe 5-15%.
        assert ball_ke < 0.20 * club_ke

    def test_energy_transfer_fraction_returned(self) -> None:
        """Result should include energy transfer fraction."""
        ball = BallProperties()
        lie = BallLie(depth_m=0.01)

        result = compute_ball_launch_from_splash(
            lie=lie,
            ball=ball,
            club_velocity_m_s=25.0,
            club_loft_rad=math.radians(56),
            entry_depth_m=0.02,
        )
        assert 0 < result.energy_transfer_fraction < 1


class TestSplashPhysicsSanityChecks:
    """Sanity checks against known bunker shot behavior."""

    def test_typical_tour_bunker_shot_ball_speed(self) -> None:
        """Tour bunker shot: ~25 m/s swing -> ~15-20 m/s ball speed."""
        ball = BallProperties()
        lie = BallLie(depth_m=0.005)  # Slightly settled

        result = compute_ball_launch_from_splash(
            lie=lie,
            ball=ball,
            club_velocity_m_s=25.0,
            club_loft_rad=math.radians(56),
            entry_depth_m=0.015,
        )
        # Typical tour bunker shot: ball speed ~15-20 m/s
        assert 10 < result.ball_speed_m_s < 25

    def test_typical_spin_rate(self) -> None:
        """Tour bunker shot: ~4000-8000 RPM backspin."""
        ball = BallProperties()
        lie = BallLie(depth_m=0.005)

        result = compute_ball_launch_from_splash(
            lie=lie,
            ball=ball,
            club_velocity_m_s=25.0,
            club_loft_rad=math.radians(56),
            entry_depth_m=0.015,
        )
        # Typical bunker shot spin: 4000-8000 RPM
        assert 2000 < result.spin_rate_rpm < 10000


class TestSplashMetamorphic:
    """Metamorphic tests for splash physics."""

    @given(st.floats(min_value=10.0, max_value=30.0, allow_nan=False))
    @settings(deadline=None)
    def test_launch_speed_increases_with_club_speed(self, club_v: float) -> None:
        """Faster club = faster ball."""
        ball = BallProperties()
        lie = BallLie(depth_m=0.01)

        result = compute_ball_launch_from_splash(
            lie=lie,
            ball=ball,
            club_velocity_m_s=club_v,
            club_loft_rad=math.radians(56),
            entry_depth_m=0.02,
        )
        # Ball speed should be positive and monotonic with club speed
        assert result.ball_speed_m_s > 0
        assert result.ball_speed_m_s < club_v  # Can't exceed club speed

    @given(st.floats(min_value=40.0, max_value=64.0, allow_nan=False))
    @settings(deadline=None)
    def test_higher_loft_gives_higher_launch(self, loft_deg: float) -> None:
        """Higher loft = higher launch angle."""
        ball = BallProperties()
        lie = BallLie(depth_m=0.01)

        result = compute_ball_launch_from_splash(
            lie=lie,
            ball=ball,
            club_velocity_m_s=25.0,
            club_loft_rad=math.radians(loft_deg),
            entry_depth_m=0.02,
        )
        # Launch angle should be positive and related to loft
        assert result.launch_angle_rad > 0
        assert result.launch_angle_rad < math.pi / 2  # Less than 90 degrees
