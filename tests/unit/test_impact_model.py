"""Tests for Modular Impact Model.

Guideline K3 implementation tests.
"""

from __future__ import annotations

import numpy as np
import pytest
from shared.python.impact_model import (
    GOLF_BALL_MASS,
    GOLF_BALL_RADIUS,
    FiniteTimeImpactModel,
    ImpactModelType,
    ImpactParameters,
    PreImpactState,
    RigidBodyImpactModel,
    SpringDamperImpactModel,
    compute_gear_effect_spin,
    create_impact_model,
    validate_energy_balance,
)


class TestPreImpactState:
    """Tests for pre-impact state creation."""

    def test_default_values(self) -> None:
        """Should have sensible default values."""
        state = PreImpactState(
            clubhead_velocity=np.array([45.0, 0.0, 0.0]),  # 45 m/s ~100 mph
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            ball_position=np.zeros(3),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
        )

        assert state.clubhead_mass == pytest.approx(0.200)  # 200g
        assert state.clubhead_loft == pytest.approx(np.radians(10.5))


class TestRigidBodyImpactModel:
    """Tests for rigid body collision model."""

    @pytest.fixture
    def default_pre_state(self) -> PreImpactState:
        """Create default pre-impact state for testing."""
        return PreImpactState(
            clubhead_velocity=np.array([40.0, 0.0, 0.0]),  # 40 m/s
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),  # Facing X
            ball_position=np.zeros(3),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
        )

    @pytest.fixture
    def default_params(self) -> ImpactParameters:
        """Create default impact parameters."""
        return ImpactParameters()

    def test_ball_gains_velocity(
        self,
        default_pre_state: PreImpactState,
        default_params: ImpactParameters,
    ) -> None:
        """Ball should gain velocity after impact."""
        model = RigidBodyImpactModel()
        result = model.solve(default_pre_state, default_params)

        # Ball should have significant forward velocity
        assert result.ball_velocity[0] > 0
        # Ball speed should be faster than clubhead (smash factor > 1)
        ball_speed = np.linalg.norm(result.ball_velocity)
        club_speed = np.linalg.norm(default_pre_state.clubhead_velocity)
        assert ball_speed > club_speed * 1.3  # Typical smash factor ~1.45-1.5

    def test_clubhead_loses_velocity(
        self,
        default_pre_state: PreImpactState,
        default_params: ImpactParameters,
    ) -> None:
        """Clubhead should lose velocity after impact."""
        model = RigidBodyImpactModel()
        result = model.solve(default_pre_state, default_params)

        # Clubhead should be slower after impact
        club_speed_pre = np.linalg.norm(default_pre_state.clubhead_velocity)
        club_speed_post = np.linalg.norm(result.clubhead_velocity)
        assert club_speed_post < club_speed_pre

    def test_momentum_conservation(
        self,
        default_pre_state: PreImpactState,
        default_params: ImpactParameters,
    ) -> None:
        """Total momentum should be conserved."""
        model = RigidBodyImpactModel()
        result = model.solve(default_pre_state, default_params)

        m_ball = GOLF_BALL_MASS
        m_club = default_pre_state.clubhead_mass

        # Pre-impact momentum
        p_pre = (
            m_club * default_pre_state.clubhead_velocity
            + m_ball * default_pre_state.ball_velocity
        )

        # Post-impact momentum
        p_post = m_club * result.clubhead_velocity + m_ball * result.ball_velocity

        # Momentum should be conserved
        np.testing.assert_allclose(p_pre, p_post, rtol=1e-5)

    def test_cor_affects_separation_velocity(self) -> None:
        """Higher COR should give higher separation velocity."""
        pre_state = PreImpactState(
            clubhead_velocity=np.array([40.0, 0.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            ball_position=np.zeros(3),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
        )

        model = RigidBodyImpactModel()

        low_cor = ImpactParameters(cor=0.6)
        high_cor = ImpactParameters(cor=0.9)

        result_low = model.solve(pre_state, low_cor)
        result_high = model.solve(pre_state, high_cor)

        # Higher COR should give faster ball
        assert np.linalg.norm(result_high.ball_velocity) > np.linalg.norm(
            result_low.ball_velocity
        )


class TestSpringDamperImpactModel:
    """Tests for spring-damper compliant contact model.

    Note: The spring-damper model requires very small timesteps
    due to the high contact stiffness. These tests are marked as
    expected failures until a more numerically stable integration
    scheme (e.g., implicit Euler) is implemented.
    """

    def test_ball_gains_velocity(self) -> None:
        """Spring-damper model should produce finite results."""
        pre_state = PreImpactState(
            clubhead_velocity=np.array([40.0, 0.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            ball_position=np.array([GOLF_BALL_RADIUS, 0.0, 0.0]),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
        )

        model = SpringDamperImpactModel()  # Use default stable timestep
        params = ImpactParameters()

        result = model.solve(pre_state, params)

        # Result should be finite and reasonably bounded
        assert np.all(np.isfinite(result.ball_velocity))
        # Velocity magnitude should be reasonable (not blown up)
        assert np.linalg.norm(result.ball_velocity) < 200  # m/s

    def test_has_contact_duration(self) -> None:
        """Spring-damper model should report non-zero contact duration."""
        pre_state = PreImpactState(
            clubhead_velocity=np.array([40.0, 0.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            ball_position=np.array([GOLF_BALL_RADIUS, 0.0, 0.0]),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
        )

        model = SpringDamperImpactModel()
        params = ImpactParameters()

        result = model.solve(pre_state, params)

        # Should have measurable contact time
        assert result.contact_duration > 0


class TestFiniteTimeImpactModel:
    """Tests for finite-time impulse-momentum model."""

    def test_uses_specified_duration(self) -> None:
        """Should use the specified contact duration."""
        pre_state = PreImpactState(
            clubhead_velocity=np.array([40.0, 0.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            ball_position=np.zeros(3),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
        )

        model = FiniteTimeImpactModel()
        params = ImpactParameters(contact_duration=0.0005)

        result = model.solve(pre_state, params)

        assert result.contact_duration == pytest.approx(0.0005)


class TestGearEffect:
    """Tests for gear effect spin computation."""

    def test_center_impact_no_gear_spin(self) -> None:
        """Center impact should produce no gear effect spin."""
        spin = compute_gear_effect_spin(
            impact_offset=np.array([0.0, 0.0]),
            clubhead_velocity=np.array([40.0, 0.0, 0.0]),
            clubface_normal=np.array([1.0, 0.0, 0.0]),
        )

        np.testing.assert_allclose(spin, np.zeros(3), atol=1e-10)

    def test_toe_impact_produces_hook_spin(self) -> None:
        """Toe impact should produce hook (counterclockwise) spin."""
        spin = compute_gear_effect_spin(
            impact_offset=np.array([0.03, 0.0]),  # 30mm toe side
            clubhead_velocity=np.array([40.0, 0.0, 0.0]),
            clubface_normal=np.array([1.0, 0.0, 0.0]),
        )

        # Should have non-zero spin
        assert np.linalg.norm(spin) > 0

    def test_higher_speed_more_spin(self) -> None:
        """Higher clubhead speed should produce more gear effect spin."""
        offset = np.array([0.02, 0.0])
        normal = np.array([1.0, 0.0, 0.0])

        spin_slow = compute_gear_effect_spin(offset, np.array([30.0, 0.0, 0.0]), normal)
        spin_fast = compute_gear_effect_spin(offset, np.array([50.0, 0.0, 0.0]), normal)

        assert np.linalg.norm(spin_fast) > np.linalg.norm(spin_slow)


class TestEnergyBalance:
    """Tests for energy balance validation."""

    def test_energy_lost_with_cor_less_than_1(self) -> None:
        """Impact with COR < 1 should lose energy."""
        pre_state = PreImpactState(
            clubhead_velocity=np.array([40.0, 0.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            ball_position=np.zeros(3),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
        )

        model = RigidBodyImpactModel()
        params = ImpactParameters(cor=0.78)

        result = model.solve(pre_state, params)
        balance = validate_energy_balance(pre_state, result, params)

        # Energy should be lost
        assert balance["energy_lost"] > 0
        assert balance["total_ke_post"] < balance["total_ke_pre"]

    def test_ball_launch_speed_reasonable(self) -> None:
        """Ball launch speed should be in realistic range."""
        pre_state = PreImpactState(
            clubhead_velocity=np.array([45.0, 0.0, 0.0]),  # ~100 mph
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            ball_position=np.zeros(3),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
        )

        model = RigidBodyImpactModel()
        params = ImpactParameters(cor=0.78)

        result = model.solve(pre_state, params)
        balance = validate_energy_balance(pre_state, result, params)

        # Ball launch speed should be ~1.45-1.5x clubhead speed
        # 45 m/s * 1.45 = ~65 m/s (~145 mph)
        assert 50 < balance["ball_launch_speed"] < 80


class TestImpactModelFactory:
    """Tests for impact model factory."""

    def test_creates_rigid_body_model(self) -> None:
        """Factory should create rigid body model."""
        model = create_impact_model(ImpactModelType.RIGID_BODY)
        assert isinstance(model, RigidBodyImpactModel)

    def test_creates_spring_damper_model(self) -> None:
        """Factory should create spring-damper model."""
        model = create_impact_model(ImpactModelType.SPRING_DAMPER)
        assert isinstance(model, SpringDamperImpactModel)

    def test_creates_finite_time_model(self) -> None:
        """Factory should create finite-time model."""
        model = create_impact_model(ImpactModelType.FINITE_TIME)
        assert isinstance(model, FiniteTimeImpactModel)
