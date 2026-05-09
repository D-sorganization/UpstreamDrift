"""Tests for Modular Impact Model.

Guideline K3 implementation tests.
"""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.core.physics_constants import (
    GOLF_BALL_MASS_KG,
    GOLF_BALL_RADIUS_M,
)
from src.shared.python.physics.impact_model import (
    FiniteTimeImpactModel,
    ImpactEvent,
    ImpactModelType,
    ImpactParameters,
    ImpactRecorder,
    ImpactSolverAPI,
    PreImpactState,
    RigidBodyImpactModel,
    SpringDamperImpactModel,
    compute_gear_effect_spin,
    create_impact_model,
    validate_energy_balance,
)


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

        m_ball = GOLF_BALL_MASS_KG
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


# =============================================================================
# Engine Integration Tests (Issue #758)
# =============================================================================
