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


# =============================================================================
# Engine Integration Tests (Issue #758)
# =============================================================================


class TestMOIEffectiveMass:
    """Tests for MOI-based effective mass at impact point (Issue #1082)."""

    @pytest.fixture
    def center_hit_state(self) -> PreImpactState:
        """Pre-impact state with center hit (no offset)."""
        return PreImpactState(
            clubhead_velocity=np.array([45.0, 0.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            ball_position=np.zeros(3),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
            clubhead_mass=0.200,
            clubhead_moi=4.5e-4,
            impact_offset=None,
        )

    @pytest.fixture
    def toe_hit_state(self) -> PreImpactState:
        """Pre-impact state with toe hit (20mm offset)."""
        return PreImpactState(
            clubhead_velocity=np.array([45.0, 0.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            ball_position=np.zeros(3),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
            clubhead_mass=0.200,
            clubhead_moi=4.5e-4,
            impact_offset=np.array([0.020, 0.0]),  # 20mm toe
        )

    def test_center_hit_equals_point_mass(
        self, center_hit_state: PreImpactState
    ) -> None:
        """Center hit should produce same result as point mass model."""
        model = RigidBodyImpactModel()
        params = ImpactParameters()

        # With impact_offset=None, should use full clubhead mass
        result = model.solve(center_hit_state, params)

        # Compare with explicit zero offset
        zero_offset_state = PreImpactState(
            clubhead_velocity=np.array([45.0, 0.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            ball_position=np.zeros(3),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
            clubhead_mass=0.200,
            clubhead_moi=4.5e-4,
            impact_offset=np.array([0.0, 0.0]),
        )
        result_zero = model.solve(zero_offset_state, params)

        np.testing.assert_allclose(
            result.ball_velocity, result_zero.ball_velocity, atol=1e-10
        )

    def test_off_center_reduces_ball_speed(
        self,
        center_hit_state: PreImpactState,
        toe_hit_state: PreImpactState,
    ) -> None:
        """Off-center hit should produce lower ball speed than center hit."""
        model = RigidBodyImpactModel()
        params = ImpactParameters()

        result_center = model.solve(center_hit_state, params)
        result_toe = model.solve(toe_hit_state, params)

        speed_center = np.linalg.norm(result_center.ball_velocity)
        speed_toe = np.linalg.norm(result_toe.ball_velocity)

        assert speed_toe < speed_center

    def test_larger_offset_lower_speed(self) -> None:
        """Larger offset from CG should produce progressively lower ball speed."""
        model = RigidBodyImpactModel()
        params = ImpactParameters()

        speeds = []
        for offset_mm in [0, 10, 20, 30, 40]:
            state = PreImpactState(
                clubhead_velocity=np.array([45.0, 0.0, 0.0]),
                clubhead_angular_velocity=np.zeros(3),
                clubhead_orientation=np.array([1.0, 0.0, 0.0]),
                ball_position=np.zeros(3),
                ball_velocity=np.zeros(3),
                ball_angular_velocity=np.zeros(3),
                clubhead_mass=0.200,
                clubhead_moi=4.5e-4,
                impact_offset=np.array([offset_mm / 1000.0, 0.0]),
            )
            result = model.solve(state, params)
            speeds.append(float(np.linalg.norm(result.ball_velocity)))

        # Speeds should be monotonically decreasing
        for i in range(len(speeds) - 1):
            assert speeds[i] >= speeds[i + 1], (
                f"Speed at {i * 10}mm ({speeds[i]:.1f}) should be >= "
                f"speed at {(i + 1) * 10}mm ({speeds[i + 1]:.1f})"
            )

    def test_higher_moi_more_forgiving(self) -> None:
        """Higher MOI should result in less ball speed loss on off-center hits."""
        model = RigidBodyImpactModel()
        params = ImpactParameters()
        offset = np.array([0.025, 0.0])  # 25mm toe hit

        # Low MOI clubhead
        low_moi = PreImpactState(
            clubhead_velocity=np.array([45.0, 0.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            ball_position=np.zeros(3),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
            clubhead_mass=0.200,
            clubhead_moi=3.0e-4,  # Lower MOI (less forgiving)
            impact_offset=offset,
        )

        # High MOI clubhead
        high_moi = PreImpactState(
            clubhead_velocity=np.array([45.0, 0.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            ball_position=np.zeros(3),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
            clubhead_mass=0.200,
            clubhead_moi=6.0e-4,  # Higher MOI (more forgiving)
            impact_offset=offset,
        )

        result_low = model.solve(low_moi, params)
        result_high = model.solve(high_moi, params)

        speed_low = np.linalg.norm(result_low.ball_velocity)
        speed_high = np.linalg.norm(result_high.ball_velocity)

        # Higher MOI = more forgiving = higher ball speed on off-center hits
        assert speed_high > speed_low

    def test_effective_mass_formula(self) -> None:
        """Verify the effective mass calculation: m_eff = 1 / (1/m + r²/I)."""
        m_club = 0.200
        I_club = 4.5e-4
        r = 0.025  # 25mm

        expected_m_eff = 1.0 / (1.0 / m_club + r**2 / I_club)

        # Should be less than actual mass
        assert expected_m_eff < m_club

        # For typical driver at 25mm offset:
        # m_eff = 1 / (1/0.2 + 0.025²/4.5e-4)
        #       = 1 / (5 + 1.389) = 1 / 6.389 ≈ 0.1565
        assert expected_m_eff == pytest.approx(0.1565, rel=0.01)

    def test_backward_compatibility_no_offset(self) -> None:
        """Without impact_offset, behavior should match original point mass model."""
        model = RigidBodyImpactModel()
        params = ImpactParameters()

        # State without MOI fields (uses defaults)
        state = PreImpactState(
            clubhead_velocity=np.array([45.0, 0.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            ball_position=np.zeros(3),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
            clubhead_mass=0.200,
        )

        result = model.solve(state, params)

        # Should produce standard point-mass result
        # m_eff = (0.0459 * 0.200) / (0.0459 + 0.200) ≈ 0.03732
        # j = (1 + 0.83) * 0.03732 * 45.0 ≈ 3.076
        # v_ball = 3.076 / 0.0459 ≈ 67.0 m/s
        speed = np.linalg.norm(result.ball_velocity)
        assert speed == pytest.approx(67.0, rel=0.05)
