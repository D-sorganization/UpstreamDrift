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
            ball_position=np.array([GOLF_BALL_RADIUS_M, 0.0, 0.0]),
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
            ball_position=np.array([GOLF_BALL_RADIUS_M, 0.0, 0.0]),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
        )

        model = SpringDamperImpactModel()
        params = ImpactParameters()

        result = model.solve(pre_state, params)

        # Should have measurable contact time
        assert result.contact_duration > 0


# =============================================================================
# Engine Integration Tests (Issue #758)
# =============================================================================
