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


# =============================================================================
# Engine Integration Tests (Issue #758)
# =============================================================================
