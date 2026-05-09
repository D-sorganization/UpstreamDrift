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


# =============================================================================
# Engine Integration Tests (Issue #758)
# =============================================================================
