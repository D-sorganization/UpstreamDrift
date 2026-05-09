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


class TestPreImpactState:
    """Tests for pre-impact state creation."""

    def test_impact_model_default_values(self) -> None:
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


# =============================================================================
# Engine Integration Tests (Issue #758)
# =============================================================================
