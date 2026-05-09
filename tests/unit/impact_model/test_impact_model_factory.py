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


class TestImpactModelFactory:
    """Tests for impact model factory."""

    @pytest.mark.parametrize(
        "model_type, expected_class",
        [
            (ImpactModelType.RIGID_BODY, RigidBodyImpactModel),
            (ImpactModelType.SPRING_DAMPER, SpringDamperImpactModel),
            (ImpactModelType.FINITE_TIME, FiniteTimeImpactModel),
        ],
        ids=["rigid-body", "spring-damper", "finite-time"],
    )
    def test_creates_correct_model(
        self, model_type: ImpactModelType, expected_class: type
    ) -> None:
        """Factory should create the correct model type."""
        model = create_impact_model(model_type)
        assert isinstance(model, expected_class)


# =============================================================================
# Engine Integration Tests (Issue #758)
# =============================================================================
