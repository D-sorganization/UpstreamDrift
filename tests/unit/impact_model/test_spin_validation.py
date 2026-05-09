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


class TestSpinValidation:
    """Tests for spin validation (Issue #758)."""

    def test_realistic_spin_rates(self) -> None:
        """Spin rates should be in realistic range for golf."""
        solver = ImpactSolverAPI()

        # Typical driver impact
        solver.solve_impact(
            timestamp=0.0,
            clubhead_velocity=np.array([45.0, 0.0, 0.0]),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
        )

        result = solver.validate_spin_behavior(max_spin_rpm=10000)

        assert result["valid"]
        # Driver backspin typically 2000-3000 RPM
        assert result["max_observed_rpm"] < 10000
