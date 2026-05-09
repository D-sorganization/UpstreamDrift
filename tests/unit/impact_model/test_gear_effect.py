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


# =============================================================================
# Engine Integration Tests (Issue #758)
# =============================================================================
