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


class TestImpactSolverAPI:
    """Tests for engine-agnostic impact solver API (Issue #758)."""

    def test_solve_impact_basic(self) -> None:
        """Should solve basic impact."""
        solver = ImpactSolverAPI()

        post = solver.solve_impact(
            timestamp=0.0,
            clubhead_velocity=np.array([40.0, 0.0, 0.0]),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
        )

        assert post.ball_velocity[0] > 0
        assert len(solver.recorder.events) == 1

    def test_solve_impact_no_record(self) -> None:
        """Should not record when record=False."""
        solver = ImpactSolverAPI()

        solver.solve_impact(
            timestamp=0.0,
            clubhead_velocity=np.array([40.0, 0.0, 0.0]),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            record=False,
        )

        assert len(solver.recorder.events) == 0

    def test_solve_with_gear_effect(self) -> None:
        """Should add gear effect spin for offset impact."""
        solver = ImpactSolverAPI()

        post = solver.solve_with_gear_effect(
            timestamp=0.0,
            clubhead_velocity=np.array([40.0, 0.0, 0.0]),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            impact_offset=np.array([0.02, 0.0]),  # Toe hit
        )

        # Should have non-zero spin from gear effect
        assert np.linalg.norm(post.ball_angular_velocity) > 0
        # Should record impact location
        np.testing.assert_allclose(post.impact_location, [0.02, 0.0])

    def test_get_energy_report(self) -> None:
        """Should generate energy balance report."""
        solver = ImpactSolverAPI()

        solver.solve_impact(
            timestamp=0.0,
            clubhead_velocity=np.array([40.0, 0.0, 0.0]),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
        )

        report = solver.get_energy_report()

        assert "impacts" in report
        assert "total_ke_pre" in report
        assert "total_energy_lost" in report
        assert len(report["impacts"]) == 1

    def test_validate_cor_behavior(self) -> None:
        """Should validate COR within tolerance."""
        solver = ImpactSolverAPI(params=ImpactParameters(cor=0.78))

        # Run several impacts
        for i in range(5):
            solver.solve_impact(
                timestamp=i * 0.1,
                clubhead_velocity=np.array([40.0 + i, 0.0, 0.0]),
                clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            )

        result = solver.validate_cor_behavior(tolerance=0.1)

        assert "valid" in result
        assert "measured_cor_mean" in result
        assert "deviation" in result

    def test_validate_spin_behavior(self) -> None:
        """Should validate spin within physical limits."""
        solver = ImpactSolverAPI()

        solver.solve_impact(
            timestamp=0.0,
            clubhead_velocity=np.array([40.0, 0.0, 0.0]),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
        )

        result = solver.validate_spin_behavior(max_spin_rpm=10000)

        assert "valid" in result
        assert "max_observed_rpm" in result

    def test_different_model_types(self) -> None:
        """Should work with different impact model types."""
        for model_type in [
            ImpactModelType.RIGID_BODY,
            ImpactModelType.FINITE_TIME,
        ]:
            solver = ImpactSolverAPI(model_type=model_type)

            post = solver.solve_impact(
                timestamp=0.0,
                clubhead_velocity=np.array([40.0, 0.0, 0.0]),
                clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            )

            assert post.ball_velocity[0] > 0

    def test_impact_model_reset_clears_state(self) -> None:
        """Reset should clear recorder."""
        solver = ImpactSolverAPI()

        solver.solve_impact(
            timestamp=0.0,
            clubhead_velocity=np.array([40.0, 0.0, 0.0]),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
        )

        assert len(solver.recorder.events) == 1

        solver.reset()

        assert len(solver.recorder.events) == 0
