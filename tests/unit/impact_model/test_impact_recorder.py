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


class TestImpactRecorder:
    """Tests for impact event recording (Issue #758)."""

    @pytest.fixture
    def pre_state(self) -> PreImpactState:
        """Create sample pre-impact state."""
        return PreImpactState(
            clubhead_velocity=np.array([40.0, 0.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            ball_position=np.zeros(3),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
        )

    def test_record_impact(self, pre_state: PreImpactState) -> None:
        """Should record impact event."""
        recorder = ImpactRecorder()
        model = RigidBodyImpactModel()
        params = ImpactParameters()

        post_state = model.solve(pre_state, params)
        event = recorder.record_impact(0.5, pre_state, post_state, params)

        assert event.impact_id == 0
        assert event.timestamp == 0.5
        assert len(recorder.events) == 1

    def test_increments_impact_id(self, pre_state: PreImpactState) -> None:
        """Should increment impact ID for each event."""
        recorder = ImpactRecorder()
        model = RigidBodyImpactModel()
        params = ImpactParameters()
        post_state = model.solve(pre_state, params)

        event1 = recorder.record_impact(0.1, pre_state, post_state, params)
        event2 = recorder.record_impact(0.2, pre_state, post_state, params)

        assert event1.impact_id == 0
        assert event2.impact_id == 1

    def test_impact_model_export_to_dict(self, pre_state: PreImpactState) -> None:
        """Should export events as dictionary."""
        recorder = ImpactRecorder()
        model = RigidBodyImpactModel()
        params = ImpactParameters()
        post_state = model.solve(pre_state, params)

        recorder.record_impact(0.1, pre_state, post_state, params)

        data = recorder.export_to_dict()

        assert "num_impacts" in data
        assert "events" in data
        assert "summary" in data
        assert data["num_impacts"] == 1

    def test_impact_model_get_summary(self, pre_state: PreImpactState) -> None:
        """Should compute summary statistics."""
        recorder = ImpactRecorder()
        model = RigidBodyImpactModel()
        params = ImpactParameters()
        post_state = model.solve(pre_state, params)

        recorder.record_impact(0.1, pre_state, post_state, params)
        recorder.record_impact(0.2, pre_state, post_state, params)

        summary = recorder.get_summary()

        assert summary["num_impacts"] == 2
        assert "mean_ball_speed" in summary
        assert "max_ball_speed" in summary

    def test_reset_clears_events(self, pre_state: PreImpactState) -> None:
        """Reset should clear all events."""
        recorder = ImpactRecorder()
        model = RigidBodyImpactModel()
        params = ImpactParameters()
        post_state = model.solve(pre_state, params)

        recorder.record_impact(0.1, pre_state, post_state, params)
        assert len(recorder.events) == 1

        recorder.reset()
        assert len(recorder.events) == 0
