"""Tests for Modular Impact Model.

Guideline K3 implementation tests.
"""

from __future__ import annotations

import numpy as np
from src.shared.python.physics.impact_model import (
    ImpactEvent,
    ImpactModelType,
    ImpactParameters,
    PreImpactState,
    RigidBodyImpactModel,
    validate_energy_balance,
)

# =============================================================================
# Engine Integration Tests (Issue #758)
# =============================================================================


class TestImpactEventDataclass:
    """Tests for ImpactEvent dataclass."""

    def test_event_contains_all_data(self) -> None:
        """ImpactEvent should contain complete impact data."""
        pre_state = PreImpactState(
            clubhead_velocity=np.array([40.0, 0.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            ball_position=np.zeros(3),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
        )

        model = RigidBodyImpactModel()
        params = ImpactParameters()
        post_state = model.solve(pre_state, params)
        energy = validate_energy_balance(pre_state, post_state, params)

        event = ImpactEvent(
            timestamp=0.5,
            pre_state=pre_state,
            post_state=post_state,
            energy_balance=energy,
            impact_id=0,
            model_type=ImpactModelType.RIGID_BODY,
        )

        assert event.timestamp == 0.5
        assert event.impact_id == 0
        assert event.model_type == ImpactModelType.RIGID_BODY
        assert "total_ke_pre" in event.energy_balance
