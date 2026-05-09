"""Tests for Contact-Based Grip Model.

Guideline K2 implementation tests.
"""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.physics.grip_contact_model import (
    ContactPoint,
    ContactState,
    GripContactExporter,
    GripContactModel,
    GripParameters,
    PressureVisualizationData,
    check_friction_cone,
    classify_contact_state,
    compute_center_of_pressure,
    compute_pressure_visualization,
    create_mujoco_grip_contacts,
    decompose_contact_force,
)


class TestContactClassification:
    """Tests for contact state classification."""

    @pytest.fixture
    def default_params(self) -> GripParameters:
        """Create default grip parameters."""
        return GripParameters()

    def test_no_contact_for_zero_normal(self, default_params: GripParameters) -> None:
        """Zero or negative normal force should be no contact."""
        state = classify_contact_state(
            normal_force=0.0,
            tangent_force=np.zeros(3),
            slip_velocity=np.zeros(3),
            params=default_params,
        )
        assert state == ContactState.NO_CONTACT

    def test_sticking_within_cone(self, default_params: GripParameters) -> None:
        """Contact within friction cone with no slip velocity should stick."""
        state = classify_contact_state(
            normal_force=100.0,
            tangent_force=np.array([10.0, 0.0, 0.0]),  # Well within cone
            slip_velocity=np.zeros(3),
            params=default_params,
        )
        assert state == ContactState.STICKING

    def test_slipping_with_velocity(self, default_params: GripParameters) -> None:
        """Contact with significant slip velocity should be slipping."""
        state = classify_contact_state(
            normal_force=100.0,
            tangent_force=np.array([10.0, 0.0, 0.0]),
            slip_velocity=np.array([0.1, 0.0, 0.0]),  # 0.1 m/s > threshold
            params=default_params,
        )
        assert state == ContactState.SLIPPING
