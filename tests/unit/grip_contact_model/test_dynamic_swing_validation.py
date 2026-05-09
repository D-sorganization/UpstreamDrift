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


class TestDynamicSwingValidation:
    """Tests for dynamic swing validation (Issue #757)."""

    def test_slip_detection_during_high_tangent(self) -> None:
        """Should detect slip when tangent force exceeds friction limit."""
        model = GripContactModel()

        # Create contact with high tangential force (simulating swing acceleration)
        positions = np.array([[0.0, 0.0, 0.0]])
        normals = np.array([[0.0, 0.0, 1.0]])
        # 100N normal, 100N tangent exceeds friction (0.8 * 100 = 80N max)
        forces = np.array([[100.0, 0.0, 100.0]])
        velocities = np.zeros((1, 3))

        model.update_from_mujoco(
            positions, normals, forces, velocities, ["hand"], timestamp=0.0
        )

        margins = model.check_slip_margin()

        # Margin should be negative (outside friction cone)
        assert margins["min_margin"] < 0
        assert margins["any_slipping"]

    def test_no_slip_within_friction_cone(self) -> None:
        """Should detect sticking when within friction cone."""
        model = GripContactModel()

        # Create contact well within friction cone
        positions = np.array([[0.0, 0.0, 0.0]])
        normals = np.array([[0.0, 0.0, 1.0]])
        # 100N normal, 50N tangent is within friction (0.8 * 100 = 80N max)
        forces = np.array([[50.0, 0.0, 100.0]])
        velocities = np.zeros((1, 3))

        model.update_from_mujoco(
            positions, normals, forces, velocities, ["hand"], timestamp=0.0
        )

        margins = model.check_slip_margin()

        # Margin should be positive
        assert margins["min_margin"] > 0
        assert not margins["any_slipping"]

    def test_multiple_contact_slip_tracking(self) -> None:
        """Should track slip across multiple contacts."""
        model = GripContactModel()

        # Create contacts: one slipping, one sticking
        positions = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
        normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        # First contact slipping (high tangent), second sticking
        forces = np.array(
            [
                [100.0, 0.0, 100.0],  # Exceeds friction
                [10.0, 0.0, 100.0],  # Within friction
            ]
        )
        velocities = np.zeros((2, 3))

        model.update_from_mujoco(
            positions, normals, forces, velocities, ["hand", "hand"], timestamp=0.0
        )

        state = model.current_state
        assert state is not None
        assert state.num_slipping >= 1
        assert state.num_sticking >= 1
