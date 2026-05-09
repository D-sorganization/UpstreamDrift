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


class TestGripContactModel:
    """Tests for GripContactModel class."""

    def test_update_from_mujoco(self) -> None:
        """Model should update from MuJoCo data."""
        model = GripContactModel()

        # Simulate 2 contact points
        positions = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
        normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        forces = np.array([[0.0, 0.0, 50.0], [0.0, 0.0, 50.0]])
        velocities = np.zeros((2, 3))
        body_names = ["left_hand", "left_hand"]

        state = model.update_from_mujoco(
            positions, normals, forces, velocities, body_names, timestamp=0.1
        )

        assert state is not None
        assert len(state.contacts) == 2
        assert state.total_normal_force == pytest.approx(100.0)
        assert state.timestamp == 0.1

    def test_static_equilibrium_check(self) -> None:
        """Model should validate static equilibrium."""
        model = GripContactModel()

        # Create contact that can support 5 N club weight
        positions = np.array([[0.0, 0.0, 0.0]])
        normals = np.array([[0.0, 0.0, 1.0]])  # Vertical normal
        forces = np.array([[0.0, 0.0, 5.0]])  # 5 N upward
        velocities = np.zeros((1, 3))
        body_names = ["hand"]

        model.update_from_mujoco(
            positions, normals, forces, velocities, body_names, timestamp=0.0
        )

        result = model.check_static_equilibrium(club_weight=5.0)

        assert result["equilibrium"]
        assert result["support_ratio"] >= 0.99

    def test_slip_margin_calculation(self) -> None:
        """Model should calculate slip margins."""
        model = GripContactModel()

        # Contact well within friction cone
        positions = np.array([[0.0, 0.0, 0.0]])
        normals = np.array([[0.0, 0.0, 1.0]])
        # Pure normal force - zero tangent
        forces = np.array([[0.0, 0.0, 100.0]])
        velocities = np.zeros((1, 3))

        model.update_from_mujoco(
            positions, normals, forces, velocities, ["hand"], timestamp=0.0
        )

        margins = model.check_slip_margin()

        assert margins["min_margin"] == pytest.approx(1.0)  # Full margin available
        assert not margins["any_slipping"]
