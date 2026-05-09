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


class TestStaticEquilibriumValidation:
    """Tests for static equilibrium validation (Issue #757)."""

    def test_sufficient_support(self) -> None:
        """Should detect sufficient support force."""
        model = GripContactModel()

        # Create contact that supports exactly the club weight
        positions = np.array([[0.0, 0.0, 0.0]])
        normals = np.array([[0.0, 0.0, 1.0]])  # Upward normal
        forces = np.array([[0.0, 0.0, 5.0]])  # 5N upward force
        velocities = np.zeros((1, 3))

        model.update_from_mujoco(
            positions, normals, forces, velocities, ["hand"], timestamp=0.0
        )

        result = model.check_static_equilibrium(club_weight=5.0)

        assert result["equilibrium"]
        assert result["support_ratio"] >= 0.99

    def test_insufficient_support(self) -> None:
        """Should detect insufficient support force."""
        model = GripContactModel()

        # Create contact with less than required support
        positions = np.array([[0.0, 0.0, 0.0]])
        normals = np.array([[0.0, 0.0, 1.0]])
        forces = np.array([[0.0, 0.0, 2.0]])  # Only 2N, need 5N
        velocities = np.zeros((1, 3))

        model.update_from_mujoco(
            positions, normals, forces, velocities, ["hand"], timestamp=0.0
        )

        result = model.check_static_equilibrium(club_weight=5.0)

        assert not result["equilibrium"]
        assert result["support_ratio"] < 0.99
