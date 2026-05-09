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


class TestCenterOfPressure:
    """Tests for center of pressure computation."""

    def test_single_contact_cop(self) -> None:
        """COP for single contact should be at contact position."""
        contacts = [
            ContactPoint(
                position=np.array([1.0, 2.0, 0.0]),
                normal=np.array([0.0, 0.0, 1.0]),
                normal_force=100.0,
                tangent_force=np.zeros(3),
                slip_velocity=np.zeros(3),
                state=ContactState.STICKING,
            )
        ]

        cop = compute_center_of_pressure(contacts)

        np.testing.assert_allclose(cop, [1.0, 2.0, 0.0])

    def test_two_equal_contacts_cop(self) -> None:
        """COP for two equal contacts should be midpoint."""
        contacts = [
            ContactPoint(
                position=np.array([0.0, 0.0, 0.0]),
                normal=np.array([0.0, 0.0, 1.0]),
                normal_force=100.0,
                tangent_force=np.zeros(3),
                slip_velocity=np.zeros(3),
                state=ContactState.STICKING,
            ),
            ContactPoint(
                position=np.array([2.0, 0.0, 0.0]),
                normal=np.array([0.0, 0.0, 1.0]),
                normal_force=100.0,
                tangent_force=np.zeros(3),
                slip_velocity=np.zeros(3),
                state=ContactState.STICKING,
            ),
        ]

        cop = compute_center_of_pressure(contacts)

        np.testing.assert_allclose(cop, [1.0, 0.0, 0.0])

    def test_weighted_cop(self) -> None:
        """COP should be weighted by normal force."""
        contacts = [
            ContactPoint(
                position=np.array([0.0, 0.0, 0.0]),
                normal=np.array([0.0, 0.0, 1.0]),
                normal_force=100.0,  # Larger force
                tangent_force=np.zeros(3),
                slip_velocity=np.zeros(3),
                state=ContactState.STICKING,
            ),
            ContactPoint(
                position=np.array([3.0, 0.0, 0.0]),
                normal=np.array([0.0, 0.0, 1.0]),
                normal_force=50.0,  # Smaller force
                tangent_force=np.zeros(3),
                slip_velocity=np.zeros(3),
                state=ContactState.STICKING,
            ),
        ]

        cop = compute_center_of_pressure(contacts)

        # COP = (100*0 + 50*3) / (100 + 50) = 150/150 = 1.0
        np.testing.assert_allclose(cop, [1.0, 0.0, 0.0])
