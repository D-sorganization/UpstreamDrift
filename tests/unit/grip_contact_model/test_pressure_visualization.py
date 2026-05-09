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


class TestPressureVisualization:
    """Tests for pressure distribution visualization (Issue #757)."""

    def test_empty_contacts(self) -> None:
        """Should handle empty contact list."""
        data = compute_pressure_visualization(
            contacts=[],
            grip_center=np.zeros(3),
        )

        assert isinstance(data, PressureVisualizationData)
        assert len(data.pressures) == 0
        assert data.max_pressure == 0.0

    def test_single_contact_pressure(self) -> None:
        """Should compute pressure for single contact."""
        contacts = [
            ContactPoint(
                position=np.array([0.0, 0.0, 0.05]),
                normal=np.array([1.0, 0.0, 0.0]),
                normal_force=100.0,
                tangent_force=np.zeros(3),
                slip_velocity=np.zeros(3),
                state=ContactState.STICKING,
            )
        ]

        data = compute_pressure_visualization(
            contacts=contacts,
            grip_center=np.zeros(3),
            contact_area=0.01,  # 0.01 m²
        )

        # Pressure = Force / Area = 100 / 0.01 = 10000 Pa
        assert data.max_pressure == pytest.approx(10000.0)
        assert len(data.normalized_pressures) == 1
        assert data.normalized_pressures[0] == pytest.approx(
            1.0
        )  # Max is normalized to 1

    def test_multiple_contacts_pressure(self) -> None:
        """Should compute pressure distribution for multiple contacts."""
        contacts = [
            ContactPoint(
                position=np.array([0.0, 0.0, 0.0]),
                normal=np.array([1.0, 0.0, 0.0]),
                normal_force=100.0,
                tangent_force=np.zeros(3),
                slip_velocity=np.zeros(3),
                state=ContactState.STICKING,
            ),
            ContactPoint(
                position=np.array([0.0, 0.0, 0.1]),
                normal=np.array([1.0, 0.0, 0.0]),
                normal_force=50.0,  # Half the force
                tangent_force=np.zeros(3),
                slip_velocity=np.zeros(3),
                state=ContactState.STICKING,
            ),
        ]

        data = compute_pressure_visualization(
            contacts=contacts,
            grip_center=np.array([0.0, 0.0, 0.05]),
            contact_area=0.01,
        )

        assert len(data.pressures) == 2
        assert data.max_pressure > data.mean_pressure
        # First contact has higher pressure
        assert data.normalized_pressures[0] > data.normalized_pressures[1]

    def test_angular_positions(self) -> None:
        """Should compute angular positions around grip axis."""
        contacts = [
            ContactPoint(
                position=np.array([0.01, 0.0, 0.0]),  # Right side
                normal=np.array([1.0, 0.0, 0.0]),
                normal_force=50.0,
                tangent_force=np.zeros(3),
                slip_velocity=np.zeros(3),
                state=ContactState.STICKING,
            ),
            ContactPoint(
                position=np.array([-0.01, 0.0, 0.0]),  # Left side
                normal=np.array([-1.0, 0.0, 0.0]),
                normal_force=50.0,
                tangent_force=np.zeros(3),
                slip_velocity=np.zeros(3),
                state=ContactState.STICKING,
            ),
        ]

        data = compute_pressure_visualization(
            contacts=contacts,
            grip_center=np.zeros(3),
            grip_axis=np.array([0.0, 0.0, 1.0]),  # Vertical axis
        )

        assert len(data.angular_positions) == 2
        # Should be ~π radians apart
        angle_diff = abs(data.angular_positions[0] - data.angular_positions[1])
        assert angle_diff == pytest.approx(np.pi, abs=0.1)
