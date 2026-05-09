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


class TestForceDecomposition:
    """Tests for contact force decomposition."""

    def test_pure_normal_force(self) -> None:
        """Pure normal force should have zero tangent."""
        normal = np.array([0.0, 0.0, 1.0])
        force = np.array([0.0, 0.0, 100.0])

        normal_f, tangent_f = decompose_contact_force(force, normal)

        assert normal_f == pytest.approx(100.0)
        np.testing.assert_allclose(tangent_f, np.zeros(3), atol=1e-10)

    def test_pure_tangent_force(self) -> None:
        """Pure tangential force should have zero normal."""
        normal = np.array([0.0, 0.0, 1.0])
        force = np.array([50.0, 30.0, 0.0])

        normal_f, tangent_f = decompose_contact_force(force, normal)

        assert normal_f == pytest.approx(0.0)
        np.testing.assert_allclose(tangent_f, force, atol=1e-10)

    def test_mixed_force_decomposition(self) -> None:
        """Mixed force should decompose correctly."""
        normal = np.array([0.0, 0.0, 1.0])
        force = np.array([30.0, 40.0, 100.0])

        normal_f, tangent_f = decompose_contact_force(force, normal)

        assert normal_f == pytest.approx(100.0)
        np.testing.assert_allclose(tangent_f, [30.0, 40.0, 0.0], atol=1e-10)
