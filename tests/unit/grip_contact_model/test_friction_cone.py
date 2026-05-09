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


class TestFrictionCone:
    """Tests for friction cone calculations."""

    def test_zero_tangent_within_cone(self) -> None:
        """Zero tangential force should be within friction cone."""
        result = check_friction_cone(
            normal_force=100.0,
            tangent_force=np.zeros(3),
            friction_coefficient=0.8,
        )
        assert result

    def test_small_tangent_within_cone(self) -> None:
        """Small tangential force should be within cone."""
        # μ * F_n = 0.8 * 100 = 80 N
        result = check_friction_cone(
            normal_force=100.0,
            tangent_force=np.array([50.0, 0.0, 0.0]),  # 50 N < 80 N
            friction_coefficient=0.8,
        )
        assert result

    def test_large_tangent_outside_cone(self) -> None:
        """Large tangential force should be outside cone (slipping)."""
        # μ * F_n = 0.8 * 100 = 80 N
        result = check_friction_cone(
            normal_force=100.0,
            tangent_force=np.array([100.0, 0.0, 0.0]),  # 100 N > 80 N
            friction_coefficient=0.8,
        )
        assert not result
