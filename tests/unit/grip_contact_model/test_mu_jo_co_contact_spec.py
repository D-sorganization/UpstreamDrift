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


class TestMuJoCoContactSpec:
    """Tests for MuJoCo contact specification generation."""

    def test_generates_contact_pairs(self) -> None:
        """Should generate contact pairs for each hand."""
        spec = create_mujoco_grip_contacts()

        assert "contact_pairs" in spec
        assert len(spec["contact_pairs"]) == 2  # left and right hand

    def test_custom_body_names(self) -> None:
        """Should use custom body names."""
        spec = create_mujoco_grip_contacts(
            grip_body_name="my_grip",
            hand_body_names=["custom_hand"],
        )

        assert spec["contact_pairs"][0]["body1"] == "custom_hand"
        assert spec["contact_pairs"][0]["body2"] == "my_grip"

    def test_friction_in_spec(self) -> None:
        """Should include friction in contact pairs."""
        spec = create_mujoco_grip_contacts(friction=(0.9, 0.7, 0.002))

        assert spec["contact_pairs"][0]["friction"] == [0.9, 0.7, 0.002]
